/* Host-OS differences for the loader and runtime: Linux (the original target) and macOS
 * (x86-64 under Rosetta 2). Everything here is a no-op wrapper on Linux. */
#ifndef BB_PLATFORM_H
#define BB_PLATFORM_H
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#ifdef __APPLE__
#include <stdlib.h>
#else
#include <sys/random.h>
#endif

#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0 /* macOS: anonymous and shared mappings are lazily backed anyway */
#endif

/* Guest address space. macOS on Apple Silicon (Rosetta) keeps the commpage at
 * 0xFC0000000-0xFFFFFFFFF and a GPU carve-out at 0x1000000000-0x6FFFFFFFFF, so host-owned
 * low memory ends below the commpage and guest user memory starts above the carve-out
 * (the same split upstream shadPS4 uses on macOS). */
#ifdef __APPLE__
#define BB_LOW_MAX UINT64_C(0x0FC0000000)
#define BB_USER_MIN UINT64_C(0x7000000000)
#else
#define BB_LOW_MAX UINT64_C(0x1000000000)
#define BB_USER_MIN UINT64_C(0x1000000000)
#endif

/* Signal context registers. */
#ifdef __APPLE__
#define BB_CTX_RIP(uc) ((uc)->uc_mcontext->__ss.__rip)
#define BB_CTX_RBP(uc) ((uc)->uc_mcontext->__ss.__rbp)
#define BB_CTX_RDI(uc) ((uc)->uc_mcontext->__ss.__rdi)
#else
#define BB_CTX_RIP(uc) ((uc)->uc_mcontext.gregs[REG_RIP])
#define BB_CTX_RBP(uc) ((uc)->uc_mcontext.gregs[REG_RBP])
#define BB_CTX_RDI(uc) ((uc)->uc_mcontext.gregs[REG_RDI])
#endif

/* Need _GNU_SOURCE on Linux (gettid, pthread_setname_np); only those files use them. */
#if defined(__APPLE__) || defined(_GNU_SOURCE)
static inline uint64_t bb_thread_id(void) {
#ifdef __APPLE__
    uint64_t id = 0;
    pthread_threadid_np(NULL, &id);
    return id;
#else
    return (uint64_t)gettid();
#endif
}

static inline void bb_set_thread_name(const char *name) {
#ifdef __APPLE__
    pthread_setname_np(name);
#else
    pthread_setname_np(pthread_self(), name);
#endif
}
#endif

#ifdef __APPLE__
/* macOS lacks the POSIX timed lock functions and per-condvar clocks. The timed locks poll
 * with short sleeps (guest timed locks are rare and short); semaphores wait relative to a
 * CLOCK_MONOTONIC deadline. */
static inline int bb_deadline_passed(const struct timespec *abs_realtime) {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return now.tv_sec > abs_realtime->tv_sec ||
           (now.tv_sec == abs_realtime->tv_sec && now.tv_nsec >= abs_realtime->tv_nsec);
}
static inline int bb_mutex_timedlock(pthread_mutex_t *m, const struct timespec *abs_realtime) {
    for (;;) {
        int e = pthread_mutex_trylock(m);
        if (e != EBUSY) return e;
        if (bb_deadline_passed(abs_realtime)) return ETIMEDOUT;
        usleep(100);
    }
}
static inline int bb_rwlock_timedlock(pthread_rwlock_t *r, int writer, const struct timespec *abs_realtime) {
    for (;;) {
        int e = writer ? pthread_rwlock_trywrlock(r) : pthread_rwlock_tryrdlock(r);
        if (e != EBUSY) return e;
        if (bb_deadline_passed(abs_realtime)) return ETIMEDOUT;
        usleep(100);
    }
}
#else
static inline int bb_mutex_timedlock(pthread_mutex_t *m, const struct timespec *abs_realtime) {
    return pthread_mutex_timedlock(m, abs_realtime);
}
static inline int bb_rwlock_timedlock(pthread_rwlock_t *r, int writer, const struct timespec *abs_realtime) {
    return writer ? pthread_rwlock_timedwrlock(r, abs_realtime) : pthread_rwlock_timedrdlock(r, abs_realtime);
}
#endif

/* Condition variable whose timed waits use a CLOCK_MONOTONIC deadline (nanoseconds). */
static inline int bb_cond_init_monotonic(pthread_cond_t *c) {
#ifdef __APPLE__
    return pthread_cond_init(c, NULL);
#else
    pthread_condattr_t attr;
    int e = pthread_condattr_init(&attr);
    if (!e) e = pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
    if (!e) e = pthread_cond_init(c, &attr);
    pthread_condattr_destroy(&attr);
    return e;
#endif
}
static inline int bb_cond_wait_until(pthread_cond_t *c, pthread_mutex_t *m, uint64_t deadline_ns) {
#ifdef __APPLE__
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t now_ns = (uint64_t)now.tv_sec * 1000000000u + (uint64_t)now.tv_nsec;
    if (now_ns >= deadline_ns) return ETIMEDOUT;
    uint64_t left = deadline_ns - now_ns;
    struct timespec rel = {.tv_sec = (time_t)(left / 1000000000u), .tv_nsec = (long)(left % 1000000000u)};
    return pthread_cond_timedwait_relative_np(c, m, &rel);
#else
    struct timespec end = {.tv_sec = (time_t)(deadline_ns / 1000000000u), .tv_nsec = (long)(deadline_ns % 1000000000u)};
    return pthread_cond_timedwait(c, m, &end);
#endif
}

/* Sleep until an absolute CLOCK_MONOTONIC time. */
static inline void bb_sleep_until(const struct timespec *abs_monotonic) {
#ifdef __APPLE__
    for (;;) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        int64_t left = (int64_t)(abs_monotonic->tv_sec - now.tv_sec) * 1000000000 + (abs_monotonic->tv_nsec - now.tv_nsec);
        if (left <= 0) return;
        struct timespec rel = {.tv_sec = (time_t)(left / 1000000000), .tv_nsec = (long)(left % 1000000000)};
        if (!nanosleep(&rel, NULL)) return;
    }
#else
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, abs_monotonic, NULL)) {}
#endif
}

static inline int bb_random_bytes(void *out, size_t size) {
#ifdef __APPLE__
    arc4random_buf(out, size);
    return 0;
#else
    return getrandom(out, size, 0) < 0 ? -1 : 0;
#endif
}

/* Maps fresh anonymous memory exactly at `address`, failing (MAP_FAILED) if anything is there. */
static inline void *bb_map_noreplace(void *address, size_t size, int prot) {
#ifdef MAP_FIXED_NOREPLACE
    return mmap(address, size, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
#else
    void *p = mmap(address, size, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED && p != address) { munmap(p, size); p = MAP_FAILED; }
    return p;
#endif
}
#endif
