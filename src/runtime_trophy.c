// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* Local trophy store (format v1, one event per line, no deduplication):
 *   {"v":1,"title_id":"CUSA03173","trophy_id":12,"unlocked_at":"2026-10-10T06:30:00Z"}
 * The file is opened in append mode for every event and each line is written with a single
 * fwrite under a process-wide mutex, so concurrent unlocks never interleave. Failures are
 * reported through the return value only: the caller logs them and the game keeps running. */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L /* gmtime_r */
#endif
#include "runtime_trophy.h"
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static pthread_mutex_t store_lock=PTHREAD_MUTEX_INITIALIZER;
static char configured_title[17];

/* [A-Z0-9]{4,16}: the only characters that reach the JSON line unescaped. */
static int valid_title(const char *title_id) {
    if (!title_id) return 0;
    size_t n=strlen(title_id);
    if (n<4 || n>16) return 0;
    for (size_t i=0;i<n;++i) {
        char c=title_id[i];
        if (!((c>='A' && c<='Z') || (c>='0' && c<='9'))) return 0;
    }
    return 1;
}

static int utc_now(char out[21]) {
    time_t now=time(NULL);
    struct tm tm;
    if (now==(time_t)-1) return -1;
#ifdef _WIN32
    if (gmtime_s(&tm,&now)) return -1;
#else
    if (!gmtime_r(&now,&tm)) return -1;
#endif
    return strftime(out,21,"%Y-%m-%dT%H:%M:%SZ",&tm)==20 ? 0 : -1;
}

int trophy_store_append(const char *user_dir, const char *title_id, int32_t trophy_id) {
    if (!user_dir || !*user_dir || !valid_title(title_id) || trophy_id<0 || trophy_id>255) return -1;
    char path[4096], stamp[21], line[128];
    int n=snprintf(path,sizeof(path),"%s/" TROPHY_STORE_FILE,user_dir);
    if (n<0 || (size_t)n>=sizeof(path) || utc_now(stamp)) return -1;
    n=snprintf(line,sizeof(line),"{\"v\":1,\"title_id\":\"%s\",\"trophy_id\":%d,\"unlocked_at\":\"%s\"}\n",
               title_id,(int)trophy_id,stamp);
    if (n<0 || (size_t)n>=sizeof(line)) return -1;
    if (pthread_mutex_lock(&store_lock)) return -1;
    int result=-1;
    FILE *f=fopen(path,"ab");
    if (f) {
        if (fwrite(line,1,(size_t)n,f)==(size_t)n && !fflush(f)) result=0;
        if (fclose(f)) result=-1;
    }
    pthread_mutex_unlock(&store_lock);
    return result;
}

void trophy_store_configure(const char *title_id) {
    if (!valid_title(title_id)) return;
    pthread_mutex_lock(&store_lock);
    memcpy(configured_title,title_id,strlen(title_id)+1);
    pthread_mutex_unlock(&store_lock);
}

const char *trophy_store_title_id(void) {
    return configured_title[0] ? configured_title : TROPHY_STORE_DEFAULT_TITLE;
}
