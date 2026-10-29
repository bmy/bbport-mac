// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* bbport: the game's files read from an Android phone over adb, each fetched when the game first
 * opens it, instead of a full copy beforehand (BB_APP0_ADB=<the game's folder on the phone>, e.g.
 * /sdcard/bloodborne/CUSA03173; BB_ADB: adb's path, default "adb").
 * The app0 directory is the local cache. At start the phone's file list (one find) becomes empty
 * placeholders there, so directory listings and case matching work as with a copy; a file whose
 * local size differs from the phone's is fetched whole (adb exec-out cat) before it is opened, and
 * stat reports the phone's size without fetching. eboot.bin, sce_module and sce_sys are read before
 * the runtime starts (prepare.py, param.sfo): the launcher pulls them first. */
#define _GNU_SOURCE
#include "runtime.h"
#include "../gpu/shim/bb_env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <winsock2.h>
typedef SOCKET Socket;
#define NO_SOCKET INVALID_SOCKET
#define close_socket closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
typedef int Socket;
#define NO_SOCKET (-1)
#define close_socket close
#endif

typedef struct { char *rel; int64_t size; int ready; } Entry;
static Entry *entries;
static size_t entry_count, slots; /* slots: a power of two, open addressing */
static char root[512], remote[512], adb[512];
static size_t root_len;
static pthread_mutex_t fetch_lock=PTHREAD_MUTEX_INITIALIZER;
static unsigned fetches;
static uint64_t fetched_bytes;
static double fetch_seconds;

static uint64_t hash(const char *s) {
    uint64_t h=1469598103934665603ULL;
    for (; *s; ++s) h=(h^(unsigned char)tolower((unsigned char)*s))*1099511628211ULL;
    return h;
}
static Entry *find(const char *rel) {
    if (!slots) return NULL;
    for (size_t i=hash(rel)&(slots-1);; i=(i+1)&(slots-1)) {
        if (!entries[i].rel) return NULL;
        if (!strcasecmp(entries[i].rel,rel)) return &entries[i];
    }
}
static void insert(const char *rel,int64_t size) {
    size_t i=hash(rel)&(slots-1);
    while (entries[i].rel) i=(i+1)&(slots-1);
    entries[i]=(Entry){strdup(rel),size,0};
    ++entry_count;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* popen of an adb command: Windows runs it through cmd /c, which strips one pair of outer quotes. */
static FILE *adb_run(const char *args) {
    char command[2048];
#ifdef _WIN32
    if ((size_t)snprintf(command,sizeof(command),"\"\"%s\" %s\"",adb,args)>=sizeof(command)) return NULL;
    return popen(command,"rb");
#else
    if ((size_t)snprintf(command,sizeof(command),"\"%s\" %s",adb,args)>=sizeof(command)) return NULL;
    return popen(command,"r");
#endif
}
/* adb's file sync protocol straight to the adb server (127.0.0.1:5037, started by the list's adb
 * run): one connection for every file, RECV <path> -> DATA chunks, DONE. A process a file cost
 * ~0.45 s (cmd, adb, the phone's shell): small files streamed at ~1.5 MB/s instead of ~35. */
static Socket sync_socket=NO_SOCKET;
static int send_all(const void *data,size_t size) {
    for (const char *p=data; size;) {
        const int n=send(sync_socket,p,(int)size,0);
        if (n<=0) return 0;
        p+=n; size-=(size_t)n;
    }
    return 1;
}
static int receive_all(void *data,size_t size) {
    for (char *p=data; size;) {
        const int n=recv(sync_socket,p,(int)size,0);
        if (n<=0) return 0;
        p+=n; size-=(size_t)n;
    }
    return 1;
}
static uint32_t le32(const unsigned char *p) { return p[0]|p[1]<<8|p[2]<<16|(uint32_t)p[3]<<24; }
static void sync_close(void) { if (sync_socket!=NO_SOCKET) close_socket(sync_socket); sync_socket=NO_SOCKET; }
static int host_request(const char *request) {
    char line[256], status[4];
    const int n=snprintf(line,sizeof(line),"%04x%s",(unsigned)strlen(request),request);
    return send_all(line,(size_t)n) && receive_all(status,4) && !memcmp(status,"OKAY",4);
}
static int sync_open(void) {
    if (sync_socket!=NO_SOCKET) return 1;
#ifdef _WIN32
    static int started;
    if (!started) { WSADATA data; if (WSAStartup(MAKEWORD(2,2),&data)) return 0; started=1; }
#endif
    sync_socket=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
    if (sync_socket==NO_SOCKET) return 0;
    struct sockaddr_in a={0};
    a.sin_family=AF_INET; a.sin_port=htons(5037); a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    const char *serial=getenv("ANDROID_SERIAL");
    char transport[128];
    snprintf(transport,sizeof(transport),serial && *serial ? "host:transport:%s" : "host:transport-any",serial);
    if (connect(sync_socket,(struct sockaddr *)&a,sizeof(a)) || !host_request(transport) || !host_request("sync:")) {
        sync_close(); return 0;
    }
    return 1;
}
/* The file to out: its size, or -1 (the connection is then closed: the next fetch opens another). */
static int64_t sync_receive(const char *path,FILE *out) {
    const size_t n=strlen(path);
    unsigned char head[8]={'R','E','C','V',(unsigned char)n,(unsigned char)(n>>8),(unsigned char)(n>>16),(unsigned char)(n>>24)};
    if (!sync_open() || !send_all(head,8) || !send_all(path,n)) { sync_close(); return -1; }
    static char buffer[65536];
    for (int64_t total=0;;) {
        if (!receive_all(head,8)) break;
        const uint32_t size=le32(head+4);
        if (!memcmp(head,"DONE",4)) return total;
        if (memcmp(head,"DATA",4) || size>sizeof(buffer)) {
            if (!memcmp(head,"FAIL",4) && size<sizeof(buffer) && receive_all(buffer,size))
                printf("Remote: adb: %.*s\n",(int)size,buffer);
            break;
        }
        if (!receive_all(buffer,size) || fwrite(buffer,1,size,out)!=size) break;
        total+=size;
    }
    sync_close();
    return -1;
}
static void make_parents(char *path) {
    for (char *p=path+root_len+1; *p; ++p) if (*p=='/') { *p=0; mkdir(path,0755); *p='/'; }
}
/* The phone's list, "<size> ./<path>" a line; kept in the cache for a start without the phone. */
static char *read_list(size_t *length) {
    char args[1200], saved[600];
    snprintf(saved,sizeof(saved),"%s/.bbremote_list",root);
    snprintf(args,sizeof(args),"exec-out \"cd '%s' && find . -type f -exec stat -c '%%s %%n' {} +\"",remote);
    FILE *in=adb_run(args);
    char *text=NULL; size_t n=0, cap=0;
    if (in) {
        char buffer[65536];
        for (size_t r; (r=fread(buffer,1,sizeof(buffer),in))>0; n+=r) {
            if (n+r+1>cap) { cap=(n+r+1)*2; text=realloc(text,cap); if (!text) { pclose(in); return NULL; } }
            memcpy(text+n,buffer,r);
        }
        pclose(in);
    }
    if (n && text) {
        text[n]=0;
        FILE *out=fopen(saved,"wb");
        if (out) { fwrite(text,1,n,out); fclose(out); }
        *length=n; return text;
    }
    free(text);
    printf("Remote: no list from the phone (adb %s); using the saved one\n",adb);
    FILE *f=fopen(saved,"rb");
    if (!f) return NULL;
    fseek(f,0,SEEK_END); long size=ftell(f); fseek(f,0,SEEK_SET);
    text=size>0 ? malloc((size_t)size+1) : NULL;
    if (text && fread(text,1,(size_t)size,f)==(size_t)size) { text[size]=0; *length=(size_t)size; }
    else { free(text); text=NULL; }
    fclose(f);
    return text;
}

void runtime_remote_configure(const char *app0) {
    const char *r=bb_env("BB_APP0_ADB");
    if (!r || !*r || entries) return;
    const char *a=bb_env("BB_ADB");
    snprintf(adb,sizeof(adb),"%s",a && *a ? a : "adb");
    snprintf(remote,sizeof(remote),"%s",r);
    snprintf(root,sizeof(root),"%s",app0);
    for (char *p=root; *p; ++p) if (*p=='\\') *p='/';
    root_len=strlen(root);
    while (root_len && root[root_len-1]=='/') root[--root_len]=0;
    mkdir(root,0755);
    double start=now();
    size_t length=0;
    char *text=read_list(&length);
    if (!text) { printf("Remote: no file list for %s; the game reads only %s\n",remote,root); return; }
    size_t lines=1;
    for (size_t i=0;i<length;++i) lines+=text[i]=='\n';
    for (slots=1024; slots<lines*2; slots*=2) {}
    entries=calloc(slots,sizeof(Entry));
    if (!entries) { free(text); slots=0; return; }
    size_t created=0, ready=0;
    for (char *line=text, *next; line && *line; line=next) {
        next=strchr(line,'\n');
        if (next) *next++=0;
        size_t n=strlen(line);
        if (n && line[n-1]=='\r') line[--n]=0;
        char *name=strchr(line,' ');
        if (!name || strncmp(name+1,"./",2)) continue;
        int64_t size=strtoll(line,NULL,10);
        const char *rel=name+3;
        if (find(rel)) continue;
        insert(rel,size);
        char path[1024];
        if ((size_t)snprintf(path,sizeof(path),"%s/%s",root,rel)>=sizeof(path)) continue;
        struct stat s;
        if (stat(path,&s)) {
            make_parents(path);
            FILE *f=fopen(path,"wb");
            if (f) { fclose(f); ++created; }
            if (!size) find(rel)->ready=1;
        } else if (s.st_size==size) { find(rel)->ready=1; ++ready; }
    }
    free(text);
    printf("Remote: %zu files on the phone (%s), %zu already here, %zu placeholders made, %.1f s\n",
           entry_count,remote,ready,created,now()-start);
}

/* The entry of a host path in the cache, NULL when it is not one. */
static Entry *entry_of(const char *host) {
    if (!entries) return NULL;
    /* --app0 keeps its backslashes on Windows; the root here has slashes */
    for (size_t i=0; i<root_len; ++i)
        if (host[i]!=root[i] && !(host[i]=='\\' && root[i]=='/')) return NULL;
    if (host[root_len]!='/' && host[root_len]!='\\') return NULL;
    char rel[1024];
    snprintf(rel,sizeof(rel),"%s",host+root_len+1);
    for (char *p=rel; *p; ++p) if (*p=='\\') *p='/';
    return find(rel);
}
int runtime_remote_size(const char *host,int64_t *size) {
    Entry *e=entry_of(host);
    if (!e) return 0;
    *size=e->size;
    return 1;
}
/* Fetches the file of a host path when it is the phone's and not here yet: 0 or a host errno. */
int runtime_remote_fetch(const char *host) {
    Entry *e=entry_of(host);
    if (!e || __atomic_load_n(&e->ready,__ATOMIC_ACQUIRE)) return 0;
    pthread_mutex_lock(&fetch_lock);
    int result=0;
    if (!e->ready) {
        char path[1024], temp[1100], args[1300];
        snprintf(path,sizeof(path),"%s/%s",root,e->rel);
        snprintf(temp,sizeof(temp),"%s.bbtmp",path);
        snprintf(args,sizeof(args),"exec-out \"cat '%s/%s'\"",remote,e->rel);
        const double start=now();
        char full[1100];
        snprintf(full,sizeof(full),"%s/%s",remote,e->rel);
        FILE *out=fopen(temp,"wb"), *in=NULL;
        int64_t n=out ? sync_receive(full,out) : -1;
        if (out && n!=e->size) { /* the sync connection failed: through adb, a process a file */
            fclose(out);
            in=adb_run(args); out=in ? fopen(temp,"wb") : NULL; n=0;
        }
        if (in && out) {
            static char buffer[1<<20];
            for (size_t r; (r=fread(buffer,1,sizeof(buffer),in))>0; n+=(int64_t)r)
                if (fwrite(buffer,1,r,out)!=r) { n=-1; break; }
        }
        if (out && fclose(out)) n=-1;
        if (in) pclose(in);
        if (n==e->size) {
#ifdef _WIN32
            unlink(path); /* Windows' rename does not replace */
#endif
            if (rename(temp,path)) result=errno ? errno : EIO;
        } else result=EIO;
        if (result) {
            unlink(temp);
            printf("Remote: fetching %s failed (%lld of %lld bytes)\n",e->rel,(long long)n,(long long)e->size);
        } else {
            const double took=now()-start;
            __atomic_store_n(&e->ready,1,__ATOMIC_RELEASE);
            ++fetches; fetched_bytes+=(uint64_t)n; fetch_seconds+=took;
            if (e->size>=(8<<20) || took>=1.0 || fetches%100==0)
                printf("Remote: %s, %.1f MB in %.2f s (%u files, %.0f MB, %.1f MB/s so far)\n",e->rel,n/1048576.0,took,
                       fetches,fetched_bytes/1048576.0,fetch_seconds>0 ? fetched_bytes/1048576.0/fetch_seconds : 0.0);
        }
    }
    pthread_mutex_unlock(&fetch_lock);
    return result;
}
