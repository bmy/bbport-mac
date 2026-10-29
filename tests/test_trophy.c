// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* Standalone test for the local trophy store (src/runtime_trophy.c only).
 * Usage: trophy-test [temp_dir]   (default: out/trophy_test_tmp) */
#include "runtime_trophy.h"
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#define make_dir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define make_dir(p) mkdir(p,0755)
#endif

#define THREADS 8
#define PER_THREAD 50
static char dir[1024], file[1100];

/* Checks one line against the exact v1 layout; returns the trophy id or -1. */
static int parse_line(const char *line, const char *title) {
    char got[32], stamp[32], tail[8];
    int id=-1;
    if (sscanf(line,"{\"v\":1,\"title_id\":\"%31[A-Z0-9]\",\"trophy_id\":%d,\"unlocked_at\":\"%31[0-9TZ:-]\"%7s",
               got,&id,stamp,tail)!=4) return -1;
    if (strcmp(got,title) || strcmp(tail,"}") || strlen(stamp)!=20 || stamp[10]!='T' || stamp[19]!='Z') return -1;
    return id;
}

/* Reads the file, validates every line and returns the count; ids[] collects the trophy ids. */
static int check_file(const char *title, int *ids, int max) {
    FILE *f=fopen(file,"rb");
    assert(f);
    char line[256];
    int count=0;
    while (fgets(line,sizeof(line),f)) {
        size_t n=strlen(line);
        assert(n>1 && line[n-1]=='\n');
        line[n-1]=0;
        int id=parse_line(line,title);
        assert(id>=0 && id<=255);
        assert(count<max);
        ids[count++]=id;
    }
    fclose(f);
    return count;
}

static void *writer(void *arg) {
    int base=(int)(size_t)arg;
    for (int i=0;i<PER_THREAD;++i) assert(trophy_store_append(dir,"CUSA03173",(base*PER_THREAD+i)%256)==0);
    return NULL;
}

int main(int argc,char **argv) {
    snprintf(dir,sizeof(dir),"%s",argc>1 ? argv[1] : "out/trophy_test_tmp");
    snprintf(file,sizeof(file),"%s/" TROPHY_STORE_FILE,dir);
    if (make_dir(dir) && errno!=EEXIST) { perror(dir); return 1; }
    remove(file);

    /* Three events, three lines, in order. */
    int ids[THREADS*PER_THREAD];
    assert(trophy_store_append(dir,"CUSA03173",0)==0);
    assert(trophy_store_append(dir,"CUSA03173",12)==0);
    assert(trophy_store_append(dir,"CUSA03173",255)==0);
    assert(check_file("CUSA03173",ids,3)==3 && ids[0]==0 && ids[1]==12 && ids[2]==255);
    puts("PASS: three events appended");

    /* Invalid input writes nothing. */
    assert(trophy_store_append(dir,"CUSA\"03173",1)==-1);
    assert(trophy_store_append(dir,"",1)==-1);
    assert(trophy_store_append(dir,NULL,1)==-1);
    assert(trophy_store_append(dir,"cusa03173",1)==-1);
    assert(trophy_store_append(dir,"ABC",1)==-1);
    assert(trophy_store_append(dir,"ABCDEFGHIJKLMNOPQ",1)==-1);
    assert(trophy_store_append(dir,"CUSA03173",-1)==-1);
    assert(trophy_store_append(dir,"CUSA03173",256)==-1);
    assert(trophy_store_append(NULL,"CUSA03173",1)==-1);
    assert(trophy_store_append("",  "CUSA03173",1)==-1);
    assert(check_file("CUSA03173",ids,3)==3);
    puts("PASS: invalid title_id / trophy_id rejected");

    /* A missing directory fails without aborting. */
    char missing[1100];
    snprintf(missing,sizeof(missing),"%s/does-not-exist",dir);
    assert(trophy_store_append(missing,"CUSA03173",1)==-1);
    puts("PASS: unwritable directory reported");

    /* Title configuration: fallback, valid override, invalid values ignored. */
    assert(!strcmp(trophy_store_title_id(),TROPHY_STORE_DEFAULT_TITLE));
    trophy_store_configure("CUSA\"X");
    trophy_store_configure(NULL);
    assert(!strcmp(trophy_store_title_id(),TROPHY_STORE_DEFAULT_TITLE));
    trophy_store_configure("CUSA00900");
    assert(!strcmp(trophy_store_title_id(),"CUSA00900"));
    puts("PASS: title configuration");

    /* Concurrent writers: no lost or interleaved lines. */
    remove(file);
    pthread_t threads[THREADS];
    for (int i=0;i<THREADS;++i) assert(!pthread_create(&threads[i],NULL,writer,(void *)(size_t)i));
    for (int i=0;i<THREADS;++i) assert(!pthread_join(threads[i],NULL));
    int count=check_file("CUSA03173",ids,THREADS*PER_THREAD);
    assert(count==THREADS*PER_THREAD);
    int seen[256]={0};
    for (int i=0;i<count;++i) ++seen[ids[i]];
    for (int i=0;i<THREADS*PER_THREAD;++i) assert(seen[i%256]>0);
    printf("PASS: %d concurrent events, %d valid lines\n",THREADS*PER_THREAD,count);

    remove(file);
    puts("PASS: trophy store");
    return 0;
}
