// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* Offline AppContent provider for the configured base-game profile.
 * No package/DLC mounting, downloads, entitlement or license emulation; add-on folders in
 * <user>/addcont are reported as installed (see addon_list). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "runtime.h"
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#define INVALID_ID ((int32_t)0x805a1000)
#define NOT_LOADED ((int32_t)0x805a1001)
#define PARAMETER ((int32_t)0x80d90002)
#define BUSY ((int32_t)0x80d90003)
static unsigned refs, initialized, configured;
static uint32_t parameters[5];
static unsigned queries, lists;
void runtime_content_configure(const uint32_t values[5]) {
    if (refs || initialized || (values[0]!=1 && values[0]!=3)) {
        fputs("ERROR: invalid or late content profile\n",stderr); exit(1);
    }
    memcpy(parameters,values,sizeof(parameters)); configured=1;
}
static void require_provider(void) {
    if (!configured) { fputs("STOP: AppContent needs --content-profile\n",stderr); exit(21); }
}
/* System modules: every library this eboot imports is provided by the host
 * runtime, so loading one only tracks a reference count. AppContent (0xb4)
 * additionally requires the configured content profile. */
static unsigned module_refs[0x10000];
static ABI int32_t module_load(uint16_t id) {
    if (!id) return INVALID_ID;
    if (id==0xb4) require_provider();
    if (module_refs[id]==UINT32_MAX) { fputs("STOP: module reference overflow\n",stderr); exit(21); }
    ++module_refs[id];
    if (id==0xb4) { refs=module_refs[id]; printf("Runtime: AppContent host provider loaded; references=%u\n",refs); }
    else printf("Runtime: system module 0x%x loaded (host implementation)\n",id);
    return 0;
}
static ABI int32_t module_loaded(uint16_t id) {
    if (!id) return INVALID_ID;
    return module_refs[id] ? 0 : NOT_LOADED;
}
static ABI int32_t module_unload(uint16_t id) {
    int32_t result=module_loaded(id);
    if (result) return result;
    --module_refs[id];
    if (id==0xb4) { refs=module_refs[id]; if (!refs) initialized=0; }
    return 0;
}
static void require_loaded(void) {
    require_provider();
    if (!refs) { fputs("STOP: AppContent called before module load\n",stderr); exit(21); }
}
static ABI int32_t content_init(const unsigned char *init,unsigned char *boot) {
    require_loaded();
    if (!init || !boot) return PARAMETER;
    if (initialized) return BUSY;
    /* Only the reserved-zero init profile is implemented. */
    for (unsigned i=0;i<32;++i) if (init[i]) return PARAMETER;
    memset(boot,0,40); initialized=1;
    puts("Runtime: AppContent initialized; offline base-game profile, no mounted add-ons");
    return 0;
}
static void require_initialized(void) {
    require_loaded();
    if (!initialized) { fputs("STOP: AppContent query before initialization\n",stderr); exit(21); }
}
static ABI int32_t param_int(uint32_t id,int32_t *out) {
    require_initialized();
    if (id>4 || !out) return PARAMETER;
    memcpy(out,&parameters[id],4); ++queries;
    printf("Runtime: AppContent parameter %u = %d\n",id,*out);
    return 0;
}
/* SceAppContentAddcontInfo: entitlement label (16 chars + NUL), padding, download status. */
#define LABEL_SIZE 17
#define ADDON_INSTALLED 4u
typedef struct { char label[LABEL_SIZE]; char padding[3]; uint32_t status; } AddonInfo;
/* Add-ons are the folders <user>/addcont/<title id>/<entitlement label> (shadPS4's layout),
 * each reported as installed. No add-on data is mounted: Bloodborne: The Old Hunters
 * (SPEXPANSIONDLC03) needs only the entitlement, its data ships with patch 1.09. Without a list
 * the count is returned, with one at most capacity entries are written. Never touch unused list
 * slots. */
static ABI int32_t addon_list(uint32_t service,void *list,uint32_t capacity,uint32_t *hits) {
    require_initialized();
    if (service) { fputs("STOP: nonzero AppContent service label unsupported\n",stderr); exit(21); }
    if ((!capacity || !list) && !hits) return PARAMETER;
    int fill=list && capacity;
    uint32_t count=0;
    char root[4096];
    snprintf(root,sizeof(root),"%s/addcont",runtime_file_user_dir());
    DIR *titles=opendir(root);
    for (struct dirent *t; titles && (t=readdir(titles));) {
        if (t->d_name[0]=='.') continue;
        int title=openat(dirfd(titles),t->d_name,O_RDONLY|O_DIRECTORY);
        DIR *labels=title>=0 ? fdopendir(title) : NULL;
        if (!labels && title>=0) close(title);
        for (struct dirent *l; labels && (l=readdir(labels));) {
            struct stat entry;
            size_t n=strlen(l->d_name);
            if (l->d_name[0]=='.' || n>=LABEL_SIZE || fstatat(dirfd(labels),l->d_name,&entry,0) ||
                !S_ISDIR(entry.st_mode) || (fill && count>=capacity)) continue;
            if (fill) {
                AddonInfo info;
                memset(&info,'\0',sizeof(info)); memcpy(info.label,l->d_name,n);
                info.status=ADDON_INSTALLED;
                memcpy((unsigned char *)list+(size_t)count*sizeof(info),&info,sizeof(info));
            }
            ++count;
        }
        if (labels) closedir(labels);
    }
    if (titles) closedir(titles);
    if (hits) *hits=count;
    if (count) printf("Runtime: AppContent add-ons reported installed (%s): %u\n",root,(unsigned)count);
    ++lists; return 0;
}
/* The folder of an add-on: <user>/addcont/<any title>/<label> (0: none). */
static int addon_dir(const char *label,char *out,size_t size) {
    char root[4096];
    snprintf(root,sizeof(root),"%s/addcont",runtime_file_user_dir());
    DIR *titles=opendir(root);
    int found=0;
    for (struct dirent *t; !found && titles && (t=readdir(titles));) {
        if (t->d_name[0]=='.') continue;
        struct stat entry;
        if ((size_t)snprintf(out,size,"%s/%s/%s",root,t->d_name,label)>=size) continue;
        found=!stat(out,&entry) && S_ISDIR(entry.st_mode);
    }
    if (titles) closedir(titles);
    return found;
}
#define ADDON_NOT_FOUND ((int32_t)0x80d90005)
/* sceAppContentAddcontMount: the add-on's files appear as /addcont0..15 (read-only data). */
static ABI int32_t addon_mount(uint32_t service,const char *label,char *point) {
    require_initialized();
    (void)service;
    if (!label || !point) return PARAMETER;
    char name[LABEL_SIZE+1]={0}, host[4096];
    memcpy(name,label,LABEL_SIZE);
    if (!addon_dir(name,host,sizeof(host))) {
        printf("Runtime: add-on '%s' not installed\n",name);
        return ADDON_NOT_FOUND;
    }
    static char mounted[16][LABEL_SIZE+1];
    int slot=-1;
    for (int i=0;i<16;++i) {
        if (!strcmp(mounted[i],name)) { slot=i; break; }
        if (slot<0 && !mounted[i][0]) slot=i;
    }
    if (slot<0) return BUSY;
    memcpy(mounted[slot],name,sizeof(name));
    char guest[32];
    snprintf(guest,sizeof(guest),"/addcont%d",slot & 15);
    memset(point,0,16); memcpy(point,guest,strlen(guest));
    if (runtime_file_mount(guest,host)) {
        fprintf(stderr,"Runtime: add-on '%s' could not be mounted: mount table full\n",name);
        return BUSY;
    }
    printf("Runtime: add-on '%s' mounted at %s (%s)\n",name,guest,host);
    return 0;
}
static ABI int32_t addon_unmount(const char *point) { (void)point; return 0; }
/* sceAppContentGetAddcontInfo: an installed add-on. */
static ABI int32_t addon_info(uint32_t service,const char *label,void *info) {
    require_initialized();
    (void)service;
    if (!label || !info) return PARAMETER;
    char name[LABEL_SIZE+1]={0}, host[4096];
    memcpy(name,label,LABEL_SIZE);
    if (!addon_dir(name,host,sizeof(host))) return ADDON_NOT_FOUND;
    AddonInfo out;
    memset(&out,0,sizeof(out)); memcpy(out.label,name,strlen(name));
    out.status=ADDON_INSTALLED;
    memcpy(info,&out,sizeof(out));
    return 0;
}
uintptr_t runtime_content_resolve(const char *name) {
    if (runtime_nid_is(name, "VANhIWcqYak")) return (uintptr_t)addon_mount;
    if (runtime_nid_is(name, "3rHWaV-1KC4")) return (uintptr_t)addon_unmount;
    if (runtime_nid_is(name, "m47juOmH0VE")) return (uintptr_t)addon_info;
    if (runtime_nid_is(name, "g8cM39EUZ6o")) return (uintptr_t)module_load;
    if (runtime_nid_is(name, "fMP5NHUOaMk")) return (uintptr_t)module_loaded;
    if (runtime_nid_is(name, "eR2bZFAAU0Q")) return (uintptr_t)module_unload;
    if (runtime_nid_is(name, "R9lA82OraNs")) return (uintptr_t)content_init;
    if (runtime_nid_is(name, "99b82IKXpH4")) return (uintptr_t)param_int;
    if (runtime_nid_is(name, "xnd8BJzAxmk")) return (uintptr_t)addon_list;
    return 0;
}
void runtime_content_report(void) {
    printf("Runtime: AppContent references=%u, initialized=%u, parameter queries=%u, list queries=%u\n",refs,initialized,queries,lists);
}
