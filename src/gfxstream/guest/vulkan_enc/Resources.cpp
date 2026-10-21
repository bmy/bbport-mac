/*
 * Copyright 2018 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "Resources.h"

#include <stdlib.h>

#include "util/detect_os.h"
#include "util/log.h"

#define GOLDFISH_VK_OBJECT_DEBUG 0

#if GOLDFISH_VK_OBJECT_DEBUG
#define D(fmt, ...) ALOGD("%s: " fmt, __func__, ##__VA_ARGS__);
#else
#ifndef D
#define D(fmt, ...)
#endif
#endif

namespace gfxstream {
namespace vk {

void appendObject(struct gfxstream_vk_object_list** begin, void* val) {
    D("for %p", val);
    struct gfxstream_vk_object_list* o = new gfxstream_vk_object_list;
    o->next = nullptr;
    o->obj = val;
    D("new ptr: %p", o);
    if (!*begin) {
        D("first");
        *begin = o;
        return;
    }

    struct gfxstream_vk_object_list* q = *begin;
    struct gfxstream_vk_object_list* p = q;

    while (q) {
        p = q;
        q = q->next;
    }

    D("set next of %p to %p", p, o);
    p->next = o;
}

void eraseObject(struct gfxstream_vk_object_list** begin, void* val) {
    D("for val %p", val);
    if (!*begin) {
        D("val %p notfound", val);
        return;
    }

    struct gfxstream_vk_object_list* q = *begin;
    struct gfxstream_vk_object_list* p = q;

    while (q) {
        struct gfxstream_vk_object_list* n = q->next;
        if (val == q->obj) {
            D("val %p found, delete", val);
            delete q;
            if (*begin == q) {
                D("val %p set begin to %p:", val, n);
                *begin = n;
            } else {
                D("val %p set pnext to %p:", val, n);
                p->next = n;
            }
            return;
        }
        p = q;
        q = n;
    }

    D("val %p notfound after looping", val);
}

void eraseObjects(struct gfxstream_vk_object_list** begin) {
    struct gfxstream_vk_object_list* q = *begin;
    struct gfxstream_vk_object_list* p = q;

    while (q) {
        p = q;
        q = q->next;
        delete p;
    }

    *begin = nullptr;
}

void forAllObjects(struct gfxstream_vk_object_list* begin, std::function<void(void*)> func) {
    struct gfxstream_vk_object_list* q = begin;
    struct gfxstream_vk_object_list* p = q;

    D("call");
    while (q) {
        D("iter");
        p = q;
        q = q->next;
        func(p->obj);
    }
}

}  // namespace vk
}  // namespace gfxstream
