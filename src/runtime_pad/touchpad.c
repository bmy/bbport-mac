// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_pad.c part: touchpad coordinates and clicks, the evdev touchpad SDL does not see (Linux).
 * #included by runtime_pad.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */
static uint16_t touch_axis(float v, int max) {
    return (uint16_t)(v<=0.0f ? 0 : v>=1.0f ? max : (int)(v*max+0.5f));
}
/* A finger on the left or right half of the touchpad (both keys: two fingers). */
static void touch_click(PadData *d, int right) {
    d->buttons|=BTN_TOUCHPAD;
    const uint16_t x=right ? 1440 : 480;
    for (int i=0;i<d->touch_count;++i) if (d->touches[i].x==x) return;
    if (d->touch_count<2) d->touches[d->touch_count++]=(PadTouch){.x=x,.y=471,.id=0};
}

#ifdef __linux__
/* A DualShock 4 / DualSense touchpad SDL does not see (no hidraw access: SDL then reads the pad
 * through evdev, whose touchpad is a separate "... Touchpad" device): read from that device, its
 * fingers and its click, so the game gets the side pressed (from the SrShadowy fork). Looked for
 * at most once a second while missing. BB_EVDEV_TOUCHPAD=0 turns it off. */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/input.h>
#include <sys/ioctl.h>
#include <unistd.h>
/* linux/input.h names the mouse buttons as this file names the d-pad's: those stay the d-pad's. */
#define EVDEV_BTN_LEFT 0x110
#include "../../gpu/shim/bb_env.h"
#undef BTN_LEFT
#undef BTN_RIGHT
static struct {
    int fd, click, finger[2], slot;
    float x[2], y[2];
    struct input_absinfo ax, ay;
    uint64_t last_scan;
} evdev_tp = {.fd=-1};
static void evdev_touchpad_close(void) {
    if (evdev_tp.fd>=0) close(evdev_tp.fd);
    evdev_tp.fd=-1; evdev_tp.click=evdev_tp.finger[0]=evdev_tp.finger[1]=0;
}
static int evdev_touchpad_open(void) {
    DIR *dir=opendir("/dev/input");
    if (!dir) return -1;
    int found=-1;
    struct dirent *e;
    while (found<0 && (e=readdir(dir))) {
        if (strncmp(e->d_name,"event",5)) continue;
        char path[300], name[256]={0};
        snprintf(path,sizeof path,"/dev/input/%s",e->d_name);
        const int fd=open(path,O_RDONLY|O_NONBLOCK|O_CLOEXEC);
        if (fd<0) continue;
        if (ioctl(fd,EVIOCGNAME(sizeof name-1),name)>=0 && SDL_strcasestr(name,"touchpad") &&
            (SDL_strcasestr(name,"Sony") || SDL_strcasestr(name,"Wireless Controller") ||
             SDL_strcasestr(name,"DualShock") || SDL_strcasestr(name,"DualSense")) &&
            ioctl(fd,EVIOCGABS(ABS_MT_POSITION_X),&evdev_tp.ax)>=0 &&
            ioctl(fd,EVIOCGABS(ABS_MT_POSITION_Y),&evdev_tp.ay)>=0) {
            printf("Runtime: controller touchpad read from %s (%s)\n",path,name);
            found=fd;
        } else {
            close(fd);
        }
    }
    closedir(dir);
    return found;
}
static void evdev_touchpad(PadData *d, uint64_t now) {
    static int enabled=-1;
    if (enabled<0) { const char *e=bb_env("BB_EVDEV_TOUCHPAD"); enabled=!(e && e[0]=='0'); }
    if (!enabled) return;
    if (evdev_tp.fd<0) {
        if (evdev_tp.last_scan && now<evdev_tp.last_scan+1000000u) return;
        evdev_tp.last_scan=now;
        if ((evdev_tp.fd=evdev_touchpad_open())<0) return;
    }
    struct input_event ev[32];
    ssize_t n;
    while ((n=read(evdev_tp.fd,ev,sizeof ev))>0) {
        for (int i=0;i<(int)(n/(ssize_t)sizeof ev[0]);++i) {
            const int s=evdev_tp.slot;
            if (ev[i].type==EV_KEY && ev[i].code==EVDEV_BTN_LEFT) evdev_tp.click=ev[i].value!=0;
            else if (ev[i].type!=EV_ABS) continue;
            else if (ev[i].code==ABS_MT_SLOT) evdev_tp.slot=ev[i].value>=0 && ev[i].value<2 ? ev[i].value : 0;
            else if (ev[i].code==ABS_MT_TRACKING_ID && s<2) evdev_tp.finger[s]=ev[i].value>=0;
            else if (ev[i].code==ABS_MT_POSITION_X && s<2) {
                const float r=(float)(evdev_tp.ax.maximum-evdev_tp.ax.minimum);
                evdev_tp.x[s]=r>0 ? (float)(ev[i].value-evdev_tp.ax.minimum)/r : 0.0f;
            } else if (ev[i].code==ABS_MT_POSITION_Y && s<2) {
                const float r=(float)(evdev_tp.ay.maximum-evdev_tp.ay.minimum);
                evdev_tp.y[s]=r>0 ? (float)(ev[i].value-evdev_tp.ay.minimum)/r : 0.0f;
            }
        }
    }
    if (n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK) { evdev_touchpad_close(); return; }
    for (int s=0;s<2 && d->touch_count<2;++s)
        if (evdev_tp.finger[s])
            d->touches[d->touch_count++]=(PadTouch){.x=touch_axis(evdev_tp.x[s],1919),
                .y=touch_axis(evdev_tp.y[s],942),.id=(uint8_t)s};
    if (evdev_tp.click) d->buttons|=BTN_TOUCHPAD;
}
#endif
