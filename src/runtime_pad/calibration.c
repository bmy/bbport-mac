// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_pad.c part: per-controller stick neutral and travel (biased clones), dead zones; the stick axis in the PS4 scale.
 * #included by runtime_pad.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */
/* Per-controller stick neutral (see the file header): the value each axis holds while the
 * sticks are untouched, taken from a quiet window after the pad opens. Works for any pad; a
 * genuine one settles at 0 and the subtraction is a no-op. */
#define PAD_CAL_QUIET 512              /* an axis is quiet when it moved no more than this */
#define PAD_CAL_QUIET_SAMPLES 8
#define PAD_CAL_TIMEOUT_US 3000000u    /* after this, an all-zero reading is accepted as the neutral */
#define PAD_CAL_BIAS 8192              /* |neutral| past this: a biased neutral and a reduced travel */
#define PAD_CAL_REFINE 90              /* % of the base travel a push must reach to count as the full scale */
#include "../../gpu/shim/bb_env.h"
static int cal_enabled=-1, pad_deadzone=-1, pad_deadzone_outer=-1, cal_manual;
static int cal_have_center, cal_started, cal_quiet_count;
static int cal_center[4], cal_prev[4], cal_manual_center[4];
static int cal_base[4];                /* 0: normal pad; else |neutral| of a biased axis */
static int cal_max[4][2];              /* largest push seen per direction on a biased axis */
static uint64_t cal_since;

static uint64_t now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000u+(uint64_t)t.tv_nsec/1000u; }
static uint8_t trigger(int16_t v) { int x=v>>7; return (uint8_t)(x<0 ? 0 : x>255 ? 255 : x); }
static void cal_load(void) {
    static int loaded;
    if (loaded) return;
    loaded=1;
    const char *v=bb_env("BB_PAD_CENTER_CAL"); cal_enabled=!(v && *v=='0');
    /* Inner/outer dead zone, as shadPS4's analog_deadzone: [inner, outer] maps linearly to the
     * full deflection, so an axis with a reduced travel (outer below 127) reaches full. */
    v=bb_env("BB_PAD_DEADZONE"); pad_deadzone=v && *v ? atoi(v) : 5;
    if (pad_deadzone<0) pad_deadzone=0; else if (pad_deadzone>126) pad_deadzone=126;
    v=bb_env("BB_PAD_DEADZONE_OUTER"); pad_deadzone_outer=v && *v ? atoi(v) : 127;
    if (pad_deadzone_outer<=pad_deadzone) pad_deadzone_outer=pad_deadzone<127 ? pad_deadzone+1 : 127;
    v=bb_env("BB_PAD_CENTER");
    if (v && *v && sscanf(v,"%d,%d,%d,%d",&cal_manual_center[0],&cal_manual_center[1],
                          &cal_manual_center[2],&cal_manual_center[3])==4) {
        cal_manual=1;
        printf("Runtime: pad center (BB_PAD_CENTER): lx=%d ly=%d rx=%d ry=%d\n",
               cal_manual_center[0],cal_manual_center[1],cal_manual_center[2],cal_manual_center[3]);
    }
}
static void cal_set_base(void) {
    int biased=0;
    for (int i=0;i<4;++i) {
        const int a=abs(cal_center[i]);
        cal_base[i]=a>=PAD_CAL_BIAS ? a : 0;   /* a biased axis spans one half of the int16 range */
        cal_max[i][0]=cal_max[i][1]=0;
        biased|=cal_base[i]!=0;
    }
    if (biased)
        printf("Runtime: pad travel (biased neutral, per-axis scaling): lx=%d ly=%d rx=%d ry=%d\n",
               cal_base[0]?cal_base[0]:32768,cal_base[1]?cal_base[1]:32768,
               cal_base[2]?cal_base[2]:32768,cal_base[3]?cal_base[3]:32768);
}
/* Full-scale travel of one direction. A biased axis starts from its neutral's magnitude and is
 * refined by the largest push seen (once that push is close to the base travel), so a direction
 * whose physical travel is a few percent shorter still reaches full deflection. */
static int cal_travel(int axis,int dir) {
    const int base=cal_base[axis] ? cal_base[axis] : 32768;
    return cal_base[axis] && cal_max[axis][dir]*100>=base*PAD_CAL_REFINE ? cal_max[axis][dir] : base;
}
static void cal_reset(void) {
    cal_load();
    cal_started=0; cal_quiet_count=0; cal_since=0; memset(cal_prev,0,sizeof cal_prev);
    if (cal_manual) { memcpy(cal_center,cal_manual_center,sizeof cal_center); cal_have_center=1; }
    else if (!cal_enabled) { memset(cal_center,0,sizeof cal_center); cal_have_center=1; }
    else { cal_have_center=0; memset(cal_center,0,sizeof cal_center); memset(cal_base,0,sizeof cal_base); }
    if (cal_have_center) cal_set_base();
}
/* The value each axis holds while the sticks are untouched (SDL's pre-report zeros are skipped);
 * once it is known, the largest push seen per direction on a biased axis (see cal_travel). */
static void cal_sample(const int16_t raw[4]) {
    if (cal_have_center) {
        for (int i=0;i<4;++i) {
            if (!cal_base[i]) continue;
            const int d=raw[i]-cal_center[i], dir=d>=0, travel=dir ? d : -d;
            if (travel>cal_max[i][dir]) cal_max[i][dir]=travel;
        }
        return;
    }
    int allzero=1; for (int i=0;i<4;++i) if (raw[i]) allzero=0;
    int quiet=1; for (int i=0;i<4;++i) if (cal_started && abs(raw[i]-cal_prev[i])>PAD_CAL_QUIET) quiet=0;
    const uint64_t now=now_us();
    if (!cal_started) { cal_started=1; cal_since=now; }
    if (allzero && now-cal_since<PAD_CAL_TIMEOUT_US) quiet=0;
    if (quiet) {
        if (++cal_quiet_count>=PAD_CAL_QUIET_SAMPLES) {
            /* Only the clone pattern (all four axes biased) is taken as the neutral: a genuine pad
             * opened with a stick held steady would otherwise keep that push as its neutral. */
            int biased=0;
            for (int i=0;i<4;++i) biased+=abs(raw[i])>=PAD_CAL_BIAS;
            for (int i=0;i<4;++i) cal_center[i]=biased==4 ? raw[i] : 0;
            cal_have_center=1;
            if (biased==4)
                printf("Runtime: pad neutral: lx=%d ly=%d rx=%d ry=%d\n",cal_center[0],cal_center[1],cal_center[2],cal_center[3]);
            else if (biased)
                printf("Runtime: pad neutral 0 (%d of 4 axes off-center at open: a stick held, not a biased pad)\n",biased);
            cal_set_base();
        }
    } else cal_quiet_count=0;
    for (int i=0;i<4;++i) cal_prev[i]=raw[i];
}
/* One stick axis in the PS4 0..255 scale. The neutral is subtracted and the axis' own travel
 * (cal_travel) is read as -128..127, so a full push reaches full deflection in both directions
 * whatever part of SDL's range the pad uses; the inner/outer dead zone (BB_PAD_DEADZONE /
 * BB_PAD_DEADZONE_OUTER) then maps that travel to the full deflection, as shadPS4 does. */
static uint8_t stick_axis(int axis,int16_t raw) {
    const int d=raw-cal_center[axis];
    const int travel=cal_travel(axis,d>=0 ? 1 : 0);
    int v=travel>0 ? (int)((long long)d*128/travel) : 0;
    if (v>127) v=127; else if (v<-128) v=-128;
    const int mag=abs(v);
    if (mag<=pad_deadzone || pad_deadzone>=pad_deadzone_outer) v=0;
    else {
        int scaled=(int)(128.0*(mag-pad_deadzone)/(float)(pad_deadzone_outer-pad_deadzone));
        if (scaled>128) scaled=128;
        v = v>=0 ? scaled : -scaled;
    }
    int x=v+128;
    return (uint8_t)(x<0 ? 0 : x>255 ? 255 : x);
}
