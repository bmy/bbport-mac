// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_pad.c part: motion for controllers without sensors (the spray can held sideways, shaking, aiming).
 * #included by runtime_pad.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */
#include "../../gpu/shim/bb_env.h"
static void quat_mul(const float a[4], const float b[4], float out[4]) { /* x y z w */
    out[0]=a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1];
    out[1]=a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0];
    out[2]=a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3];
    out[3]=a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2];
}
/* The world's up (0, 1, 0) in the frame of the controller turned by q (q's inverse applied). */
static void up_in_frame(const float q[4], float out[3]) {
    const float x=q[0], y=q[1], z=q[2], w=q[3];
    out[0]=2.0f*(x*y+w*z);
    out[1]=1.0f-2.0f*(x*x+z*z);
    out[2]=2.0f*(y*z-w*x);
}

/* ---- Motion for controllers without sensors -------------------------------------------------
 * Games read the DualShock 4's sensors: one asks to hold the controller sideways like a
 * spray can, shake it and aim by tilting it. Keyboard and mouse:
 *   G               take the can (the controller held sideways), G again to put it away;
 *   the mouse       aims while the can is held (the camera does not turn meanwhile);
 *   left button     sprays (R2) while the can is held;
 *   H, held         shakes the can;
 *   T               brings the aim back to the middle;
 *   Tab + the mouse  a finger on the touchpad moved by the mouse (swipes: "swipe the touchpad to
 *                    open the door"); Tab + Backspace two fingers ("push the car": both, then
 *                    the mouse up).
 * A pad without sensors (Xbox): View is a modifier, View + A the can, View + B shaking, the right
 * stick aims while the can is held; View + the right stick otherwise swipes the touchpad that
 * way with one finger, View + the left stick with two (one on each half); a short press of View
 * alone is the touchpad, on release.
 * Reported consistently: the orientation quaternion, gravity in the controller's frame (1 g up
 * when flat, plus the shake) and the angular velocity of the aim. On for the games that need it:
 * bbport.ini pad_motion=1 (or BB_PAD_MOTION=1); elsewhere nothing changes. */
static struct {
    int view_held, view_used, prev_a, prev_g, sideways, shaking;
    float yaw, pitch, prev_yaw, prev_pitch; /* the aim, radians */
    uint64_t last_us, click_until_us;
    /* A swipe from the right stick: the finger goes from one side of the touchpad (0..1919 x
     * 0..942) through the middle to the other, in the stick's direction. */
    int swiping, swipe_fingers;
    float swipe_x, swipe_y, swipe_end_x, swipe_end_y;
    /* Tab + the mouse: how far the mouse has moved the finger since Tab went down. */
    float drag_x, drag_y;
} motion;

static int motion_on(void) {
    static int env_checked;
    if (!env_checked) {
        env_checked=1;
        const char *env=bb_env("BB_PAD_MOTION");
        if (env && *env) motion_enabled=env[0]=='1';
        if (motion_enabled>0)
            printf("Pad: motion emulation on (G: spray can, mouse aims, left button sprays, H shakes, "
                   "Tab+mouse swipes the touchpad, Tab+Backspace two fingers; pad: View+A, View+B, "
                   "right stick, View+right stick swipes, View+left stick two fingers)\n");
    }
    return motion_enabled>0;
}
static void set_sideways(int on) {
    motion.sideways=on; motion.yaw=motion.pitch=motion.prev_yaw=motion.prev_pitch=0.0f;
    if (on) bbgpu_mouse_capture(); /* the mouse aims now: it must not leave the window */
    printf("Pad: motion: %s\n",on ? "spray can taken (controller sideways)" : "spray can put away");
}
/* The aim from screen directions: the controller is held rolled 90 degrees, so the turn that
 * moves the game's aim across the screen is its pitch, and up and down its yaw. `right` and `up`
 * in radians. */
static void aim_screen(float right, float up);
static void aim(float dyaw, float dpitch) {
    const float limit=0.61f; /* 35 degrees each way: further, the game's aim leaves the picture */
    motion.yaw=fmaxf(-limit,fminf(limit,motion.yaw+dyaw));
    motion.pitch=fmaxf(-limit,fminf(limit,motion.pitch+dpitch));
}

/* The pad's part: View as the modifier, the right stick aiming at up to 120 degrees a second. */
static void motion_from_pad(PadData *d, SDL_Gamepad *g, float dt) {
    const int view=SDL_GetGamepadButton(g,SDL_GAMEPAD_BUTTON_BACK);
    const int real_touchpad=SDL_GetGamepadButton(g,SDL_GAMEPAD_BUTTON_TOUCHPAD);
    const int a=(d->buttons & BTN_CROSS)!=0, b=(d->buttons & BTN_CIRCLE)!=0;
    if (view) {
        if (!motion.view_held) { motion.view_held=1; motion.view_used=0; motion.prev_a=a; }
        if (a && !motion.prev_a) { set_sideways(!motion.sideways); motion.view_used=1; }
        if (b) { motion.shaking=1; motion.view_used=1; }
        motion.prev_a=a;
        /* View is the modifier while held: neither a touchpad press nor A or B for the game. */
        d->buttons&=~(uint32_t)(BTN_CROSS|BTN_CIRCLE);
        if (!real_touchpad) d->buttons&=~(uint32_t)BTN_TOUCHPAD;
    } else if (motion.view_held) {
        motion.view_held=0;
        if (!motion.view_used) motion.click_until_us=d->timestamp+120000; /* a tap: the touchpad */
    }
    if (!view && d->timestamp<motion.click_until_us) d->buttons|=BTN_TOUCHPAD;
    if (motion.sideways) {
        const float rx=((int)d->right_x-128)/127.0f, ry=((int)d->right_y-128)/127.0f;
        if (fabsf(rx)>0.05f || fabsf(ry)>0.05f) {
            aim_screen(rx*1.2f*dt,-ry*1.2f*dt);
            if (view) motion.view_used=1;
        }
        d->right_x=d->right_y=128;
    } else if (view) {
        /* View + the right stick: a one-finger swipe that way, a quarter of a second across the
         * touchpad; View + the left stick: two fingers (one on each half) moved that way. */
        const float rx=((int)d->right_x-128)/127.0f, ry=((int)d->right_y-128)/127.0f;
        const float lx=((int)d->left_x-128)/127.0f, ly=((int)d->left_y-128)/127.0f;
        const int two=sqrtf(rx*rx+ry*ry)<=0.5f && sqrtf(lx*lx+ly*ly)>0.5f;
        const float sx=two ? lx : rx, sy=two ? ly : ry;
        const float len=sqrtf(sx*sx+sy*sy);
        if (len>0.5f) {
            if (!motion.swiping || motion.swipe_fingers!=(two ? 2 : 1)) {
                const float dx=sx/len, dy=sy/len;
                const float reach_x=two ? 400.0f : 800.0f; /* each finger stays on its half */
                motion.swiping=1;
                motion.swipe_fingers=two ? 2 : 1;
                motion.swipe_x=960.0f-dx*reach_x; motion.swipe_y=471.0f-dy*400.0f;
                motion.swipe_end_x=960.0f+dx*reach_x; motion.swipe_end_y=471.0f+dy*400.0f;
            }
            const float step=6400.0f*dt;
            const float ex=motion.swipe_end_x-motion.swipe_x, ey=motion.swipe_end_y-motion.swipe_y;
            const float left=sqrtf(ex*ex+ey*ey);
            if (left>step && left>0.0f) {
                motion.swipe_x+=ex/left*step; motion.swipe_y+=ey/left*step;
            } else {
                motion.swipe_x=motion.swipe_end_x; motion.swipe_y=motion.swipe_end_y;
            }
            const uint16_t y=(uint16_t)fmaxf(0.0f,fminf(942.0f,motion.swipe_y));
            if (motion.swipe_fingers==2) {
                /* The pair's middle follows the swipe; the fingers 960 apart. */
                d->touch_count=0;
                d->touches[d->touch_count++]=(PadTouch){
                    .x=(uint16_t)fmaxf(0.0f,fminf(1919.0f,motion.swipe_x-480.0f)),.y=y,.id=0};
                d->touches[d->touch_count++]=(PadTouch){
                    .x=(uint16_t)fmaxf(0.0f,fminf(1919.0f,motion.swipe_x+480.0f)),.y=y,.id=1};
            } else if (d->touch_count<2) {
                d->touches[d->touch_count++]=(PadTouch){
                    .x=(uint16_t)fmaxf(0.0f,fminf(1919.0f,motion.swipe_x)),.y=y,.id=0};
            }
            motion.view_used=1;
        } else {
            motion.swiping=0;
        }
        d->right_x=d->right_y=128; /* not the camera */
        d->left_x=d->left_y=128;   /* nor walking */
    }
    if (!view) motion.swiping=0;
}

/* The keyboard's and mouse's part: G takes or puts away the can, H shakes, the mouse aims and its
 * left button sprays while the can is held (the mouse then neither turns the camera nor
 * attacks). */
static void motion_from_desktop(PadData *d, const bool *k, double *mouse_dx, double *mouse_dy,
                                int captured) {
    const int g_key=k && k[SDL_SCANCODE_G];
    if (g_key && !motion.prev_g) set_sideways(!motion.sideways);
    motion.prev_g=g_key;
    if (k && k[SDL_SCANCODE_H]) motion.shaking=1;
    /* Tab (the touchpad key) held: the mouse moves the finger it put down (a swipe). */
    if ((key_down(k,IN_TOUCHPAD) || key_down(k,IN_TOUCHPAD_RIGHT)) && d->touch_count>0) {
        if (captured) {
            motion.drag_x+=(float)*mouse_dx*2.0f;
            motion.drag_y+=(float)*mouse_dy*2.0f;
            *mouse_dx=*mouse_dy=0; /* not the camera */
        }
        for (int i=0;i<d->touch_count;++i) {
            PadTouch *t=&d->touches[i];
            t->x=(uint16_t)fmaxf(0.0f,fminf(1919.0f,(float)t->x+motion.drag_x));
            t->y=(uint16_t)fmaxf(0.0f,fminf(942.0f,(float)t->y+motion.drag_y));
        }
    } else {
        motion.drag_x=motion.drag_y=0.0f;
    }
    if (!motion.sideways) return;
    if (k && k[SDL_SCANCODE_T]) motion.yaw=motion.pitch=0.0f; /* the aim back to the middle */
    if (captured) {
        const float per_pixel=0.0012f*mouse_sensitivity; /* ~800 pixels across the 70 degrees */
        aim_screen((float)*mouse_dx*per_pixel,-(float)*mouse_dy*per_pixel);
        *mouse_dx=*mouse_dy=0; /* not the camera */
    }
    if (captured && (mouse_buttons & SDL_BUTTON_LMASK)) {
        d->buttons&=~(uint32_t)BTN_R1; /* the left button's usual attack */
        d->buttons|=BTN_R2;
        d->r2=255;
    }
}

static void aim_screen(float right, float up) { aim(up, -right); }

/* The state the game reads, from the can and the aim. */
static void motion_report(PadData *d, float dt) {
    const float s=0.70710678f;
    /* The can: rolled 90 degrees about the controller's front-back axis; then the aim. */
    const float roll[4]={0.0f,0.0f,motion.sideways ? s : 0.0f,motion.sideways ? s : 1.0f};
    const float yawq[4]={0.0f,sinf(motion.yaw*0.5f),0.0f,cosf(motion.yaw*0.5f)};
    const float pitchq[4]={sinf(motion.pitch*0.5f),0.0f,0.0f,cosf(motion.pitch*0.5f)};
    float tilt[4], q[4];
    quat_mul(yawq,pitchq,tilt);
    quat_mul(roll,tilt,q);
    for (int i=0;i<4;++i) d->orientation[i]=q[i];
    float up[3];
    up_in_frame(q,up);
    for (int i=0;i<3;++i) d->acceleration[i]=up[i];
    if (motion.shaking) {
        /* 2.5 g back and forth along the controller's long axis, 8 times a second. */
        d->acceleration[0]+=2.5f*sinf((float)(d->timestamp%1000000)*1e-6f*2.0f*3.14159265f*8.0f);
    }
    d->angular_velocity[0]=dt>0.0f ? (motion.pitch-motion.prev_pitch)/dt : 0.0f;
    d->angular_velocity[1]=dt>0.0f ? (motion.yaw-motion.prev_yaw)/dt : 0.0f;
    d->angular_velocity[2]=0.0f;
    motion.prev_yaw=motion.yaw; motion.prev_pitch=motion.pitch;
}
