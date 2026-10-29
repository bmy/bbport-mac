// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_pad.c part: one pad sample: the host devices, BB_PAD_FILE injection, recording and replay.
 * #included by runtime_pad.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */

/* A Steam Deck's or Steam Controller's own controls as the gamepad: in desktop mode Steam's
 * desktop layout also turns them into keys and the mouse (buttons into keys and clicks, the right
 * trackpad and stick into the mouse) while SDL reads them as the gamepad, and the game took each
 * press twice (the keyboard's prompts, the mouse turning the camera: the right stick also moves
 * Steam's mouse, and the camera jumped with it, 2026-10-23). The gamepad is then the gamepad: no
 * keys and no mouse at all. BB_DECK_DESKTOP_INPUT=1 keeps the keyboard and mouse. */
/* For the window (gpu/shim/window.cpp): its keyboard and mouse events are dropped as well. */
#include "../../gpu/shim/bb_env.h"
int runtime_pad_steam_only(void) { return atomic_load_explicit(&steam_only_now,memory_order_relaxed); }

static int steam_pad_only(SDL_Gamepad *g) {
    static int off=-1;
    if (off<0) { const char *e=bb_env("BB_DECK_DESKTOP_INPUT"); off=e && e[0]=='1'; }
    if (off || !g) return 0;
    const char *name=SDL_GetGamepadName(g);
    const int steam=name && (strstr(name,"Steam Deck") || strstr(name,"Steam Controller"));
    static int told;
    if (steam && !told) {
        told=1;
        printf("Runtime: %s as the gamepad: keyboard and mouse ignored (Steam's desktop layout; "
               "BB_DECK_DESKTOP_INPUT=1 keeps them)\n",name);
    }
    return steam;
}

static void sample_host(PadData *d) {
    report_guest_heap();
    memset(d,0,sizeof(*d));
    d->left_x=d->left_y=d->right_x=d->right_y=128;
    d->orientation[3]=1.0f;
    d->acceleration[1]=1.0f; /* 1 g up: the controller lying flat (zeros read as free fall) */
    d->connected=1; d->connected_count=connected_count ? connected_count : 1;
    d->timestamp=now_us();
    const float motion_dt=motion.last_us && d->timestamp>motion.last_us
                              ? (float)(d->timestamp-motion.last_us)*1e-6f : 0.0f;
    motion.last_us=d->timestamp;
    motion.shaking=0;
    SDL_Gamepad *g=current_gamepad();
    if (!bindings_ready) { load_bindings(); bindings_ready=1; bbgpu_mouse_look_enable(mouse_look); }
    /* The mouse's motion and wheel since the last sample (always taken: none of it piles up while
     * the menu is open); it counts while the window holds the mouse. */
    double mouse_dx=0, mouse_dy=0;
    int wheel_up=0, wheel_down=0;
    bbgpu_mouse_take(&mouse_dx,&mouse_dy,&wheel_up,&wheel_down);
    const int captured=bbgpu_mouse_captured();
    if (bbgpu_overlay_captures_input()) { apply_mouse_look(d,0,0,d->timestamp,0); return; } /* menu open: neutral input */
    /* bbport: through the GPU library (bb-gpu's window with the native GPU process). */
    const bool *k=bbgpu_keyboard_state();
    mouse_buttons=captured ? bbgpu_mouse_buttons() : 0;
    const int pad_only=steam_pad_only(g);
    atomic_store_explicit(&steam_only_now,pad_only,memory_order_relaxed);
    if (pad_only) {
        k=NULL; mouse_buttons=0; wheel_up=wheel_down=0; mouse_dx=mouse_dy=0;
        memset((void *)key_latch,0,sizeof(key_latch)); /* taps latched before it counted */
    }
    mouse_wheel_step(d->timestamp,captured ? wheel_up : 0,captured ? wheel_down : 0);
    if (g) {
        int touch_right=0;
        for (int i=IN_CROSS;i<=IN_RIGHT;++i) {
            const int v=pad_value(g,i);
            if (i==IN_L2) d->l2=(uint8_t)v;
            if (i==IN_R2) d->r2=(uint8_t)v;
            if (i==IN_TOUCHPAD_RIGHT) touch_right=v>30;
            else if (v>30) d->buttons|=input_buttons[i];
        }
        static const SDL_GamepadAxis stick_axes[4]={SDL_GAMEPAD_AXIS_LEFTX,SDL_GAMEPAD_AXIS_LEFTY,
                                                    SDL_GAMEPAD_AXIS_RIGHTX,SDL_GAMEPAD_AXIS_RIGHTY};
        int16_t raw[4];
        for (int i=0;i<4;++i) raw[i]=(int16_t)SDL_GetGamepadAxis(g,stick_axes[i]);
        cal_load();
        cal_sample(raw);
        uint8_t *axis_out[4]={&d->left_x,&d->left_y,&d->right_x,&d->right_y};
        for (int i=0;i<4;++i)
            *axis_out[i]=stick_axis(i,raw[i]); /* center 0 until the neutral is known */
        if (SDL_GetNumGamepadTouchpads(g)>0) {
            const int fingers=SDL_GetNumGamepadTouchpadFingers(g,0);
            for (int finger=0;finger<fingers && d->touch_count<2;++finger) {
                bool down=false;
                float x=0, y=0;
                if (SDL_GetGamepadTouchpadFinger(g,0,finger,&down,&x,&y,NULL) && down) {
                    d->touches[d->touch_count++]=(PadTouch){.x=touch_axis(x,1919),
                        .y=touch_axis(y,942),.id=(uint8_t)finger};
                }
            }
        }
#ifdef __linux__
        else {
            const SDL_GamepadType type=SDL_GetGamepadType(g);
            if (type==SDL_GAMEPAD_TYPE_PS4 || type==SDL_GAMEPAD_TYPE_PS5) evdev_touchpad(d,d->timestamp);
        }
#endif
        if (motion_on()) motion_from_pad(d,g,motion_dt);
        // Back/Select on pads without a touch surface is a left-side click.
        if ((d->buttons & BTN_TOUCHPAD) && !d->touch_count) touch_click(d,0);
        if (touch_right) touch_click(d,1);
    }
    {
        /* BB_PAD_TRACE=1 (diagnostics): the gamepad's buttons and touches as read, on change. */
        static int trace=-1; static uint32_t last_buttons=~0u; static int last_touches=-1;
        if (trace<0) { const char *e=bb_env("BB_PAD_TRACE"); trace=e && e[0]=='1'; }
        if (trace && g && (d->buttons!=last_buttons || d->touch_count!=last_touches)) {
            last_buttons=d->buttons; last_touches=d->touch_count;
            static int mapped; if (!mapped) { mapped=1; char *m=SDL_GetGamepadMapping(g); printf("Pad trace: mapping %s; joystick buttons %d\n",m ? m : "(none)",SDL_GetNumJoystickButtons(SDL_GetGamepadJoystick(g))); SDL_free(m); }
            printf("Pad trace: buttons 0x%x touches %d back %d\n",d->buttons,d->touch_count,
                   (int)SDL_GetGamepadButton(g,SDL_GAMEPAD_BUTTON_BACK));
        }
    }
    const int pad_active=g && (d->buttons || d->l2>30 || d->r2>30 || d->touch_count ||
                               stick_moved(d->left_x) || stick_moved(d->left_y) ||
                               stick_moved(d->right_x) || stick_moved(d->right_y));
    /* The Steam controller's trackpad mouse is the gamepad's: its prompts stay. */
    int keyboard_active=!pad_only &&
                        ((captured && (mouse_dx*mouse_dx+mouse_dy*mouse_dy>9.0 || wheel_up || wheel_down)) ||
                         (mouse_buttons!=0));
    for (int i=0;i<IN_COUNT && !keyboard_active;++i) keyboard_active=key_down(k,i);
    prompt_note(g,pad_active,keyboard_active);
    apply_keyboard(d,k); /* the bound mouse buttons too */
    video_skip(d,k);
    if (motion_on()) {
        motion_from_desktop(d,k,&mouse_dx,&mouse_dy,captured);
        motion_report(d,motion_dt);
    }
    apply_mouse_look(d,mouse_dx,mouse_dy,d->timestamp,captured);
}

/* BB_PAD_FILE=<file>: scripted input for automated runs. The file holds whitespace-separated
 * tokens, re-read when it changes: button names (cross circle square triangle l1 r1 l2 r2 l3 r3
 * options touchpad touchpad_left touchpad_right up down left right) are held while listed;
 * touchpad defaults to a left-side click; lx= ly= rx= ry= (0..255) override
 * the sticks; mdx= mdy= move the mouse that many pixels at every pad read, as if the window held
 * it (mouse look tests). An empty file releases everything. */
static struct { uint32_t buttons; int stick[4]; int touch_side; float mouse[2]; } injected={0,{-1,-1,-1,-1},-1,{0,0}};
static int replay_armed;      /* 1 while a BB_PAD_REPLAY recording plays, 2 once it ended */
static int replay_running;    /* 0: (re)start at the next sample */
static uint64_t replay_start;
static void read_inject(void) {
    static const char *path; static int checked; static uint64_t last_check; static struct timespec mtime;
    if (!checked) { path=bb_env("BB_PAD_FILE"); checked=1; }
    if (!path || !*path) return;
    uint64_t now=now_us();
    if (now-last_check<20000) return;
    last_check=now;
    struct stat st;
    if (stat(path,&st)!=0) return;
    if (st.st_mtim.tv_sec==mtime.tv_sec && st.st_mtim.tv_nsec==mtime.tv_nsec) return;
    mtime=st.st_mtim;
    FILE *f=fopen(path,"r");
    if (!f) return;
    static const struct { const char *name; uint32_t ps; } names[]={
        {"cross",BTN_CROSS}, {"circle",BTN_CIRCLE}, {"square",BTN_SQUARE}, {"triangle",BTN_TRIANGLE},
        {"l1",BTN_L1}, {"r1",BTN_R1}, {"l2",BTN_L2}, {"r2",BTN_R2}, {"l3",BTN_L3}, {"r3",BTN_R3},
        {"options",BTN_OPTIONS}, {"touchpad",BTN_TOUCHPAD},
        {"up",BTN_UP}, {"down",BTN_DOWN}, {"left",BTN_LEFT}, {"right",BTN_RIGHT},
    };
    static const char *sticks[]={"lx=","ly=","rx=","ry="};
    injected.buttons=0;
    injected.touch_side=-1;
    for (int i=0;i<4;++i) injected.stick[i]=-1;
    injected.mouse[0]=injected.mouse[1]=0;
    int vpad_down[SDL_GAMEPAD_BUTTON_COUNT]={0};
    char token[64];
    while (fscanf(f,"%63s",token)==1) {
        if (!strncmp(token,"vpad:",5)) {
            const SDL_GamepadButton b=SDL_GetGamepadButtonFromString(token+5);
            if (b!=SDL_GAMEPAD_BUTTON_INVALID) vpad_down[b]=1;
        }
        if (!strcmp(token,"replay") && replay_armed!=1) { replay_armed=1; replay_running=0; } /* BB_PAD_REPLAY */
        if (!strcmp(token,"touchpad_left") || !strcmp(token,"touchpad_right")) {
            injected.buttons|=BTN_TOUCHPAD;
            injected.touch_side=!strcmp(token,"touchpad_right");
        }
        for (size_t i=0;i<sizeof(names)/sizeof(*names);++i) if (!strcmp(token,names[i].name)) injected.buttons|=names[i].ps;
        for (int i=0;i<4;++i) if (!strncmp(token,sticks[i],3)) { int v=atoi(token+3); injected.stick[i]=v<0 ? 0 : v>255 ? 255 : v; }
        if (!strncmp(token,"mdx=",4)) injected.mouse[0]=(float)atof(token+4);
        if (!strncmp(token,"mdy=",4)) injected.mouse[1]=(float)atof(token+4);
    }
    if (test_vpad) {
        for (int b=0;b<SDL_GAMEPAD_BUTTON_MISC1;++b)
            if (!SDL_SetJoystickVirtualButton(test_vpad,b,vpad_down[b]) && vpad_down[b])
                printf("Runtime: test gamepad button %d: %s\n",b,SDL_GetError());
    }
    fclose(f);
    printf("Runtime: pad file: buttons 0x%x sticks %d %d %d %d\n",injected.buttons,
           injected.stick[0],injected.stick[1],injected.stick[2],injected.stick[3]);
}
/* BB_PAD_RECORD=<file>: F9 starts and stops recording the pad state (gamepad or keyboard) with
 * the time since F9; BB_PAD_REPLAY=<file> plays such a recording back, started by the token
 * "replay" in BB_PAD_FILE (scripted tests repeat a route the player ran once). Lines: ms buttons
 * lx ly rx ry l2 r2, written when the state changes. */
typedef struct { uint32_t ms, buttons; uint8_t axes[4], l2, r2; } PadSample;
/* The clock of recordings and replays: time while the game shows its world (bbgpu_world_active),
 * counted per frame in the GPU layer, so a loading screen of another length (a lamp travel) does
 * not shift the inputs after it. BB_PAD_CLOCK=wall: the wall clock, as before. */
static uint64_t pad_clock_us(void) {
    static int wall=-1;
    if (wall<0) { const char *e=bb_env("BB_PAD_CLOCK"); wall=e && !strcmp(e,"wall"); }
    if (wall) return now_us();
    return bbgpu_world_us(); /* each world frame's whole interval (Display::Flip) */
}
static FILE *record_file;
static uint64_t record_start;
static PadSample record_last;
static void record_sample(const PadData *d) {
    static const char *path; static int checked, f9_was_down;
    if (!checked) { path=bb_env("BB_PAD_RECORD"); checked=1; }
    if (!path || !*path || !sdl_ready) return;
    const bool *k=bbgpu_keyboard_state();
    const int f9=k && k[SDL_SCANCODE_F9];
    if (f9 && !f9_was_down) {
        if (record_file) {
            fclose(record_file); record_file=NULL;
            printf("Runtime: pad recording stopped (%s)\n",path);
        } else if ((record_file=fopen(path,"w"))) {
            record_start=pad_clock_us();
            memset(&record_last,0xff,sizeof(record_last));
            printf("Runtime: pad recording started (%s, F9 stops)\n",path);
        }
    }
    f9_was_down=f9;
    if (!record_file) return;
    PadSample s={(uint32_t)((pad_clock_us()-record_start)/1000),d->buttons,
                 {d->left_x,d->left_y,d->right_x,d->right_y},d->l2,d->r2};
    if (s.buttons==record_last.buttons && !memcmp(s.axes,record_last.axes,4) &&
        s.l2==record_last.l2 && s.r2==record_last.r2) return;
    record_last=s;
    fprintf(record_file,"%u %u %u %u %u %u %u %u\n",s.ms,s.buttons,s.axes[0],s.axes[1],s.axes[2],
            s.axes[3],s.l2,s.r2);
    fflush(record_file);
}
static PadSample *replay; static size_t replay_count, replay_next;
static void replay_sample(PadData *d) {
    if (!replay_armed) return;
    if (!replay_running) {
        static int loaded;
        if (!loaded) {
            loaded=1;
            const char *path=bb_env("BB_PAD_REPLAY");
            FILE *f=path ? fopen(path,"r") : NULL;
            PadSample s; unsigned v[8]; size_t cap=0;
            while (f && fscanf(f,"%u %u %u %u %u %u %u %u",&v[0],&v[1],&v[2],&v[3],&v[4],&v[5],&v[6],&v[7])==8) {
                s=(PadSample){v[0],v[1],{(uint8_t)v[2],(uint8_t)v[3],(uint8_t)v[4],(uint8_t)v[5]},(uint8_t)v[6],(uint8_t)v[7]};
                if (replay_count==cap && !(replay=realloc(replay,(cap=cap ? cap*2 : 1024)*sizeof(*replay)))) break;
                replay[replay_count++]=s;
            }
            if (f) fclose(f);
            /* From the first input that is not neutral: the recording started (F9) a moment the replay
             * cannot know, while the player stood. */
            size_t lead=0;
            while (lead<replay_count && !replay[lead].buttons && replay[lead].l2<8 && replay[lead].r2<8 &&
                   abs(replay[lead].axes[0]-128)<24 && abs(replay[lead].axes[1]-128)<24 &&
                   abs(replay[lead].axes[2]-128)<24 && abs(replay[lead].axes[3]-128)<24) ++lead;
            if (lead<replay_count) {
                const uint32_t t0=replay[lead].ms;
                for (size_t i=lead;i<replay_count;++i) replay[i-lead]=replay[i], replay[i-lead].ms-=t0;
                replay_count-=lead;
            }
            printf("Runtime: pad replay of %zu samples from %s\n",replay_count,path ? path : "(unset)");
        }
        replay_start=pad_clock_us();
        replay_running=1;
        replay_next=0;
    }
    const uint32_t ms=(uint32_t)((pad_clock_us()-replay_start)/1000);
    while (replay_next<replay_count && replay[replay_next].ms<=ms) ++replay_next;
    if (!replay_next) return;
    if (replay_next==replay_count && ms>replay[replay_count-1].ms+500) {
        if (replay_armed==1) { puts("Runtime: pad replay finished"); replay_armed=2; }
        return;
    }
    const PadSample *s=&replay[replay_next-1];
    d->buttons=s->buttons;
    d->left_x=s->axes[0]; d->left_y=s->axes[1]; d->right_x=s->axes[2]; d->right_y=s->axes[3];
    d->l2=s->l2; d->r2=s->r2;
}
/* Touches as the DualShock 4 reports them: every finger that goes down gets a new id (1..127, kept
 * while it stays down) and the time since the first one went down. With id 0 and no hold time
 * the game ignored touchpad presses: no gesture menu from the touchpad, Back or Tab. */
static void touch_ids(PadData *d) {
    static uint8_t next_id=1, ids[2]; static int down[2]; static uint64_t since;
    for (int i=0;i<2;++i) {
        const int now=i<d->touch_count;
        if (now && !down[i]) { ids[i]=next_id; next_id=next_id==127 ? 1 : next_id+1; }
        down[i]=now;
        if (now) d->touches[i].id=ids[i];
    }
    if (!d->touch_count) since=0;
    else if (!since) since=d->timestamp;
    d->touch_held_time=d->touch_count ? (uint32_t)(d->timestamp-since) : 0;
}
static void sample(PadData *d) {
    sample_host(d);
    if (bbgpu_overlay_captures_input()) return;
    record_sample(d);
    read_inject();
    replay_sample(d);
    d->buttons|=injected.buttons;
    if (injected.touch_side>=0) touch_click(d,injected.touch_side);
    else if ((d->buttons & BTN_TOUCHPAD) && !d->touch_count) touch_click(d,0);
    if (injected.buttons & BTN_L2) d->l2=255;
    if (injected.buttons & BTN_R2) d->r2=255;
    uint8_t *axes[4]={&d->left_x,&d->left_y,&d->right_x,&d->right_y};
    for (int i=0;i<4;++i) if (injected.stick[i]>=0) *axes[i]=(uint8_t)injected.stick[i];
    /* BB_PAD_TAP_TEST=ms (diagnostics, #325): the right stick fully right for exactly that long
     * every 1.2 s, by the read's own clock, 20 s after the first read. */
    static int tap_ms=-1; static uint64_t tap_start;
    if (tap_ms<0) { const char *v=bb_env("BB_PAD_TAP_TEST"); tap_ms=v && *v ? atoi(v) : 0; tap_start=d->timestamp+20000000; }
    if (tap_ms>0 && d->timestamp>=tap_start && (d->timestamp-tap_start)%1200000<(uint64_t)tap_ms*1000) d->right_x=255;
    if (injected.mouse[0]!=0 || injected.mouse[1]!=0)
        apply_mouse_look(d,injected.mouse[0],injected.mouse[1],d->timestamp,1);
    /* Game modules (a menu pointer's presses). */
    const uint8_t sticks[4]={d->left_x,d->left_y,d->right_x,d->right_y};
    bbgpu_filter_pad(&d->buttons,sticks);
    touch_ids(d);
}
