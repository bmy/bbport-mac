/* libScePad on SDL3 gamepads and the keyboard. SDL events are pumped by the window thread
 * (gpu/shim/window.cpp); here state is only sampled.
 *
 * Keyboard layout (Dark Souls III's on PC, with the player's changes; also with a gamepad
 * connected: both drive the game):
 *   WASD left stick, IJKL right stick (the camera), Space the dodge (a short Circle press: never a
 *   run), Left Shift the run (Circle held), Shift + Space the jump (Circle let go and pressed again
 *   while running), F and Enter Cross
 *   (interact, confirm), Q Triangle (blood vial), E L1 (transform the weapon), Tab and R Square
 *   (the quick item), C R3 (lock-on), Left Ctrl L3, Escape Options (the menu; Circle, the way back, while a game menu is open), G left touchpad
 *   (gestures), Backspace right touchpad; the d-pad on 1 (left: the left hand's weapon), 2 (right:
 *   the right hand's), 3 (up), 4 (down) and the arrow keys.
 *   Mouse (once the game window is clicked; F1 or leaving the window releases it): look (the
 *   right stick), Left R1 (attack), Shift+Left R2, Right L2 (the gun), Middle R3, side buttons R2
 *   (X1) and L1 (X2), wheel the d-pad's up/down (quick items).
 *   A binding may be a chord, "Left Shift+Mouse Left": while it is held, the second key's plain
 *   bindings do not count (Shift+Left is R2, not R2 and R1).
 *
 * Stick neutral: SDL exposes no way to read a pad's calibration and some clones report a
 * biased neutral (a Switch-style pad was seen returning both sticks at a constant ~ +/-16380).
 * The neutral of each axis is taken from a quiet window after the pad opens (SDL returns zero
 * until the first report arrives, so those samples are skipped) and subtracted when all four axes
 * are biased; otherwise (a genuine pad, perhaps opened with a stick held) the neutral is 0.
 *
 * Travel: such a clone also uses only part of SDL's -32768..32767 span (its neutral sits in the
 * middle of one half), so reading the axis as -128..127 would reach only half deflection and a
 * full push would never run. Each axis is instead scaled by its own travel: a biased axis starts
 * from its neutral's magnitude and is refined by the largest push seen per direction, so both
 * directions reach full deflection even when the two travels differ by a few percent (that
 * difference is what makes one direction run and the other only walk). A genuine pad keeps the
 * plain -32768..32767 -> -128..127 read. The stick is then converted as shadPS4 does, with an
 * inner/outer dead zone (BB_PAD_DEADZONE, default 5; BB_PAD_DEADZONE_OUTER, default 127) mapping
 * the axis' travel up to full deflection. BB_PAD_CENTER=lx,ly,rx,ry overrides the neutral,
 * BB_PAD_CENTER_CAL=0 disables the measurement. */
#define _GNU_SOURCE
#include "runtime.h"
#include "platform.h"
#include "gpu/bbgpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <SDL3/SDL.h>
#include <sys/stat.h>

#define ERR_INVALID_ARG ((int32_t)0x80920001)
#define ERR_INVALID_HANDLE ((int32_t)0x80920003)
#define ERR_ALREADY_OPENED ((int32_t)0x80920004)
#define ERR_NOT_INITIALIZED ((int32_t)0x80920005)
#define PAD_HANDLE 1

enum {
    BTN_L3=0x2, BTN_R3=0x4, BTN_OPTIONS=0x8, BTN_UP=0x10, BTN_RIGHT=0x20, BTN_DOWN=0x40, BTN_LEFT=0x80,
    BTN_L2=0x100, BTN_R2=0x200, BTN_L1=0x400, BTN_R1=0x800, BTN_TRIANGLE=0x1000, BTN_CIRCLE=0x2000,
    BTN_CROSS=0x4000, BTN_SQUARE=0x8000, BTN_TOUCHPAD=0x100000,
};
typedef struct { uint16_t x, y; uint8_t id, reserve[3]; } PadTouch;
typedef struct {
    uint32_t buttons;
    uint8_t left_x, left_y, right_x, right_y;
    uint8_t l2, r2, analog_padding[2];
    float orientation[4], acceleration[3], angular_velocity[3];
    uint8_t touch_count, touch_reserve[3];
    uint32_t touch_held_time;
    PadTouch touches[2];
    uint8_t connected, pad0[3];
    uint64_t timestamp;
    uint8_t extension[16];
    uint8_t connected_count, reserve[2], unique_length, unique[12];
} PadData;
typedef struct {
    float pixel_density; uint16_t resolution_x, resolution_y;
    uint8_t dead_zone_left, dead_zone_right, connection_type, connected_count;
    uint8_t connected, pad[3];
    int32_t device_class;
    uint8_t reserve[8];
} ControllerInfo;
_Static_assert(sizeof(PadData)==120,"OrbisPadData layout");
_Static_assert(sizeof(PadTouch)==8,"OrbisPadTouch layout");
_Static_assert(__builtin_offsetof(PadData,touches)==60,"OrbisPadData touch offset");
_Static_assert(__builtin_offsetof(PadData,timestamp)==80,"OrbisPadData timestamp offset");
_Static_assert(sizeof(ControllerInfo)==28,"OrbisPadControllerInformation layout");

static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static int initialized, opened, sdl_ready;
static SDL_Gamepad *gamepad;
static size_t reads;
static uint8_t connected_count;

/* Per-controller stick neutral (see the file header): the value each axis holds while the
 * sticks are untouched, taken from a quiet window after the pad opens. Works for any pad; a
 * genuine one settles at 0 and the subtraction is a no-op. */
#define PAD_CAL_QUIET 512              /* an axis is quiet when it moved no more than this */
#define PAD_CAL_QUIET_SAMPLES 8
#define PAD_CAL_TIMEOUT_US 3000000u    /* after this, an all-zero reading is accepted as the neutral */
#define PAD_CAL_BIAS 8192              /* |neutral| past this: a biased neutral and a reduced travel */
#define PAD_CAL_REFINE 90              /* % of the base travel a push must reach to count as the full scale */
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
    const char *v=getenv("BB_PAD_CENTER_CAL"); cal_enabled=!(v && *v=='0');
    /* Inner/outer dead zone, as shadPS4's analog_deadzone: [inner, outer] maps linearly to the
     * full deflection, so an axis with a reduced travel (outer below 127) reaches full. */
    v=getenv("BB_PAD_DEADZONE"); pad_deadzone=v && *v ? atoi(v) : 5;
    if (pad_deadzone<0) pad_deadzone=0; else if (pad_deadzone>126) pad_deadzone=126;
    v=getenv("BB_PAD_DEADZONE_OUTER"); pad_deadzone_outer=v && *v ? atoi(v) : 127;
    if (pad_deadzone_outer<=pad_deadzone) pad_deadzone_outer=pad_deadzone<127 ? pad_deadzone+1 : 127;
    v=getenv("BB_PAD_CENTER");
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

/* BB_GAMEPAD (the launcher's controller choice): its SDL GUID, or part of its name. Issue #15:
 * wheels and other controllers connected for good came first. */
static const char *preferred_gamepad(void) {
    static const char *want; static int read;
    if (!read) { want=getenv("BB_GAMEPAD"); if (want && !*want) want=NULL; read=1; }
    return want;
}
static int is_preferred(SDL_JoystickID id, const char *want) {
    char guid[33];
    SDL_GUIDToString(SDL_GetGamepadGUIDForID(id),guid,sizeof guid);
    const char *name=SDL_GetGamepadNameForID(id);
    return !strcasecmp(guid,want) || (name && strcasestr(name,want));
}
/* The chosen gamepad, else the first while it is not connected (checked again every second, it
 * is taken as soon as it connects); called under lock. */
static SDL_Gamepad *current_gamepad(void) {
    static int on_preferred; static uint64_t last_scan;
    if (!sdl_ready) sdl_ready = SDL_WasInit(SDL_INIT_GAMEPAD) ? 1 : SDL_InitSubSystem(SDL_INIT_GAMEPAD) ? 1 : -1;
    if (sdl_ready<0) return NULL;
    if (gamepad && !SDL_GamepadConnected(gamepad)) { SDL_CloseGamepad(gamepad); gamepad=NULL; }
    const char *want=preferred_gamepad();
    const uint64_t now=now_us();
    if (!gamepad || (want && !on_preferred && now-last_scan>1000000)) {
        last_scan=now;
        int count=0, pick=-1;
        SDL_JoystickID *ids=SDL_GetGamepads(&count);
        for (int i=0; want && ids && i<count && pick<0; ++i) if (is_preferred(ids[i],want)) pick=i;
        if (pick<0 && !gamepad && ids && count>0) pick=0;
        if (pick>=0 && (!gamepad || SDL_GetGamepadID(gamepad)!=ids[pick])) {
            if (gamepad) SDL_CloseGamepad(gamepad);
            gamepad=SDL_OpenGamepad(ids[pick]);
            cal_reset(); /* a new pad has its own neutral */
            on_preferred=want && gamepad && is_preferred(ids[pick],want);
            if (gamepad) {
                ++connected_count;
                printf("Runtime: gamepad connected: %s%s\n",SDL_GetGamepadName(gamepad),
                       !want ? "" : on_preferred ? " (the chosen one)" : " (the chosen one is not connected)");
            }
        }
        SDL_free(ids);
    }
    return gamepad;
}
/* bbport (frame stats): the game's libc heap, read every 5 s from a game thread (it reads the pad)
 * with libc.prx's malloc_stats (export stub at libc.prx+0x1f8c0; malloc_stats_fast returns 1 in this libc, the module at image +0x56e0000):
 * in use now and at most, and what it took from the system. A heap that keeps growing is a leak. */
typedef struct { uint16_t size, version; uint32_t reserved; uint64_t max_system, system, max_in_use, in_use; } MallocManagedSize;
static void report_guest_heap(void) {
    static uint64_t last;
    if (!runtime_frame_stats()) return; /* bbport (macOS): before the clock read */
    const uint64_t now=now_us();
    if (now-last<5000000) return;
    last=now;
    /* Bloodborne's libc.prx (another game's libc has its exports elsewhere). */
    if (!runtime_libc_base) return;
    const uint8_t *stub=(const uint8_t *)(runtime_libc_base+0x1f8c0);
    if (runtime_libc_base!=0x800000000ull+0x56e0000 || stub[0]!=0xff || stub[1]!=0x25) return;
    ABI int (*stats)(MallocManagedSize *)=(ABI int (*)(MallocManagedSize *))(uintptr_t)stub;
    MallocManagedSize m={.size=sizeof(m),.version=1};
    const int result=stats(&m);
    if (result!=0) { static int told; if (!told++) printf("Guest heap: malloc_stats returned %#x\n",(unsigned)result); return; }
    {
        printf("Guest heap: %.1f MB in use (most %.1f), %.1f MB from the system (most %.1f)\n",
               m.in_use/1048576.0,m.max_in_use/1048576.0,m.system/1048576.0,m.max_system/1048576.0);
    }
}
/* Controls: what each PS4 input is bound to. Defaults below; bbport.ini (BB_CONFIG) lines
 * key.<input>=<SDL key names> and pad.<input>=<SDL gamepad button names>, comma-separated,
 * replace an input's binding (empty: unbound). Inputs: the buttons (cross ... right, touchpad =
 * a left-side click, touchpad_right), and on the keyboard the sticks: move_* (left), look_*
 * (right). Gamepad names as SDL's: a b x y back start leftstick rightstick leftshoulder
 * rightshoulder dpup dpdown dpleft dpright touchpad misc1 paddle1-4, plus lefttrigger and
 * righttrigger. The keyboard works next to a gamepad (the Steam Deck always has one): its buttons
 * add to the gamepad's, a held move/look key moves the stick all the way. */
enum {
    IN_CROSS, IN_CIRCLE, IN_SQUARE, IN_TRIANGLE, IN_L1, IN_R1, IN_L2, IN_R2, IN_L3, IN_R3,
    IN_OPTIONS, IN_TOUCHPAD, IN_TOUCHPAD_RIGHT, IN_UP, IN_DOWN, IN_LEFT, IN_RIGHT,
    IN_MOVE_UP, IN_MOVE_DOWN, IN_MOVE_LEFT, IN_MOVE_RIGHT, IN_LOOK_UP, IN_LOOK_DOWN, IN_LOOK_LEFT,
    IN_LOOK_RIGHT, IN_SPRINT, IN_DODGE, IN_COUNT
};
static const char *const input_names[IN_COUNT]={
    "cross","circle","square","triangle","l1","r1","l2","r2","l3","r3","options","touchpad",
    "touchpad_right","up","down","left","right","move_up","move_down","move_left","move_right",
    "look_up","look_down","look_left","look_right","sprint","dodge",
};
static const uint32_t input_buttons[IN_COUNT]={
    BTN_CROSS,BTN_CIRCLE,BTN_SQUARE,BTN_TRIANGLE,BTN_L1,BTN_R1,BTN_L2,BTN_R2,BTN_L3,BTN_R3,
    BTN_OPTIONS,BTN_TOUCHPAD,0,BTN_UP,BTN_DOWN,BTN_LEFT,BTN_RIGHT,
};
#define MAX_BIND 4
/* Mouse buttons and the wheel in key.<input> lines ("Mouse Left", "Wheel Down", ...): codes past
 * SDL's scancodes. They count only while the window holds the mouse for looking (gpu/shim
 * window.cpp): a click that takes the mouse is not an attack. */
enum { MOUSE_KEY=SDL_SCANCODE_COUNT, MOUSE_WHEEL_UP=MOUSE_KEY+SDL_BUTTON_X2+1, MOUSE_WHEEL_DOWN };
static const struct { const char *name; int code; } mouse_keys[]={
    {"Mouse Left",MOUSE_KEY+SDL_BUTTON_LEFT}, {"Mouse Right",MOUSE_KEY+SDL_BUTTON_RIGHT},
    {"Mouse Middle",MOUSE_KEY+SDL_BUTTON_MIDDLE}, {"Mouse X1",MOUSE_KEY+SDL_BUTTON_X1},
    {"Mouse X2",MOUSE_KEY+SDL_BUTTON_X2}, {"Wheel Up",MOUSE_WHEEL_UP}, {"Wheel Down",MOUSE_WHEEL_DOWN},
};
enum { PAD_LEFT_TRIGGER=SDL_GAMEPAD_BUTTON_COUNT, PAD_RIGHT_TRIGGER }; /* triggers as buttons */
/* keys[i] needs mods[i] held too when mods[i] is not -1 (a chord: "Left Shift+Mouse Left"). */
typedef struct { int key_count, pad_count; int keys[MAX_BIND]; int mods[MAX_BIND]; int pad[MAX_BIND]; } Binding;
static Binding bindings[IN_COUNT];
static int bindings_ready;

static void bind_defaults(void) {
    /* The header comment's layout. */
    enum { NONE=-1, SHIFT=SDL_SCANCODE_LSHIFT };
    static const struct { int input; int key; int mod; } keys[]={
        {IN_DODGE,SDL_SCANCODE_SPACE,NONE}, {IN_SPRINT,SHIFT,NONE}, {IN_CROSS,SDL_SCANCODE_F,NONE},
        {IN_CROSS,SDL_SCANCODE_RETURN,NONE}, {IN_TRIANGLE,SDL_SCANCODE_Q,NONE},
        {IN_L1,SDL_SCANCODE_E,NONE}, {IN_SQUARE,SDL_SCANCODE_TAB,NONE}, {IN_SQUARE,SDL_SCANCODE_R,NONE},
        {IN_R3,SDL_SCANCODE_C,NONE}, {IN_L3,SDL_SCANCODE_LCTRL,NONE},
        {IN_OPTIONS,SDL_SCANCODE_ESCAPE,NONE}, {IN_TOUCHPAD,SDL_SCANCODE_G,NONE},
        {IN_TOUCHPAD_RIGHT,SDL_SCANCODE_BACKSPACE,NONE},
        {IN_LEFT,SDL_SCANCODE_1,NONE}, {IN_RIGHT,SDL_SCANCODE_2,NONE}, {IN_UP,SDL_SCANCODE_3,NONE},
        {IN_DOWN,SDL_SCANCODE_4,NONE},
        {IN_UP,SDL_SCANCODE_UP,NONE}, {IN_DOWN,SDL_SCANCODE_DOWN,NONE}, {IN_LEFT,SDL_SCANCODE_LEFT,NONE},
        {IN_RIGHT,SDL_SCANCODE_RIGHT,NONE},
        {IN_MOVE_UP,SDL_SCANCODE_W,NONE}, {IN_MOVE_DOWN,SDL_SCANCODE_S,NONE},
        {IN_MOVE_LEFT,SDL_SCANCODE_A,NONE}, {IN_MOVE_RIGHT,SDL_SCANCODE_D,NONE},
        {IN_LOOK_UP,SDL_SCANCODE_I,NONE}, {IN_LOOK_DOWN,SDL_SCANCODE_K,NONE},
        {IN_LOOK_LEFT,SDL_SCANCODE_J,NONE}, {IN_LOOK_RIGHT,SDL_SCANCODE_L,NONE},
        {IN_R1,MOUSE_KEY+SDL_BUTTON_LEFT,NONE}, {IN_R2,MOUSE_KEY+SDL_BUTTON_LEFT,SHIFT},
        {IN_L2,MOUSE_KEY+SDL_BUTTON_RIGHT,NONE},
        {IN_R3,MOUSE_KEY+SDL_BUTTON_MIDDLE,NONE}, {IN_R2,MOUSE_KEY+SDL_BUTTON_X1,NONE},
        {IN_L1,MOUSE_KEY+SDL_BUTTON_X2,NONE},
        {IN_UP,MOUSE_WHEEL_UP,NONE}, {IN_DOWN,MOUSE_WHEEL_DOWN,NONE},
    };
    static const struct { int input, button; } pads[]={
        {IN_CROSS,SDL_GAMEPAD_BUTTON_SOUTH}, {IN_CIRCLE,SDL_GAMEPAD_BUTTON_EAST},
        {IN_SQUARE,SDL_GAMEPAD_BUTTON_WEST}, {IN_TRIANGLE,SDL_GAMEPAD_BUTTON_NORTH},
        {IN_L1,SDL_GAMEPAD_BUTTON_LEFT_SHOULDER}, {IN_R1,SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER},
        {IN_L2,PAD_LEFT_TRIGGER}, {IN_R2,PAD_RIGHT_TRIGGER},
        {IN_L3,SDL_GAMEPAD_BUTTON_LEFT_STICK}, {IN_R3,SDL_GAMEPAD_BUTTON_RIGHT_STICK},
        {IN_OPTIONS,SDL_GAMEPAD_BUTTON_START},
        {IN_TOUCHPAD,SDL_GAMEPAD_BUTTON_BACK}, {IN_TOUCHPAD,SDL_GAMEPAD_BUTTON_TOUCHPAD},
        {IN_UP,SDL_GAMEPAD_BUTTON_DPAD_UP}, {IN_DOWN,SDL_GAMEPAD_BUTTON_DPAD_DOWN},
        {IN_LEFT,SDL_GAMEPAD_BUTTON_DPAD_LEFT}, {IN_RIGHT,SDL_GAMEPAD_BUTTON_DPAD_RIGHT},
    };
    memset(bindings,0,sizeof bindings);
    for (size_t i=0;i<sizeof(keys)/sizeof(*keys);++i) {
        Binding *b=&bindings[keys[i].input];
        b->mods[b->key_count]=keys[i].mod; b->keys[b->key_count++]=keys[i].key;
    }
    for (size_t i=0;i<sizeof(pads)/sizeof(*pads);++i) {
        Binding *b=&bindings[pads[i].input]; b->pad[b->pad_count++]=pads[i].button;
    }
}
static int pad_button_from_name(const char *name) {
    if (!SDL_strcasecmp(name,"lefttrigger")) return PAD_LEFT_TRIGGER;
    if (!SDL_strcasecmp(name,"righttrigger")) return PAD_RIGHT_TRIGGER;
    const SDL_GamepadButton b=SDL_GetGamepadButtonFromString(name);
    return b==SDL_GAMEPAD_BUTTON_INVALID ? -1 : (int)b;
}
static int key_from_name(const char *name) {
    for (size_t i=0;i<sizeof(mouse_keys)/sizeof(*mouse_keys);++i)
        if (!SDL_strcasecmp(name,mouse_keys[i].name)) return mouse_keys[i].code;
    const SDL_Scancode s=SDL_GetScancodeFromName(name);
    return s==SDL_SCANCODE_UNKNOWN ? -1 : (int)s;
}
/* Mouse look (bbport.ini): mouse_look=0 leaves the mouse alone; mouse_sensitivity (default 1:
 * ~1000 pixels/s turn the camera at full speed), mouse_invert_y=1. */
static int mouse_look=1, mouse_invert_y;
static float mouse_sensitivity=1.0f;
/* key.<input>= / pad.<input>= lines of the settings file. */
/* Motion emulation (apply_motion): -1 unknown yet, from bbport.ini pad_motion; BB_PAD_MOTION wins. */
static int motion_enabled = -1;
static void load_bindings(void) {
    bind_defaults();
    mouse_look=1; mouse_invert_y=0; mouse_sensitivity=1.0f;
    const char *path=getenv("BB_CONFIG");
    FILE *f=path ? fopen(path,"r") : NULL;
    if (!f) return;
    char line[512];
    while (fgets(line,sizeof line,f)) {
        const int keyboard=!strncmp(line,"key.",4), pad=!strncmp(line,"pad.",4);
        char *eq=strchr(line,'=');
        if (eq && !keyboard && !pad) {
            const char *value=eq+1;
            if (!strncmp(line,"mouse_look=",11)) mouse_look=atoi(value)!=0;
            else if (!strncmp(line,"pad_motion=",11) && motion_enabled<0) motion_enabled=atoi(value)!=0;
            else if (!strncmp(line,"mouse_invert_y=",15)) mouse_invert_y=atoi(value)!=0;
            else if (!strncmp(line,"mouse_sensitivity=",18)) {
                const float v=strtof(value,NULL);
                if (v>=0.05f && v<=20.0f) mouse_sensitivity=v;
            }
            continue;
        }
        if ((!keyboard && !pad) || !eq) continue;
        *eq=0;
        int input=-1;
        for (int i=0;i<IN_COUNT;++i) if (!strcmp(line+4,input_names[i])) input=i;
        if (input<0 || (pad && input>=IN_MOVE_UP)) { printf("Runtime: controls: unknown input %s\n",line); continue; }
        Binding *b=&bindings[input];
        if (keyboard) b->key_count=0; else b->pad_count=0;
        for (char *name=strtok(eq+1,",\r\n"); name; name=strtok(NULL,",\r\n")) {
            while (*name==' ') ++name;
            for (char *end=name+strlen(name); end>name && end[-1]==' ';) *--end=0;
            if (!*name) continue;
            if (keyboard) {
                /* A chord: "<modifier>+<key>" ("Keypad +" is one key: nothing after its '+'). */
                int mod=-1;
                char *plus=strrchr(name,'+');
                if (plus && plus>name && plus[1]) {
                    *plus=0;
                    mod=key_from_name(name);
                    if (mod<0) *plus='+'; else name=plus+1;
                    while (*name==' ') ++name;
                }
                const int s=key_from_name(name);
                if (s<0) printf("Runtime: controls: unknown key \"%s\" for %s\n",name,line+4);
                else if (b->key_count<MAX_BIND) { b->mods[b->key_count]=mod; b->keys[b->key_count++]=s; }
            } else {
                const int button=pad_button_from_name(name);
                if (button<0) printf("Runtime: controls: unknown gamepad button \"%s\" for %s\n",name,line+4);
                else if (b->pad_count<MAX_BIND) b->pad[b->pad_count++]=button;
            }
        }
    }
    fclose(f);
}
/* The mouse while the window holds it: buttons, and the wheel as short presses (one per notch,
 * each held and released for a few frames so the game sees it). */
static SDL_MouseButtonFlags mouse_buttons;
static int wheel_pending[2], wheel_phase[2]; /* [up, down]; phase: 0 idle, else press/release */
static uint64_t wheel_until[2];
static void mouse_wheel_step(uint64_t now, int up, int down) {
    wheel_pending[0]+=up; wheel_pending[1]+=down;
    for (int i=0;i<2;++i) {
        if (wheel_pending[i]>8) wheel_pending[i]=8;
        if (wheel_phase[i] && now<wheel_until[i]) continue;
        if (wheel_phase[i]==1) { wheel_phase[i]=2; wheel_until[i]=now+60000; } /* released */
        else if (wheel_pending[i]>0) { --wheel_pending[i]; wheel_phase[i]=1; wheel_until[i]=now+60000; }
        else wheel_phase[i]=0;
    }
}
static int mouse_key_down(int code) {
    if (code==MOUSE_WHEEL_UP) return wheel_phase[0]==1;
    if (code==MOUSE_WHEEL_DOWN) return wheel_phase[1]==1;
    return (mouse_buttons & SDL_BUTTON_MASK(code-MOUSE_KEY))!=0;
}
/* Short taps: a key pressed and released between two pad reads (a 60-100 Hz poll against a light
 * tap) would never show in SDL_GetKeyboardState. A key-down event watch latches it until the
 * next read ends (apply_keyboard), so every tap counts for at least one sample. */
static volatile uint8_t key_latch[SDL_SCANCODE_COUNT];
static bool key_watch(void *userdata, SDL_Event *e) {
    (void)userdata;
    if (e->type==SDL_EVENT_KEY_DOWN && !e->key.repeat && (int)e->key.scancode<SDL_SCANCODE_COUNT)
        key_latch[e->key.scancode]=1;
    return true;
}
static void key_watch_start(void) {
    static int started;
    if (!started && SDL_WasInit(SDL_INIT_VIDEO)) { SDL_AddEventWatch(key_watch,NULL); started=1; }
}
static int code_down(const bool *k, int code) {
    if (code>=0 && code<SDL_SCANCODE_COUNT && key_latch[code]) return 1;
    return code>=MOUSE_KEY ? mouse_key_down(code) : k && k[code];
}
/* Whether a chord on `code` (any input's) has its modifier held: the plain bindings of the key
 * then wait (Shift+Left is R2 alone). */
static int chord_held(const bool *k, int code) {
    for (int input=0;input<IN_COUNT;++input)
        for (int i=0;i<bindings[input].key_count;++i)
            if (bindings[input].keys[i]==code && bindings[input].mods[i]>=0 &&
                code_down(k,bindings[input].mods[i])) return 1;
    return 0;
}
static int key_down(const bool *k, int input) {
    for (int i=0;i<bindings[input].key_count;++i) {
        const int code=bindings[input].keys[i], mod=bindings[input].mods[i];
        if (!code_down(k,code)) continue;
        if (mod>=0 ? code_down(k,mod) : !chord_held(k,code)) return 1;
    }
    return 0;
}
/* The bound gamepad buttons' state; triggers as their analog value. */
static int pad_value(SDL_Gamepad *g, int input) {
    int value=0;
    for (int i=0;i<bindings[input].pad_count;++i) {
        const int b=bindings[input].pad[i];
        const int v=b==PAD_LEFT_TRIGGER ? trigger(SDL_GetGamepadAxis(g,SDL_GAMEPAD_AXIS_LEFT_TRIGGER))
                  : b==PAD_RIGHT_TRIGGER ? trigger(SDL_GetGamepadAxis(g,SDL_GAMEPAD_AXIS_RIGHT_TRIGGER))
                  : SDL_GetGamepadButton(g,(SDL_GamepadButton)b) ? 255 : 0;
        if (v>value) value=v;
    }
    return value;
}

/* The keys as a stick. A digital key would be the stick thrown to its stop at once, and the
 * diagonal the corner of the square (1.41 long): the game's character then snaps to the new
 * direction and slips into a run. Here the keys give a direction on the unit circle (a diagonal
 * is as long as one key), and the stick follows it as a real thumb does: its length ramps
 * (BB_KEY_RISE_MS 70, BB_KEY_FALL_MS 45) and its angle turns at BB_KEY_TURN_DPS (1100 degrees
 * a second) the short way round, so a reversal sweeps through the sides, not through a dead
 * centre. A stick almost at rest takes the new angle at once. */
typedef struct { float angle, mag; uint64_t last_us; } KeyStick;
static float key_tunable(const char *name, float def) {
    const char *env=getenv(name);
    if (!env || !*env) return def;
    const float v=(float)atof(env);
    return v>0 ? v : def;
}
static int key_stick(KeyStick *st, uint64_t now, int left, int right, int up, int down, uint8_t *x, uint8_t *y) {
    static float rise_ms, fall_ms, turn_dps;
    if (rise_ms==0) {
        rise_ms=key_tunable("BB_KEY_RISE_MS",70.0f); fall_ms=key_tunable("BB_KEY_FALL_MS",45.0f);
        turn_dps=key_tunable("BB_KEY_TURN_DPS",1100.0f);
    }
    const int tx=(right?1:0)-(left?1:0), ty=(down?1:0)-(up?1:0);
    float dt=st->last_us && now>st->last_us ? (float)(now-st->last_us)*1e-6f : 0.0f;
    st->last_us=now;
    if (dt>0.05f) dt=0.05f;
    const float target_mag=(tx || ty) ? 1.0f : 0.0f;
    if (tx || ty) {
        const float target=atan2f((float)ty,(float)tx);
        if (st->mag<0.12f) st->angle=target;
        else {
            float diff=target-st->angle;
            while (diff>(float)M_PI) diff-=2.0f*(float)M_PI;
            while (diff<-(float)M_PI) diff+=2.0f*(float)M_PI;
            const float step=turn_dps*(float)M_PI/180.0f*dt;
            st->angle += diff>step ? step : diff<-step ? -step : diff;
        }
    }
    const float rate=target_mag>st->mag ? 1.0f/(rise_ms*1e-3f) : 1.0f/(fall_ms*1e-3f);
    const float dm=rate*dt;
    st->mag += target_mag>st->mag ? (target_mag-st->mag<dm ? target_mag-st->mag : dm)
                                  : -(st->mag-target_mag<dm ? st->mag-target_mag : dm);
    if (st->mag<0.003f) { st->mag=0; return 0; }
    const float fx=cosf(st->angle)*st->mag, fy=sinf(st->angle)*st->mag;
    int vx=128+(int)lroundf(fx*127.0f), vy=128+(int)lroundf(fy*127.0f);
    *x=(uint8_t)(vx<0 ? 0 : vx>255 ? 255 : vx);
    *y=(uint8_t)(vy<0 ? 0 : vy>255 ? 255 : vy);
    return 1;
}
static KeyStick move_stick, look_stick;
/* Circle from the keyboard. Dodge (Space) is a short press whatever the key does: held, it would
 * be the run (the game tells a tap from a hold by the time). Sprint (Left Shift) holds Circle: the
 * run. Both together, Space while Shift is held, is the jump: Circle let go for a moment and pressed
 * again while the run goes on, as a gamepad's B is. While a game menu is open they do nothing, and
 * Escape is Circle (back) instead of Options. */
#define DODGE_PULSE_US 90000u
/* The jump, Shift + Space: the way the player gets it with the keys alone, Space twice while
 * running: Circle let go, pressed, let go, then held again (the run goes on). One press of it
 * only let go and pressed once, which stopped the run and gave no jump. BB_JUMP_GAP_MS (70) is
 * each pause, BB_JUMP_TAP_MS (70) the press between them. */
static uint64_t dodge_pulse_until, jump_start_us;
static uint32_t jump_gap_us, jump_tap_us;
static int dodge_was_down;
static int keyboard_circle(const bool *k, uint64_t now, int menu) {
    if (!jump_gap_us) {
        jump_gap_us=(uint32_t)(key_tunable("BB_JUMP_GAP_MS",70.0f)*1000.0f);
        jump_tap_us=(uint32_t)(key_tunable("BB_JUMP_TAP_MS",70.0f)*1000.0f);
    }
    const int dodge=key_down(k,IN_DODGE), sprint=key_down(k,IN_SPRINT);
    if (menu) { dodge_was_down=dodge; dodge_pulse_until=jump_start_us=0; return 0; }
    const int jumping=jump_start_us && now<jump_start_us+2u*jump_gap_us+jump_tap_us;
    if (dodge && !dodge_was_down && !jumping) {
        if (sprint) { jump_start_us=now; dodge_pulse_until=0; }
        else { dodge_pulse_until=now+DODGE_PULSE_US; jump_start_us=0; }
    }
    dodge_was_down=dodge;
    if (jump_start_us && now<jump_start_us+2u*jump_gap_us+jump_tap_us) {
        const uint64_t t=now-jump_start_us;
        return t>=jump_gap_us && t<jump_gap_us+jump_tap_us; /* low, the press, low */
    }
    return sprint || now<dodge_pulse_until;
}


static void apply_keyboard(PadData *d, const bool *k) {
    key_watch_start();
    /* bbport (macOS native GPU process): the window and its key events are bb-gpu's. */
    bbgpu_keyboard_take_latched(key_latch,SDL_SCANCODE_COUNT);
    const int menu=bbgpu_pointer_wanted()!=0;
    /* The way back in a game menu is Escape; Options opens the menu at other times. A press that
     * began outside a menu is Options for as long as it is held: the menu it opens must not take
     * the same press for its way back (it opened and closed at once). */
    static int options_latched;
    if (!key_down(k,IN_OPTIONS)) options_latched=0;
    else if (!menu) options_latched=1;
    for (int i=IN_CROSS;i<=IN_RIGHT;++i) {
        if (i==IN_TOUCHPAD || i==IN_TOUCHPAD_RIGHT || !key_down(k,i)) continue;
        d->buttons |= (i==IN_OPTIONS && menu && !options_latched) ? BTN_CIRCLE : input_buttons[i];
    }
    if (keyboard_circle(k,d->timestamp,menu)) d->buttons|=BTN_CIRCLE;
    if (key_down(k,IN_TOUCHPAD)) touch_click(d,0);
    if (key_down(k,IN_TOUCHPAD_RIGHT)) touch_click(d,1);
    if (key_down(k,IN_L2)) d->l2=255;
    if (key_down(k,IN_R2)) d->r2=255;
    /* A gamepad's stick is kept while no key moves it. */
    key_stick(&move_stick,d->timestamp,key_down(k,IN_MOVE_LEFT),key_down(k,IN_MOVE_RIGHT),
              key_down(k,IN_MOVE_UP),key_down(k,IN_MOVE_DOWN),&d->left_x,&d->left_y);
    key_stick(&look_stick,d->timestamp,key_down(k,IN_LOOK_LEFT),key_down(k,IN_LOOK_RIGHT),
              key_down(k,IN_LOOK_UP),key_down(k,IN_LOOK_DOWN),&d->right_x,&d->right_y);
    memset((void *)key_latch,0,sizeof(key_latch));
}

/* Mouse look: the game turns its camera at a rate set by the right stick, so the stick follows
 * the mouse's speed. Each sample adds the motion since the last one and the sum decays over
 * LOOK_TAU (linearly per sample: the level a steady speed holds, speed * gain * LOOK_TAU, is the
 * same at any frame rate). ~1000 pixels/s reach full deflection at sensitivity 1. The game ignores
 * small deflections, so a moving mouse starts past them (BB_MOUSE_DEADZONE, default 16 of 127):
 * slow aiming still turns the camera. While the mouse moves it replaces the right stick. */
#define LOOK_TAU_US 60000.0f
static float look_x, look_y;
static uint64_t look_last_us;
static uint8_t look_axis(float value, uint8_t stick, int deadzone) {
    if (value>-0.5f && value<0.5f) return stick;
    const float magnitude=value<0 ? -value : value;
    const float out=deadzone+magnitude*(127-deadzone)/127.0f;
    const int v=128+(int)(value<0 ? -out : out);
    return (uint8_t)(v<0 ? 0 : v>255 ? 255 : v);
}
/* Native mouse look (gpu/shim bbport_guest_hooks.cpp): the camera's own update takes the turn, as
 * far as the mouse moved: MOUSE_RAD_PER_PIXEL at sensitivity 1 (1000 pixels: 1.5 rad, 86 degrees).
 * The stick below only without that hook (another game version, BB_NATIVE_MOUSE=0). */
#define MOUSE_RAD_PER_PIXEL 0.0015f
static void apply_mouse_look(PadData *d, double dx, double dy, uint64_t now, int captured) {
    /* The free camera (patch) takes the mouse through the right stick below. */
    if (captured && (dx!=0 || dy!=0) && !bbgpu_free_camera()) {
        const float k=MOUSE_RAD_PER_PIXEL*mouse_sensitivity;
        /* The camera's pitch grows as it rises (looking down): the mouse moving down (dy > 0)
           looks down, as the stick pushed down does. */
        if (bbgpu_camera_look((float)dx*k,(float)(mouse_invert_y ? -dy : dy)*k)) dx=dy=0;
    }
    float dt=look_last_us && now>look_last_us ? (float)(now-look_last_us) : 0.0f;
    look_last_us=now;
    if (!captured) { look_x=look_y=0; return; }
    if (dt>LOOK_TAU_US) dt=LOOK_TAU_US;
    static int deadzone=-1;
    if (deadzone<0) {
        const char *env=getenv("BB_MOUSE_DEADZONE");
        deadzone=env && *env ? atoi(env) : 16;
        if (deadzone<0 || deadzone>100) deadzone=16;
    }
    const float gain=127.0f/(1000.0f*LOOK_TAU_US/1e6f)*mouse_sensitivity, decay=1.0f-dt/LOOK_TAU_US;
    look_x=look_x*decay+(float)dx*gain;
    look_y=look_y*decay+(float)(mouse_invert_y ? -dy : dy)*gain;
    look_x=look_x<-127 ? -127 : look_x>127 ? 127 : look_x;
    look_y=look_y<-127 ? -127 : look_y>127 ? 127 : look_y;
    d->right_x=look_axis(look_x,d->right_x,deadzone);
    d->right_y=look_axis(look_y,d->right_y,deadzone);
}

/* Button prompts (gpu/ bbport_button_prompts: the game's key guide icons drawn for the device in
 * use): the device whose input came last. 0 a PlayStation pad (the game's own icons), 1 another
 * gamepad (its labels), 2 the keyboard and mouse. The generation changes with the device, the pad
 * type and the bindings. */
static int prompt_device=-1;
static SDL_GamepadType prompt_pad_type=SDL_GAMEPAD_TYPE_UNKNOWN;
static unsigned prompt_generation=1;
static void prompt_note(SDL_Gamepad *g, int pad_active, int keyboard_active) {
    int device=prompt_device;
    const SDL_GamepadType type=g ? SDL_GetGamepadType(g) : SDL_GAMEPAD_TYPE_UNKNOWN;
    const int playstation=type==SDL_GAMEPAD_TYPE_PS3 || type==SDL_GAMEPAD_TYPE_PS4 || type==SDL_GAMEPAD_TYPE_PS5;
    if (keyboard_active) device=2;
    else if (g && (pad_active || device<0)) device=playstation ? 0 : 1;
    else if (device<0) device=2;
    if (device!=prompt_device || (device!=2 && type!=prompt_pad_type)) {
        prompt_device=device;
        if (device!=2) prompt_pad_type=type;
        __atomic_add_fetch(&prompt_generation,1,__ATOMIC_RELAXED);
    }
}
static int stick_moved(uint8_t v) { return v<128-48 || v>128+48; }
int runtime_pad_prompt_device(void) {
    return __atomic_load_n(&prompt_device,__ATOMIC_RELAXED);
}
unsigned runtime_pad_prompt_generation(void) {
    return __atomic_load_n(&prompt_generation,__ATOMIC_RELAXED);
}
/* Mouse look on (bbport.ini mouse_look): the right stick's prompt is the mouse. */
int runtime_pad_mouse_look(void) {
    return __atomic_load_n(&mouse_look,__ATOMIC_RELAXED);
}
/* A key's name for a prompt icon: short (an icon is a key cap), arrows as arrows. */
static void key_label(int code, char *out, int size) {
    static const struct { const char *sdl, *label; } names[]={
        {"Left Shift","Shift"}, {"Right Shift","RShift"}, {"Left Ctrl","Ctrl"}, {"Right Ctrl","RCtrl"},
        {"Left Alt","Alt"}, {"Right Alt","AltGr"}, {"Return","Enter"}, {"Keypad Enter","Enter"},
        {"Backspace","Bksp"}, {"Escape","Esc"}, {"CapsLock","Caps"}, {"Delete","Del"},
        {"Insert","Ins"}, {"PageUp","PgUp"}, {"PageDown","PgDn"}, {"Up","\xe2\x86\x91"},
        {"Down","\xe2\x86\x93"}, {"Left","\xe2\x86\x90"}, {"Right","\xe2\x86\x92"},
    };
    const char *name=SDL_GetScancodeName((SDL_Scancode)code);
    if (!name || !*name) name="?";
    for (size_t i=0;i<sizeof(names)/sizeof(*names);++i)
        if (!strcmp(name,names[i].sdl)) { name=names[i].label; break; }
    if (!strncmp(name,"Keypad ",7)) { snprintf(out,size,"Num%s",name+7); return; }
    snprintf(out,size,"%s",name);
}
/* The prompt of PS input `name` (input_names: cross ... look_right) on the device in use: its
 * label (UTF-8) in `label`; with prefer_mouse, a bound mouse button before a key (mouse look on).
 * Returns the kind: 0 none (the game's own icon stays), 'K' a key,
 * 'M' a mouse button (Left Right Middle X1 X2, Wheel Up, Wheel Down), 'F' a face button (A B X Y),
 * 'S' a shoulder or trigger (LB RB LT RT; L R ZL ZR), 'C' another round button or a stick (LS RS
 * Menu View ...). */
int runtime_pad_prompt(const char *name, int prefer_mouse, char *label, int size) {
    if (!name || !label || size<2) return 0;
    label[0]=0;
    int input=-1;
    for (int i=0;i<IN_COUNT;++i) if (!strcmp(name,input_names[i])) input=i;
    if (input<0) return 0;
    pthread_mutex_lock(&lock);
    if (!bindings_ready) { load_bindings(); bindings_ready=1; }
    const int device=prompt_device;
    const Binding b=bindings[input];
    /* Circle has no key of its own: the way back in a game menu is Escape (Options' key). */
    const Binding keyboard_b=input==IN_CIRCLE && bindings[input].key_count==0 ? bindings[IN_OPTIONS] : bindings[input];
    const int look=mouse_look;
    const SDL_GamepadType type=prompt_pad_type;
    pthread_mutex_unlock(&lock);
    if (device==2) {
        int key=-1, mouse=-1, key_mod=-1, mouse_mod=-1;
        for (int i=0;i<keyboard_b.key_count;++i) {
            if (keyboard_b.keys[i]>=MOUSE_KEY) { if (mouse<0) { mouse=keyboard_b.keys[i]; mouse_mod=keyboard_b.mods[i]; } }
            else if (key<0) { key=keyboard_b.keys[i]; key_mod=keyboard_b.mods[i]; }
        }
        /* A chord's modifier as a prefix: Shift as its arrow (R2: "⇧Left"). */
        char prefix[16]="";
        const int mod=mouse>=0 && ((look && prefer_mouse) || key<0) ? mouse_mod : key_mod;
        if (mod>=0) {
            if (mod==SDL_SCANCODE_LSHIFT || mod==SDL_SCANCODE_RSHIFT) snprintf(prefix,sizeof prefix,"\xe2\x87\xa7");
            else { key_label(mod,prefix,(int)sizeof(prefix)-1); strcat(prefix,"+"); }
        }
        /* With mouse look the hand is on the mouse: its button first (R1: Left), unless a key
           cluster is drawn (prefer_mouse 0: the d-pad's keys, not the wheel). */
        if (mouse>=0 && ((look && prefer_mouse) || key<0)) {
            for (size_t i=0;i<sizeof(mouse_keys)/sizeof(*mouse_keys);++i)
                if (mouse_keys[i].code==mouse)
                    snprintf(label,size,"%s%s",prefix,mouse_keys[i].name+(strncmp(mouse_keys[i].name,"Mouse ",6) ? 0 : 6));
            return 'M';
        }
        if (key<0) return 0;
        char name[32];
        key_label(key,name,sizeof name);
        snprintf(label,size,"%s%s",prefix,name);
        return 'K';
    }
    if (device!=1) return 0; /* a PlayStation pad: the game's own icons */
    const int nintendo=type==SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO ||
                       type==SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT ||
                       type==SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT ||
                       type==SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR;
    if (input>=IN_MOVE_UP) { snprintf(label,size,"%s",input<IN_LOOK_UP ? "LS" : "RS"); return 'C'; }
    if (b.pad_count==0) return 0;
    const int button=b.pad[0];
    switch (button) {
    case SDL_GAMEPAD_BUTTON_SOUTH: case SDL_GAMEPAD_BUTTON_EAST:
    case SDL_GAMEPAD_BUTTON_WEST: case SDL_GAMEPAD_BUTTON_NORTH: {
        static const char *const by_position[]={"A","B","X","Y"};
        const SDL_GamepadButtonLabel l=SDL_GetGamepadButtonLabelForType(type,(SDL_GamepadButton)button);
        const char *text=l==SDL_GAMEPAD_BUTTON_LABEL_A ? "A" : l==SDL_GAMEPAD_BUTTON_LABEL_B ? "B"
                        : l==SDL_GAMEPAD_BUTTON_LABEL_X ? "X" : l==SDL_GAMEPAD_BUTTON_LABEL_Y ? "Y"
                        : by_position[button-SDL_GAMEPAD_BUTTON_SOUTH];
        snprintf(label,size,"%s",text);
        return 'F';
    }
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: snprintf(label,size,"%s",nintendo ? "L" : "LB"); return 'S';
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: snprintf(label,size,"%s",nintendo ? "R" : "RB"); return 'S';
    case PAD_LEFT_TRIGGER: snprintf(label,size,"%s",nintendo ? "ZL" : "LT"); return 'S';
    case PAD_RIGHT_TRIGGER: snprintf(label,size,"%s",nintendo ? "ZR" : "RT"); return 'S';
    case SDL_GAMEPAD_BUTTON_LEFT_STICK: snprintf(label,size,"LS"); return 'C';
    case SDL_GAMEPAD_BUTTON_RIGHT_STICK: snprintf(label,size,"RS"); return 'C';
    case SDL_GAMEPAD_BUTTON_START: snprintf(label,size,"%s",nintendo ? "+" : "Menu"); return 'C';
    case SDL_GAMEPAD_BUTTON_BACK: snprintf(label,size,"%s",nintendo ? "-" : "View"); return 'C';
    case SDL_GAMEPAD_BUTTON_MISC1: snprintf(label,size,"Share"); return 'C';
    case SDL_GAMEPAD_BUTTON_TOUCHPAD: snprintf(label,size,"Pad"); return 'C';
    default: return 0; /* the d-pad: the game's icons fit any pad */
    }
}
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
        const char *env=getenv("BB_PAD_MOTION");
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
    const bool *k=bbgpu_keyboard_state();
    mouse_buttons=captured ? bbgpu_mouse_buttons() : 0;
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
        if (motion_on()) motion_from_pad(d,g,motion_dt);
        // Back/Select on pads without a touch surface is a left-side click.
        if ((d->buttons & BTN_TOUCHPAD) && !d->touch_count) touch_click(d,0);
        if (touch_right) touch_click(d,1);
    }
    const int pad_active=g && (d->buttons || d->l2>30 || d->r2>30 || d->touch_count ||
                               stick_moved(d->left_x) || stick_moved(d->left_y) ||
                               stick_moved(d->right_x) || stick_moved(d->right_y));
    int keyboard_active=(captured && (mouse_dx*mouse_dx+mouse_dy*mouse_dy>9.0 || wheel_up || wheel_down)) ||
                        (mouse_buttons!=0);
    for (int i=0;i<IN_COUNT && !keyboard_active;++i) keyboard_active=key_down(k,i);
    prompt_note(g,pad_active,keyboard_active);
    apply_keyboard(d,k); /* the bound mouse buttons too */
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
static uint64_t replay_start; /* 0: (re)start at the next sample */
static void read_inject(void) {
    static const char *path; static int checked; static uint64_t last_check; static struct timespec mtime;
    if (!checked) { path=getenv("BB_PAD_FILE"); checked=1; }
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
    char token[64];
    while (fscanf(f,"%63s",token)==1) {
        if (!strcmp(token,"replay") && replay_armed!=1) { replay_armed=1; replay_start=0; } /* BB_PAD_REPLAY */
        if (!strcmp(token,"touchpad_left") || !strcmp(token,"touchpad_right")) {
            injected.buttons|=BTN_TOUCHPAD;
            injected.touch_side=!strcmp(token,"touchpad_right");
        }
        for (size_t i=0;i<sizeof(names)/sizeof(*names);++i) if (!strcmp(token,names[i].name)) injected.buttons|=names[i].ps;
        for (int i=0;i<4;++i) if (!strncmp(token,sticks[i],3)) { int v=atoi(token+3); injected.stick[i]=v<0 ? 0 : v>255 ? 255 : v; }
        if (!strncmp(token,"mdx=",4)) injected.mouse[0]=(float)atof(token+4);
        if (!strncmp(token,"mdy=",4)) injected.mouse[1]=(float)atof(token+4);
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
static FILE *record_file;
static uint64_t record_start;
static PadSample record_last;
static void record_sample(const PadData *d) {
    static const char *path; static int checked, f9_was_down;
    if (!checked) { path=getenv("BB_PAD_RECORD"); checked=1; }
    if (!path || !*path || !sdl_ready) return;
    const bool *k=bbgpu_keyboard_state();
    const int f9=k && k[SDL_SCANCODE_F9];
    if (f9 && !f9_was_down) {
        if (record_file) {
            fclose(record_file); record_file=NULL;
            printf("Runtime: pad recording stopped (%s)\n",path);
        } else if ((record_file=fopen(path,"w"))) {
            record_start=now_us();
            memset(&record_last,0xff,sizeof(record_last));
            printf("Runtime: pad recording started (%s, F9 stops)\n",path);
        }
    }
    f9_was_down=f9;
    if (!record_file) return;
    PadSample s={(uint32_t)((now_us()-record_start)/1000),d->buttons,
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
    if (!replay_start) {
        static int loaded;
        if (!loaded) {
            loaded=1;
            const char *path=getenv("BB_PAD_REPLAY");
            FILE *f=path ? fopen(path,"r") : NULL;
            PadSample s; unsigned v[8]; size_t cap=0;
            while (f && fscanf(f,"%u %u %u %u %u %u %u %u",&v[0],&v[1],&v[2],&v[3],&v[4],&v[5],&v[6],&v[7])==8) {
                s=(PadSample){v[0],v[1],{(uint8_t)v[2],(uint8_t)v[3],(uint8_t)v[4],(uint8_t)v[5]},(uint8_t)v[6],(uint8_t)v[7]};
                if (replay_count==cap && !(replay=realloc(replay,(cap=cap ? cap*2 : 1024)*sizeof(*replay)))) break;
                replay[replay_count++]=s;
            }
            if (f) fclose(f);
            printf("Runtime: pad replay of %zu samples from %s\n",replay_count,path ? path : "(unset)");
        }
        replay_start=now_us();
        replay_next=0;
    }
    const uint32_t ms=(uint32_t)((now_us()-replay_start)/1000);
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
    if (injected.mouse[0]!=0 || injected.mouse[1]!=0)
        apply_mouse_look(d,injected.mouse[0],injected.mouse[1],d->timestamp,1);
    /* Game modules (a menu pointer's presses). */
    const uint8_t sticks[4]={d->left_x,d->left_y,d->right_x,d->right_y};
    bbgpu_filter_pad(&d->buttons,sticks);
    touch_ids(d);
}

static ABI int32_t pad_init(void) { pthread_mutex_lock(&lock); initialized=1; pthread_mutex_unlock(&lock); return 0; }
static ABI int32_t pad_open(int32_t user, int32_t type, int32_t index, const void *param) {
    (void)param;
    if (!initialized) return ERR_NOT_INITIALIZED;
    if (user!=1 || (type!=0 && type!=2) || index) { /* only user 1's standard / special port */
        printf("Runtime: scePadOpen(user %d, type %d, index %d) refused\n",user,type,index);
        return ERR_INVALID_ARG;
    }
    pthread_mutex_lock(&lock);
    int already=opened; opened=1;
    pthread_mutex_unlock(&lock);
    if (already) return ERR_ALREADY_OPENED;
    puts("Runtime: pad opened for user 1 (SDL gamepad or keyboard)");
    return PAD_HANDLE;
}
static ABI int32_t pad_close(int32_t handle) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    opened=0; return 0;
}
/* Reverse engineering: hardware watchpoints on the game's memory (runtime_rewatch.c). */
void runtime_re_watch_tick(const void *pad_buffer, uint64_t reads);

static ABI int32_t pad_read_state(int32_t handle, PadData *data) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!data) return ERR_INVALID_ARG;
    pthread_mutex_lock(&lock);
    sample(data); ++reads;
    pthread_mutex_unlock(&lock);
    static int trace=-1;
    if (trace<0) { const char *v=getenv("BB_PAD_TRACE"); trace=v && *v=='1'; }
    if (trace) {
        static uint32_t last_buttons; static uint64_t last_us;
        if (data->buttons!=last_buttons || data->timestamp-last_us>5000000) {
            printf("Pad: read %zu buttons 0x%x sticks %u %u %u %u\n",reads,data->buttons,data->left_x,
                   data->left_y,data->right_x,data->right_y);
            last_buttons=data->buttons; last_us=data->timestamp;
        }
    }
    runtime_re_watch_tick(data,(uint64_t)reads);
    return 0;
}
/* Buffered read: the port samples once per call, so one entry is returned. */
static ABI int32_t pad_read(int32_t handle, PadData *data, int32_t count) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!data || count<1 || count>64) return ERR_INVALID_ARG;
    pad_read_state(handle,data);
    return 1;
}
static ABI int32_t pad_info(int32_t handle, ControllerInfo *info) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!info) return ERR_INVALID_ARG;
    memset(info,0,sizeof(*info));
    info->pixel_density=44.86f; info->resolution_x=1920; info->resolution_y=943;
    info->dead_zone_left=info->dead_zone_right=2;
    info->connection_type=0; info->connected=1; info->device_class=0;
    pthread_mutex_lock(&lock);
    current_gamepad();
    info->connected_count=connected_count ? connected_count : 1;
    pthread_mutex_unlock(&lock);
    return 0;
}
static ABI int32_t pad_vibration(int32_t handle, const uint8_t *param) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!param) return ERR_INVALID_ARG;
    pthread_mutex_lock(&lock);
    SDL_Gamepad *g=current_gamepad();
    if (g) SDL_RumbleGamepad(g,(uint16_t)(param[0]*257),(uint16_t)(param[1]*257),1000);
    pthread_mutex_unlock(&lock);
    return 0;
}
static ABI int32_t pad_ok_handle(int32_t handle) { return handle==PAD_HANDLE && opened ? 0 : ERR_INVALID_HANDLE; }
static ABI int32_t pad_ok_handle_flag(int32_t handle, uint8_t flag) { (void)flag; return pad_ok_handle(handle); }

static const RuntimeExport exports[]={
    {"scePadInit",pad_init}, {"scePadOpen",pad_open}, {"scePadClose",pad_close},
    {"scePadReadState",pad_read_state}, {"scePadRead",pad_read},
    {"scePadGetControllerInformation",pad_info}, {"scePadSetVibration",pad_vibration},
    {"scePadResetOrientation",pad_ok_handle},
    {"scePadSetAngularVelocityDeadbandState",pad_ok_handle_flag}, {"scePadSetTiltCorrectionState",pad_ok_handle_flag},
    {"scePadSetMotionSensorState",pad_ok_handle_flag},
};
uintptr_t runtime_pad_resolve(const char *name) { return RUNTIME_LOOKUP(exports,name); }
void runtime_pad_report(void) { printf("Runtime: pad reads=%zu, gamepad=%s\n",reads,gamepad ? SDL_GetGamepadName(gamepad) : "none"); }
