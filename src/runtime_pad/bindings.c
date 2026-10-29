// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_pad.c part: controls: bindings (bbport.ini key./pad. lines), keyboard sticks, mouse buttons and wheel, mouse look.
 * #included by runtime_pad.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */
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
    IN_LOOK_RIGHT, IN_SPRINT, IN_DODGE, IN_WALK, IN_COUNT
};
static const char *const input_names[IN_COUNT]={
    "cross","circle","square","triangle","l1","r1","l2","r2","l3","r3","options","touchpad",
    "touchpad_right","up","down","left","right","move_up","move_down","move_left","move_right",
    "look_up","look_down","look_left","look_right","sprint","dodge","walk",
};
static const uint32_t input_buttons[IN_COUNT]={
    BTN_CROSS,BTN_CIRCLE,BTN_SQUARE,BTN_TRIANGLE,BTN_L1,BTN_R1,BTN_L2,BTN_R2,BTN_L3,BTN_R3,
    BTN_OPTIONS,BTN_TOUCHPAD,0,BTN_UP,BTN_DOWN,BTN_LEFT,BTN_RIGHT,
};
#define MAX_BIND 4
#include "../../gpu/shim/bb_env.h"
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
        {IN_WALK,SDL_SCANCODE_LALT,NONE},
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
 * ~1000 pixels/s turn the camera at full speed), mouse_invert_y=1, mouse_invert_x=1. */
/* Issue #299: bbport.ini button_prompts=auto (the device used last), playstation, xbox or keyboard. */
static int prompt_forced=-1;
static int mouse_look=1, mouse_invert_y, mouse_invert_x;
static float mouse_sensitivity=1.0f;
/* key.<input>= / pad.<input>= lines of the settings file. */
/* Motion emulation (apply_motion): -1 unknown yet, from bbport.ini pad_motion; BB_PAD_MOTION wins. */
static int motion_enabled = -1;
static void load_bindings(void) {
    bind_defaults();
    mouse_look=1; mouse_invert_y=0; mouse_invert_x=0; mouse_sensitivity=1.0f; prompt_forced=-1;
    const char *path=bb_env("BB_CONFIG");
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
            else if (!strncmp(line,"button_prompts=",15)) {
                prompt_forced=!strncmp(value,"playstation",11) ? 0 : !strncmp(value,"xbox",4) ? 1 :
                              !strncmp(value,"keyboard",8) ? 2 : -1;
            }
            else if (!strncmp(line,"mouse_invert_y=",15)) mouse_invert_y=atoi(value)!=0;
            else if (!strncmp(line,"mouse_invert_x=",15)) mouse_invert_x=atoi(value)!=0;
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
/* A Steam Deck's or Steam Controller's own controls are the gamepad (steam_pad_only): no keys. */
static atomic_int steam_only_now;
static bool key_watch(void *userdata, SDL_Event *e) {
    (void)userdata;
    if (atomic_load_explicit(&steam_only_now,memory_order_relaxed)) return true;
    if (e->type==SDL_EVENT_KEY_DOWN && !e->key.repeat && (int)e->key.scancode<SDL_SCANCODE_COUNT)
        key_latch[e->key.scancode]=1;
    return true;
}
static void key_watch_start(void) {
    static int started;
    if (!started && SDL_WasInit(SDL_INIT_VIDEO)) { SDL_AddEventWatch(key_watch,NULL); started=1; }
}
static int code_down(const bool *k, int code) {
    if (!k) return 0; /* no keyboard (or none of it may count: steam_pad_only) */
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
    const char *env=bb_env(name);
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


/* A video playing (the opening movie, the endings): Enter or Escape, or the gamepad's Options, stops
 * it at its press, and the game gets Options (how it skips). From the SrShadowy fork. */
static void video_skip(PadData *d, const bool *k) {
    static int was_down;
    const int down=(k && (k[SDL_SCANCODE_RETURN] || k[SDL_SCANCODE_KP_ENTER] || k[SDL_SCANCODE_ESCAPE])) ||
                   (d->buttons & BTN_OPTIONS);
    if (down && !was_down && bbgpu_video_active() && bbgpu_video_skip()) {
        d->buttons=(d->buttons & ~(uint32_t)(BTN_CIRCLE|BTN_CROSS))|BTN_OPTIONS;
    }
    was_down=down;
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
    /* Walk (default Left Alt, held): the left stick at 40%, the game's walking pace, from the move
     * keys or a gamepad's stick (FromSoftware's PC ports bind it the same way). */
    if (key_down(k,IN_WALK)) {
        d->left_x=(uint8_t)(128+((int)d->left_x-128)*2/5);
        d->left_y=(uint8_t)(128+((int)d->left_y-128)*2/5);
    }
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
/* Native mouse look (gpu/games/bloodborne/hooks_camera.cpp): the camera's own update takes the turn, as
 * far as the mouse moved: MOUSE_RAD_PER_PIXEL at sensitivity 1 (1000 pixels: 1.5 rad, 86 degrees).
 * The stick below only without that hook (another game version, BB_NATIVE_MOUSE=0). */
#define MOUSE_RAD_PER_PIXEL 0.0015f
/* While the mouse turns the camera, the game's own camera auto-rotation (it swings the camera
 * behind the character on strafes, made for a stick) fights it: its three stores (1.09) are NOPs
 * then, put back when the mouse is let go (#352). Only where the bytes are the game's own.
 * BB_MOUSE_AUTO_ROTATE=1: left as the game does it. */
static void mouse_auto_rotation_hold(int hold) {
    static int held=-1, off=-1;
    if (off<0) { const char *e=bb_env("BB_MOUSE_AUTO_ROTATE"); off=e && e[0]=='1'; }
    if (off || hold==held) return;
    static const struct { uintptr_t va; uint8_t game[9]; } sites[3]={
        {UINT64_C(0x80143c6e8),{0xc4,0xc1,0x7a,0x11,0x85,0x44,0x01,0x00,0x00}},
        {UINT64_C(0x80143c984),{0xc4,0xc1,0x7a,0x11,0x8d,0x30,0x01,0x00,0x00}},
        {UINT64_C(0x80143dde6),{0xc4,0xc1,0x7a,0x11,0x85,0x94,0x02,0x00,0x00}},
    };
    static const uint8_t nops[9]={0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90};
    for (int i=0;i<3;++i) {
        int prot=0; uintptr_t end=0;
        /* The game's image mapped there (not in the tests) with its code. */
        if (!runtime_memory_low_info(sites[i].va,&prot,&end) || end<sites[i].va+9) { held=hold; return; }
        const uint8_t *now=(const uint8_t *)sites[i].va;
        if (memcmp(now,sites[i].game,9) && memcmp(now,nops,9)) { held=hold; return; } /* another version */
    }
    for (int i=0;i<3;++i) {
        uint8_t *p=(uint8_t *)sites[i].va;
        const uintptr_t lo=sites[i].va&~UINT64_C(0xfff), hi=(sites[i].va+9+0xfff)&~UINT64_C(0xfff);
        if (mprotect((void *)lo,hi-lo,PROT_READ|PROT_WRITE|PROT_EXEC)==0) {
            memcpy(p,hold ? nops : sites[i].game,9);
            mprotect((void *)lo,hi-lo,PROT_READ|PROT_EXEC);
        }
    }
    held=hold;
}
static void apply_mouse_look(PadData *d, double dx, double dy, uint64_t now, int captured) {
    mouse_auto_rotation_hold(mouse_look && captured && !bbgpu_free_camera());
    /* The free camera (patch) takes the mouse through the right stick below. */
    if (captured && (dx!=0 || dy!=0) && !bbgpu_free_camera()) {
        const float k=MOUSE_RAD_PER_PIXEL*mouse_sensitivity;
        /* The camera's pitch grows as it rises (looking down): the mouse moving down (dy > 0)
           looks down, as the stick pushed down does. */
        if (bbgpu_camera_look((float)(mouse_invert_x ? -dx : dx)*k,(float)(mouse_invert_y ? -dy : dy)*k)) dx=dy=0;
    }
    float dt=look_last_us && now>look_last_us ? (float)(now-look_last_us) : 0.0f;
    look_last_us=now;
    if (!captured) { look_x=look_y=0; return; }
    if (dt>LOOK_TAU_US) dt=LOOK_TAU_US;
    static int deadzone=-1;
    if (deadzone<0) {
        const char *env=bb_env("BB_MOUSE_DEADZONE");
        deadzone=env && *env ? atoi(env) : 16;
        if (deadzone<0 || deadzone>100) deadzone=16;
    }
    const float gain=127.0f/(1000.0f*LOOK_TAU_US/1e6f)*mouse_sensitivity, decay=1.0f-dt/LOOK_TAU_US;
    look_x=look_x*decay+(float)(mouse_invert_x ? -dx : dx)*gain;
    look_y=look_y*decay+(float)(mouse_invert_y ? -dy : dy)*gain;
    look_x=look_x<-127 ? -127 : look_x>127 ? 127 : look_x;
    look_y=look_y<-127 ? -127 : look_y>127 ? 127 : look_y;
    d->right_x=look_axis(look_x,d->right_x,deadzone);
    d->right_y=look_axis(look_y,d->right_y,deadzone);
}
