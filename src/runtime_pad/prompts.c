// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_pad.c part: button prompts: the device in use and the labels of its keys and buttons.
 * #included by runtime_pad.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */

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
    SDL_GamepadType shown_type=type;
    if (prompt_forced>=0) {
        device=prompt_forced;
        if (device==1) shown_type=SDL_GAMEPAD_TYPE_XBOXONE;
    }
    if (device!=prompt_device || (device!=2 && shown_type!=prompt_pad_type)) {
        prompt_device=device;
        if (device!=2) prompt_pad_type=shown_type;
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
