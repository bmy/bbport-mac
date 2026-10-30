# Environment variables

Every `BB_*` variable the port reads, by kind. Made by `scripts/env_vars.py --doc` from
`gpu/shim/bb_env_vars.def`; edit that list, not this file. The game prints the variables set
at its start (`Environment:` in the log) and warns about `BB_*` names it does not know.

`=1: ...` / `=0: ...` / `=N: ...` say what that value does; `set: ...` means any value
turns it on (even `0`).

## Settings (87)

What players and the launchers set. The menu, bbport.ini and the launcher write most of them; set one by hand to override them.

| Variable | What it does |
|---|---|
| `BB_ANISO` | =N: anisotropic filtering of scene textures (default 16) |
| `BB_ASYNC_PIPELINES` | =0: new pipelines of passes drawn every frame compiled at once instead of skipping a draw for a frame or two |
| `BB_AUTO_RENDER_RES` | =1: render resolution picked from the measured frame rate |
| `BB_BBSC` | =1: seamless co-op (BBSC), the launcher's co-op switch |
| `BB_BBSC_LOCAL_SERVER` | =1 / auto (default) / 0: the BBSC game server on this PC (scripts/bbsc_gameserver) |
| `BB_BUTTON_PROMPTS` | =0: the game's PlayStation button icons always |
| `BB_COMMUNITY_SERVER` | URL of the online pool server (default: the community server) |
| `BB_CONFIG` | =path: the controls file (default bbport.ini) |
| `BB_CONTENT_SKU` | =full / ...: the content profile run.py prepares |
| `BB_CURSOR` | =0: the system's mouse cursor instead of the port's |
| `BB_DECK_DESKTOP_INPUT` | =1: keep keyboard and mouse on a Steam Deck in game mode |
| `BB_DISPLAY` | =N: the monitor to open the window on |
| `BB_DLSS` | =0: no DLSS even on NVIDIA RTX |
| `BB_DMEM_EXTRA_MB` | =N (0-2048, default 512): direct memory beyond the PS4's for the port's extras |
| `BB_DMEM_MB` | =N: the game's direct memory size in MiB (PS4: 5056; raised for high resolutions and large heaps) |
| `BB_EVDEV_TOUCHPAD` | =0: no DualSense/DualShock touchpad through evdev |
| `BB_FFX_DIR` | the folder with AMD's FSR DLLs for Vulkan, amd_fidelityfx_loader_vk.dll and amd_fidelityfx_upscaler_vk.dll (upscaler fsr_amd; default <user>/fsr, then beside the game) |
| `BB_FFX_FG_ML` | =1: AMD's ML frame generation (4.0.x) accepts any GPU (AMD ships it for RDNA4); =0: refuses it (FSR 3.1) |
| `BB_FFX_FG_VERSION` | =<part of a name>: the frame generation AMD's DLL runs, from those the log lists (default: theirs) |
| `BB_FFX_VERSION` | =<part of a name>: the upscaler AMD's FSR DLLs run, from those the log lists (default: theirs) |
| `BB_FPS` | =uncap (default) / 90 / 60 / 30: the frame rate (the launcher's) |
| `BB_FPS_LIMIT` | =N: frame rate limit (the launcher's) |
| `BB_FRAME_GEN` | =fsr3 / dlss / fsr_amd (1: fsr3): frame generation |
| `BB_FRAME_GEN_MULTIPLIER` | =2..4: frames shown per rendered one with frame generation |
| `BB_FRAME_STATS` | set: frame statistics in the log every 5 s and slow frames (the launcher's statistics switch) |
| `BB_FSR411_DIR` | the FSR 4.1.1 assets folder (default fsr4_411) |
| `BB_FSR411_VARIANT` | =int8 / fp8 / fp8emu / navi2 / lite: FSR 4.1.1 model variant (default: the menu's model) |
| `BB_FSR4_DIR` | the FSR 4 v07 assets folder (default fsr4_shaders) |
| `BB_FSR_SHARPNESS` | =0..1: RCAS sharpening (the menu's) |
| `BB_FULLSCREEN` | =0 / 1 / exclusive: window mode |
| `BB_GAMEPAD` | =SDL GUID or part of the name: the controller to use (the launcher's) |
| `BB_GAME_DIR` | the game's folder (with eboot.bin) |
| `BB_GAME_MENU` | =0: no port pages in the game's System menu |
| `BB_GPU_ID` | =N: the GPU to use, by Vulkan index (the launcher's) |
| `BB_GPU_PROFILE` | =1: GPU profiler: time per labelled pass in the log (the launcher's) |
| `BB_HDR` | =1: HDR output allowed |
| `BB_HUD_ANCHOR` | =0: on wide screens the HUD stays in the 16:9 middle |
| `BB_INTEL_TONEMAP_FIX` | =0 / 1: Intel CPU tone mapping fix patch off / on (default on on Intel) |
| `BB_JITTER` | =0 / 1: upscaler camera jitter (the menu's) |
| `BB_JUMP_GAP_MS` | =N (default 70): pause around a keyboard jump while running |
| `BB_JUMP_TAP_MS` | =N (default 70): press length of a keyboard jump while running |
| `BB_KEY_FALL_MS` | =N (default 45): keyboard stick release time |
| `BB_KEY_RISE_MS` | =N (default 70): keyboard stick rise time |
| `BB_KEY_TURN_DPS` | =N (default 1100): keyboard stick turn speed in degrees a second |
| `BB_LANGUAGE` | =N: the game's language, PS4 number (1 English, 8 Russian; the launcher's) |
| `BB_LARGE_HEAPS` | =1 / 0: larger graphics heaps for texture packs on / off (default: detected) |
| `BB_LAYER_MEMORY` | =1 / 0: the layer's memory module (default on non-AMD GPUs) |
| `BB_LOW_LATENCY` | =off / on / boost: NVIDIA Reflex / AMD Anti-Lag (the menu's) |
| `BB_MENU_LANGUAGE` | =ru / en / ...: the port's menu language (default: the launcher's, else the system's) |
| `BB_MENU_POINTER` | =0: no mouse pointer in the game's menus |
| `BB_MODS_ENABLED` | =0: no mods |
| `BB_MOUSE_AUTO_ROTATE` | =1: the game's camera auto rotation also with the mouse |
| `BB_MOUSE_DEADZONE` | =N (default 16 of 127): mouse look dead zone |
| `BB_NATIVE_GPU` | =0 (macOS): the GPU work in the game process instead of the native GPU process bb-gpu (the default where bb-gpu is built); =1: bb-gpu or nothing |
| `BB_NATIVE_MOUSE` | =0: mouse look through the right stick instead of the camera hook |
| `BB_NET_BACKEND` | =thd: The Hunter's Dream online pool (the launcher's online pool) |
| `BB_OBJECT_MOTION` | =0 / 1: motion vectors of moving objects (the menu's) |
| `BB_ONLINE` | =1: online play (the launcher's) |
| `BB_P2P_PORT` | =N: the UDP port for P2P online traffic (default 3658, 0: any) |
| `BB_PAD_DEADZONE` | =N (default 5): stick inner dead zone |
| `BB_PAD_DEADZONE_OUTER` | =N (default 127): stick outer limit |
| `BB_PAD_MOTION` | =0 / 1: motion sensor emulation (over bbport.ini pad_motion) |
| `BB_PATCHES` | =name,...: extra patches by name (patches.py) |
| `BB_PROCESS_PRIORITY` | =normal / above / high: process priority on Windows (default above) |
| `BB_REACTIVE` | =1: FSR reactive mask (the menu's) |
| `BB_REACTIVE_MAX` | reactive mask maximum (the menu's) |
| `BB_REACTIVE_SCALE` | reactive mask scale (the menu's) |
| `BB_REACTIVE_THRESHOLD` | reactive mask threshold (the menu's) |
| `BB_RENDER_RES` | =WxH: render resolution through the startup patch (compatibility) |
| `BB_SAVE_LOG` | =1: the log and frame statistics also kept in logs/ |
| `BB_SHADER_PROGRESS` | =0: no shader compilation progress on screen |
| `BB_SHADNET_NPID` | the shadNet account name (the launcher's) |
| `BB_SHADNET_PASSWORD` | the shadNet account password (the launcher's) |
| `BB_SHADNET_SERVER` | the shadNet server |
| `BB_SHADNET_TOKEN` | the shadNet account token (the launcher's) |
| `BB_SHADNET_WEBAPI` | the shadNet web API server |
| `BB_SHOW_FPS` | =1 / 0: the FPS counter on / off (show_fps) |
| `BB_SKIP_CPU_CHECK` | =1: skip the check of the CPU's bsr/bsf behaviour before the game starts |
| `BB_SKIP_GAME_CHECK` | =1: start with game files that did not pass the check |
| `BB_TEXTURE_QUALITY` | texture quality over bbport.ini's texture_quality |
| `BB_THD_ONLINE_ID` | The Hunter's Dream online name (the launcher's) |
| `BB_THD_TOKEN` | The Hunter's Dream account token (the launcher's) |
| `BB_TROPHY_KEY` | the trophy data key (from the environment or the port's .env) |
| `BB_UPNP` | =1: UPnP port forwarding for online play |
| `BB_UPSCALER` | =none / fsr3 / fsr4 / dlss / ...: the upscaler over the menu's (scripts, A/B) |
| `BB_UPSCALE_PRESET` | the upscaler quality preset (the menu's) |
| `BB_USER_NAME` | the PS4 user name (default Hunter) |

## Tuning (259)

Defaults to change on a machine where they misbehave, workarounds, and switches that turn a fix off for comparison.

| Variable | What it does |
|---|---|
| `BB_ADB` | path of adb for BB_APP0_ADB (default: adb) |
| `BB_ANTI_LAG` | =1: AMD Anti-Lag 2 through Mesa's layer (measured ~0 gain, #289) |
| `BB_APP0_ADB` | =serial: the game's files fetched from a phone over adb on first open |
| `BB_ASYNC_PIPELINE_THREADS` | =N: threads compiling pipelines in the background |
| `BB_ASYNC_SUBMIT` | =0: the presenter submits the frame itself instead of a submit thread |
| `BB_AUDIO` | =none: no audio device (headless runs) |
| `BB_AUTO_NOHIZ` | =0 (run.sh): no automatic RADV_DEBUG=nohiz on the GPUs that need it |
| `BB_AVPLAYER_ASYNC_READY` | =1: movie StateReady on the controller thread, as before (crashed at 200+ FPS) |
| `BB_BDA_DEDICATED_MB` | =N: buffers with GPU addresses get dedicated memory from N MiB |
| `BB_BIND_MEMO` | =0: every resource binding takes the full path (no memo) |
| `BB_BREADCRUMBS` | =0: no GPU breadcrumbs (what the GPU was drawing at a hang) |
| `BB_CHUNK_TEXTURES` | =0: texture data copied instead of read in place from guest memory chunks |
| `BB_CHUNK_UPLOADS` | =0: uploads through CPU staging instead of binding guest memory in place |
| `BB_COMMIT_IMPORT_LOCKED` | =1: guest memory commits import under the lock |
| `BB_COMPUTE_RELEASE_FLUSH` | =0: work before a compute release waits for the rest of the frame |
| `BB_CONSTANTS_IN_PLACE` | =1: shader constants bound in place from guest memory instead of copied |
| `BB_CONSTANT_RING_MB` | =N: size of the constant ring in MiB |
| `BB_CONSTRAM_GPU_WAIT` | =0: a constant RAM dump does not wait for queued draws reading it (DoF artifacts #44) |
| `BB_CONSTRAM_PIPE` | =0: a constant RAM dump waits for all queued draws |
| `BB_COPY_DEST_GAP` | =bytes: merge gap of copy shader destinations (default 256) |
| `BB_COPY_DEST_PRECISE` | =0: the whole bound destination of a copy shader counts as written |
| `BB_COPY_FILTER_GATE` | =1 with BB_COPY_SKIP_SAME: the copy filter off while the draw recording thread is over 92% busy |
| `BB_COPY_GPU_BUFFERS` | =0: command buffers decoded from guest memory instead of a copy per submission |
| `BB_COPY_MERGE_KB` | =N: copy shader ranges closer than N KiB merged (default 64) |
| `BB_COPY_PAGED_RANGES` | =0: copy shader ranges whole instead of by page |
| `BB_COPY_SHADER_NATIVE` | =0: the copy shader's list read on the CPU |
| `BB_COPY_SKIP_SAME` | =0: off (default on): of the game's copy shader calls only the copies that change their destinations run (checked on thread bb:CopyFilter) |
| `BB_COPY_THREADS` | =N (0-16): threads for large guest memory copies |
| `BB_CORE_RESERVE` | =0 / 1: keep the game's threads off a reserved core (Linux, SMT CPUs with 6+ cores) |
| `BB_CP_WAIT_SPIN_US` | =N: microseconds the command thread spins while every queue waits, then it sleeps between checks (default 100 on arm64 macOS; -1, the default elsewhere: it spins until the wait ends) |
| `BB_DEAD_IMAGES` | =0: images of freed memory left to the collector instead of dropped on the GPU thread |
| `BB_DEAD_IMAGES_LATE` | set: dead images dropped at the frame's end only, as before |
| `BB_DETILE_SKIPPED` | =1: texture levels the quality setting leaves out are still detiled and read |
| `BB_DIRECT_IMPORT_LOCKED` | =1: direct memory imports under the lock |
| `BB_DIRECT_MEMORY_ACCESS` | =1: shadPS4's direct memory access mode of the shader translator |
| `BB_DIRECT_SEGMENT` | =0: the presenter waits for the recording threads instead of taking its own command segment |
| `BB_DISABLE_HTTPS` | =1: the game's HTTPS requests go out as plain HTTP (online) |
| `BB_DRAIN_SPIN_US` | =N: microseconds the draw pipe's stage A spins in a drain before it sleeps (default 50 on macOS; -1, the default elsewhere: it spins, then yields) |
| `BB_DRAW_PIPE` | =0 / N: the draw pipeline off, or its depth |
| `BB_FAST_WRITER_FAULTS` | =1: fast path for guest write faults (off: a device loss on a GTX 1050) |
| `BB_FAULT_WINDOW` | =KiB: window unprotected around a guest write fault (default 256) |
| `BB_FFX_OWN_SHADERS` | =0: FSR frame generation with the packaged shaders instead of the port's own |
| `BB_FG_ASYNC` | =1: FSR 3.1 frame generation on a second compute queue (no gain measured) |
| `BB_FG_EARLY_RETIRE` | =0: with frame generation the frame counts as finished only after it |
| `BB_FIXED_ARENA` | =1 / 0: force the fixed arena on or off (tests) |
| `BB_FIX_CHARA_ID` | =0: without the character id guard fix |
| `BB_FIX_HIT_COLLECTOR` | =0: without the hit collector overflow fix |
| `BB_FIX_PLAY_TIME` | =0: without the play time fix at high frame rates |
| `BB_FLIP_AHEAD` | =N (default 1, 0 off): the buffer shown before a flip goes back to the game that many flips early |
| `BB_FLIP_RATE` | =N: the flip rate the game asks for replaced (0: every vblank; games without frame rate hooks) |
| `BB_FRAMES_AHEAD` | =N (default 1): guest frames the command processor may run ahead of the GPU |
| `BB_FRAME_HOOKS` | =0: the community's frame rate patches instead of the port's hooks |
| `BB_FSR3_FP32` | =1: FSR 3.1 without its FP16 shader variants (#85) |
| `BB_FSR3_WAVE32` | =1: FSR 3.1 without its wave64 shader variants (#85) |
| `BB_FSR4_LINEAR` | =0: FSR 4 gets the encoded frame as before |
| `BB_FSR4_OPT` | =0: FSR 4 v07 passes without the optimized set (opt/) |
| `BB_FX_INIT_CLEAR` | =0: the effect buffers' initialization clears through the game's path |
| `BB_GARLIC_VRAM` | =0 / 1: VRAM copies of the game's GPU memory off / on (default: on discrete GPUs) |
| `BB_GC_BUDGET_MB` | =N: texture cache budget in MiB, as on integrated GPUs |
| `BB_GC_IDLE_SECONDS` | =N (default 20): unused textures evicted after N seconds |
| `BB_GC_LIVE_BUDGET` | =0: texture cache marks from the startup budget instead of the driver's current one |
| `BB_GC_PRESSURE_IDLE_SECONDS` | =N (default 5): the same under memory pressure |
| `BB_GC_SMALL_MARKS` | =P,C: pressure and critical marks in percent on small cards |
| `BB_GC_THRASH_GUARD` | =0 (macOS): the texture collector evicts images in use even when what is over its critical mark is not images (re-uploaded every tick) |
| `BB_GEOMETRY_SHADERS` | =0: the geometry stage hidden and its draws skipped (macOS A/B: KosmicKrisp emulates it with compute passes that end the render pass) |
| `BB_GPU_COMMAND_WRITES` | =0: GPU command writes not done in place |
| `BB_GPU_DATA_CHECK` | =0: GPU-written data not checked before CPU reads (#122) |
| `BB_GPU_LABELS` | =1 / 0: GPU-written labels on / off (default off on AMD GFX8) |
| `BB_GPU_LABELS_PORTABLE` | =1: portable GPU label writes |
| `BB_GPU_QOS` | =1 (macOS experiment): the threads every frame waits on run as user-interactive work (performance cores) |
| `BB_GPU_SPIN_US` | =N: microseconds the GPU thread spins before sleeping (default 200 with 12+ threads) |
| `BB_GPU_TIMESTAMPS` | =0: timestamps written by the CPU instead of the GPU |
| `BB_GPU_WATCHDOG_MS` | =N (macOS, Metal command path): milliseconds its committed Metal work may make no progress before the process exits (default 3000; 0: off; a stuck GPU takes WindowServer down) |
| `BB_GPU_WRITES_IN_PLACE` | =1: GPU-written buffers in place with write tracking, as before |
| `BB_GPU_WRITE_TWINS` | =1: VRAM twins of GPU-written guest pages |
| `BB_GPU_WRITE_TWINS_MAX` | =N: pages with at most N GPU-written ranges get twins (default 1024) |
| `BB_GUEST_HOOKS` | =0: no guest code hooks at all (aspect ratio, HUD, mouse, fixes) |
| `BB_GUEST_LAZY` | =1: lazy guest memory with the layer's memory module (off: broke buffers) |
| `BB_GUEST_MEMORY` | =dmabuf / host / private: how guest memory is shared with the GPU (default per vendor) |
| `BB_GUEST_UNCACHED` | =1: guest memory only in memory types the GPU reads past its caches |
| `BB_GUEST_VRAM` | =1 (experiment): direct memory in VRAM mapped through the PCI BAR |
| `BB_GUEST_VRAM_RANGES` | =lo-hi,...: physical ranges for BB_GUEST_VRAM (default: the graphics heaps) |
| `BB_GX_COMMAND_HEAP` | =0: without the larger GX command heap patch (patches.py) |
| `BB_GX_QUERY_GUARD` | =0: without the GX query creation guard |
| `BB_HEAP_GROW` | =0: the graphics heap whole from the start instead of growing on demand |
| `BB_HEAP_GROW_AHEAD` | =N (default 2, 0 off): heap pieces made ahead |
| `BB_HEAP_GROW_INITIAL_MB` | =N (default 512): heap memory backed from the start |
| `BB_HEAP_GROW_PIECE_MB` | =N (default 320, at least 258): heap growth piece |
| `BB_HEAP_SHRINK` | =0: the heap only grows |
| `BB_HEAP_SHRINK_DELAY` | =seconds (default 60): a piece free this long is given back |
| `BB_HEAP_TOP_HOOK` | =0: without the hook on allocations from the heap's top |
| `BB_HIT_COLLECTOR_CAP` | =N (default 65536): hits per call before the overflow fix takes over |
| `BB_HONEST_DRAIN_FLUSH` | =0: no flush when the GPU is on its last submission |
| `BB_HONEST_FLIPS` | =1 (experiment): a flip completes once the GPU has finished the frame |
| `BB_HONEST_FLUSH_US` | =N (default 2000): submissions batched over N microseconds while the GPU has work |
| `BB_HONEST_IDLE_BACKLOG` | =bytes (default 64 KiB): queued draws that make an idle GPU get a flush |
| `BB_HONEST_IDLE_FLUSH` | =0: no flush for an idle GPU |
| `BB_HONEST_IDLE_US` | =N (default 200): microseconds of GPU idle before the guest may be waiting |
| `BB_HOST_COPY_WAITS` | =all: host copies always wait, as before |
| `BB_HOST_ONLY_DEFAULT` | =0: none of the game profile's allocations in host memory only |
| `BB_HOT_EFFECTS` | =0: effect settings applied as patches at start instead of live |
| `BB_HOT_PAGES` | =1: hot page tracking |
| `BB_HOT_VERTEX_COPY` | =1: vertex data bound in place copied into VRAM once per submission (no reads over the bus per pass) |
| `BB_IMAGE_RECYCLE_MB` | =N (default 256, 0 off): memory of freed images kept for reuse |
| `BB_IMAGE_RECYCLE_SECONDS` | =N (default 90): recycled images older than this go |
| `BB_IMAGE_SPARES` | =1 / 0: spare images made ahead on a thread (default on on Windows) |
| `BB_IMAGE_SPARES_MB` | =N: memory for spare images |
| `BB_IMAGE_TRAPS` | =0: no write traps on image memory |
| `BB_IMPORT_GRANULE_MB` | =N (2..64, default 32): lazy guest memory import granule |
| `BB_IMPORT_OVERLAP_MB` | =N: overlap budget of lazy imports |
| `BB_INDIRECT_GUARD` | =0: indirect draw and dispatch counts used as the game gives them |
| `BB_INDIRECT_GUARD_MAX` | =N (default 4M): indirect dispatch groups allowed in all |
| `BB_INDIRECT_GUARD_MAX_VERTICES` | =N (default 64M): indirect draws over this run nothing |
| `BB_INPLACE_TEXTURES` | =1: all texture data read in place (off: looked smeared) |
| `BB_INTERP_INT_FIX` | =0 / 1: flat integer interpolation fix off / on (default: on where barycentrics exist) |
| `BB_JOB_SLOT_FIX` | =0: without the render job slot leak fix |
| `BB_KEEP_DECOMPRESS_DRAWS` | =1: the game's depth decompress passes are drawn |
| `BB_KERNEL_CLAMP` | =0: without the effect kernel clamp fix |
| `BB_KIT_RESTORE` | =0: the online kit's code changes are not restored |
| `BB_LAYER_CPU_MARK_SECONDS` | =N (default 5, 0: until handed out): CPU marks of layer blocks expire after N seconds |
| `BB_LAYER_DOMAINS` | =0: mirrors not per game allocation |
| `BB_LAYER_GPU_WRITES` | =0 / 1 / 2: where GPU writes may land in VRAM copies (0 nowhere, 1 default) |
| `BB_LAYER_MIRROR_FLOOR_PERCENT` | =N (max 90): VRAM budget share kept for mirrors |
| `BB_LAYER_MIRROR_GROWTH_MB` | =N (default 64, 0 none): room a growing mirror gets |
| `BB_LAYER_MIRROR_HEADROOM_MB` | =N (0: the whole floor): mirror headroom |
| `BB_LAYER_READ_TRAPS` | =1: data the GPU wrote into a VRAM copy only is trapped for CPU reads |
| `BB_LAYER_RELOAD_RESET` | =0: blocks reloaded in place keep their state, as before |
| `BB_LAYER_STITCH` | =0: bindings across heap pieces from the arena, as before (exploding vertices) |
| `BB_LAYER_UNMAP_RESET` | =0: unmapped blocks keep their state, as before |
| `BB_LAYER_VOLATILE` | =1: volatile blocks |
| `BB_LAYER_VOLATILE_IN_PASS` | =0: volatile blocks refreshed in render passes too, as before |
| `BB_LAYER_WATCH` | =0: blocks with unannounced bytes get no watched VRAM copy |
| `BB_LAYER_WRITE_TRAPS` | =0: no write traps on layer blocks |
| `BB_LIGHT_GRID_HOOK` | =0: without the light grid hook |
| `BB_LIGHT_GRID_OPT` | =1 / 0: light grid optimization on / off, over the setting (tests) |
| `BB_LIST_RESTART` | =1 (macOS): primitive restart kept on list topologies on KosmicKrisp (off there by default: each such indexed draw cost an extra compute dispatch) |
| `BB_LIVE_HYBRID` | =1: the startup resolution patch with live presets over it |
| `BB_LOW_HEAP_TOP` | =0: the low heap at the bottom of the low region again, as before 0.3 (A/B) |
| `BB_MEM_BUDGET` | =metal (macOS): the GPU memory budget is Metal's limit only, not also at most the Mac's RAM less what macOS keeps and the game process's footprint |
| `BB_MEM_PRESSURE` | =0 (macOS): macOS's memory pressure notifications ignored (else textures go sooner, at lower marks, for 30 s after one) |
| `BB_MEM_RESERVE_MB` | =N (macOS): MiB of RAM the GPU memory budget leaves macOS (default 20 %, at least 2.5 GiB) |
| `BB_MEM_SEMAPHORE_ORDER` | =0: memory semaphores without ordering |
| `BB_METALFX_AUTO_RETRY` | =0 (macOS): MetalFX not tried again by itself after a run-time failure (FSR 3.1 until it is chosen again; default: after 30 s, 2 min and 10 min) |
| `BB_METALFX_EVENTS` | =2 (default) / 1 / 0 (macOS): MetalFX from a thread of its own / that and GPU-side waits on Vulkan / on the submission thread with CPU waits |
| `BB_METALFX_STALL_MS` | =N (macOS): MetalFX work not done N ms after it went to Metal is given up, FSR 3.1 takes over (default 1000; 0: no cutoff) |
| `BB_METALFX_SYNC` | =1 (macOS): MetalFX synchronous (the CPU waits for each frame) |
| `BB_METAL_EMULATED_WAIT_MS` | =N (macOS, BB_METAL_FRAME): milliseconds a tessellated or geometry-shader draw waits for its Metal pipeline before KosmicKrisp's emulation draws it (default 1000; 0: no wait) |
| `BB_METAL_FRAME` | =1 (macOS, arm64 bb-gpu with SPIRV-Cross): the Metal command path, most GPU work encoded in Metal directly, the rest on KosmicKrisp (docs/MACOS_METAL_RENDERER.md; needs the validation layers off) |
| `BB_METAL_HAZARD_TRACKING` | =1 (macOS, Metal command path): Metal tracks hazards on every buffer instead of the explicit ordering |
| `BB_METAL_ORDER` | =barriers (macOS, BB_METAL_FRAME): Metal encoders between two of the renderer's barriers may overlap on the GPU (default chain: one after another) |
| `BB_METAL_PRECISE` | =all (macOS, Metal command path): exact float math kept in every stage (default: the stages that make positions only) |
| `BB_METAL_SCRATCH_TRIM` | =1 (macOS, BB_METAL_FRAME): the old small scratch pool, its buffers taken out of the residency set when given back |
| `BB_MIRROR_ALLOC_PERCENT` | =N (0 off): a mirror that would take device memory past N percent is refused |
| `BB_MIRROR_COMPACT_AGE` | =N (default 30): mirrors younger than N seconds are left alone |
| `BB_MIRROR_COMPACT_MB` | =N (default 64): mirror memory without valid blocks over this is given back |
| `BB_MIRROR_COMPACT_SECONDS` | =N (default 1): mirror compaction interval |
| `BB_MIRROR_FREE_BACKLOG_MB` | =N (default 128): retired mirror memory allowed to wait |
| `BB_MIRROR_FREE_MB_PER_FRAME` | =N (default 64): mirror memory freed per frame |
| `BB_MIRROR_MERGE_GROWTH` | =0: a mirror made over neighbours grows at the top only, as before |
| `BB_MIRROR_POOL_MB` | =N: retired mirrors kept for new ones |
| `BB_MIRROR_REPROMOTE_SECONDS` | =N (default 0): blocks sent back are not copied again that soon |
| `BB_MIRROR_UNBOUND_SECONDS` | =N (0 never): blocks no binding named for N seconds leave VRAM |
| `BB_MIRROR_VRAM_ONLY` | =1: mirrors in VRAM or not at all |
| `BB_MOTION_LARGE` | =0: large skeletons left to camera motion vectors, as before |
| `BB_MOTION_MATCH` | =order: repeated draws of one mesh find last frame's by draw order (the default off macOS), not by the nearest object |
| `BB_MOTION_VERTICES_MI` | =N (1-8): object motion vertex buffer in Mi vertices |
| `BB_MSL_SAFE_MATH` | =1 (macOS, Metal command path): Metal shaders compiled with safe math everywhere (default fast math, exact where the shader marks it) |
| `BB_NET_MODULE` | =path: the online module to load (default gpu/libbbnet20.so beside bb-probe) |
| `BB_NP_COUNTRY` | =code: the online user's country (default us) |
| `BB_NP_LANGUAGE` | =code: the online user's language (default en) |
| `BB_NVIDIA_EGL` | =0: the launcher leaves the EGL vendor files as they are (#107) |
| `BB_NVIDIA_LIB_DIR` | =dir: extra folder with NVIDIA's libraries for the package |
| `BB_NVIDIA_MAILBOX` | set: keep Mailbox presentation on NVIDIA Windows instead of FIFO |
| `BB_NV_POWER` | =0: leave the NVIDIA driver profile of bb-probe.exe alone (Windows; by default its power management mode is set to prefer maximum performance unless the user chose one) |
| `BB_OCCLUSION_INTERVALS` | =0: occlusion samples counted all the time, not only while a query is open |
| `BB_OCCLUSION_LABEL_ORDER` | =0: occlusion labels may go ahead |
| `BB_OCCLUSION_QUERIES` | =0: a fixed occlusion result sequence instead of real queries |
| `BB_OF_FAST` | =1: optical flow without a search on the finest level |
| `BB_PACE_CATCHUP` | =1: frame pacing catches up after a late frame |
| `BB_PAD_CENTER` | =lx,ly,rx,ry: the sticks' neutral position |
| `BB_PAD_CENTER_CAL` | =0: no measurement of the sticks' neutral position |
| `BB_PAD_WORLD_DRAWS` | =N (default 150): a frame with this many draws counts as the world for pad timing |
| `BB_PAGED_CROSSING` | =0: paged bindings only in windows |
| `BB_PAGED_FORMATTED` | =0: formatted buffers never paged |
| `BB_PAGED_HOLES` | =1 (experiment): paged reads skip memory the game never wrote |
| `BB_PAGED_MIN_MB` | =N (default 64): bindings from this size are paged |
| `BB_PENDING_SUMMARY` | =0: pending checks without the summary level |
| `BB_PIPELINE_CACHE` | =0: no pipeline cache on disk |
| `BB_PIPE_SIGNALS` | =0: the frame's cache work on the GPU thread instead of the recording thread |
| `BB_PIPE_SPIN_US` | =N: microseconds the draw pipe's threads spin before sleeping |
| `BB_PM4_RECOVER` | =0: a command buffer the game rewrote while read stops decoding, as before |
| `BB_PM4_REPAIR` | =0: corrupt submitted command buffers decoded as they are |
| `BB_PM4_WAIT_US` | =N (default 20000): command processor wait timeout in microseconds |
| `BB_PRELOAD_BACKGROUND` | =0: cached pipelines loaded before the game starts instead of in the background |
| `BB_PREP_WORKERS` | =N (0-16): draw preparation threads |
| `BB_PRESENT_MODE` | =Mailbox / Fifo / Immediate: swapchain present mode (default Mailbox) |
| `BB_PRESENT_THREAD` | =0: present on the vblank thread instead of its own |
| `BB_PROMOTE_MB_PER_FRAME` | =N (0 none; default 32 on cards up to 4 GiB, else 256): VRAM copies per frame |
| `BB_READBACKS` | =0 / 1 / 2: GPU readbacks off / relaxed (default) / precise |
| `BB_READBACK_LINEAR` | =1: linear images read back to guest memory |
| `BB_READBACK_QUEUE` | =0: no separate queue for readbacks |
| `BB_REGS_FULL_RESET` | set: the whole register file reset, as before |
| `BB_REMOTE_FRESH_SYNC` | =1 (native GPU process): the game's fresh GPU ranges reach bb-gpu as calls that return once its buffer cache has them, before the game writes there (default: messages; to measure) |
| `BB_RENDER_RESOURCE_CACHE_MB` | =N (160: the game's own): the game's render resource cache size |
| `BB_REPAIRED_DRAWS` | =1: repaired command buffers still draw |
| `BB_REPAIRED_WAIT_MS` | =N (default 500): waits of repaired command buffers give up after this |
| `BB_RESOLVE_WORKERS` | =N (0-4): threads that check and copy draw resources (stage 2) |
| `BB_RING_DEDUP` | =0: off (default on): constants already in the constant ring with the same bytes bound there again, not copied over the bus once more |
| `BB_SCENE_DEPTH_PROXIES` | =0: depth images read at the native size, not as reduced proxies |
| `BB_SCENE_STENCIL_ZERO` | =0: no depth proxies where stencil export is missing, the scene stays native |
| `BB_SCENE_UPSIZE` | =0: a scene larger than native drawn at the native size |
| `BB_SCENE_WITHOUT_UI` | =0: cinematics without a UI pass are not finished as scenes |
| `BB_SHADOWS` | =1: shadow copies of written buffers on discrete GPUs |
| `BB_SKIP_EMPTY_DRAWS` | =1 (experiment): draws that write nothing are left out |
| `BB_SKIP_NOOP_DRAWS` | =1: draws without effect left out (#352, no gain measured) |
| `BB_SOUND_HACK` | =0: saves left untouched (no sound settings fix) |
| `BB_SPEC_MATCH` | =0: shader specializations always built and compared |
| `BB_SPIRV_PRECISE` | =vertex (macOS): exact float math (NoContraction) only in the vertex, tessellation and geometry stages, not in fragment and compute shaders |
| `BB_STAGE_PRIORITY` | =0 (default) / 1 / 2: draw stage thread priority off / above normal / highest |
| `BB_STAGING_KEEP_MB` | =N (default 512): upload staging kept when idle |
| `BB_STAGING_VRAM_IDLE_MS` | =N (default 3000): VRAM staging given back after this long unused |
| `BB_STREAM_BUFFER_MB` | =N: stream buffer size |
| `BB_SUBMIT_AHEAD` | =N (default 2): a frame's first submissions go straight to the GPU |
| `BB_SUBMIT_LOCK` | =frame: a frame's last submission decoded under the lock |
| `BB_SWAPCHAIN_IMAGES` | =N: swapchain images |
| `BB_SYNC_PIPELINE_WAIT_MS` | =N: milliseconds a draw redrawn every frame waits for a pipeline it needs at once before it goes without it until compiled (default 250 on macOS; -1, the default elsewhere: as long as it takes) |
| `BB_TESS_CLAMP` | =1 (macOS): hull shader tessellation factors clamped to the game's maximum tess level (VGT_HOS_MAX_TESS_LEVEL), as the PS4 does |
| `BB_TEXTURE_COMPRESSION` | =0 (macOS, KosmicKrisp): images made with their full usage (storage, mutable formats), so that Metal does not compress them; also KosmicKrisp patch 0006's part |
| `BB_TEXTURE_HELPER` | =0 / 1: spinning texture helper thread (default: with cores to spare) |
| `BB_TEXTURE_UPLOAD_MB_PER_FRAME` | =N (default 32, 0 off): texture bytes uploaded a frame before new textures start small |
| `BB_THD_AUTH` | The Hunter's Dream account service URL |
| `BB_THD_P2P_ADDR` | our address for The Hunter's Dream peers (default: detected) |
| `BB_THD_RULESET` | The Hunter's Dream ruleset (default vanilla) |
| `BB_THD_SERVER` | The Hunter's Dream session service URL |
| `BB_THD_STUN` | =host[:port] / off: STUN server (default: the server's host on 3478) |
| `BB_THD_VERIFY_TLS` | =0: The Hunter's Dream TLS certificates not checked |
| `BB_TILE_BIND_VRAM` | =0: tiled data read in place over the bus instead of from a VRAM copy |
| `BB_TIMELINE_POLL_US` | =N (default 50): timeline semaphore poll interval |
| `BB_UNIFIED_MEMORY` | =1 / 0: treat the GPU as unified memory or not (over the device type) |
| `BB_VBAO_BLUR` | =N (default 2, 0: raw): VBAO denoise taps per side |
| `BB_VBAO_BLUR_DEPTH` | =f (default 0.05): relative depth difference that stops the VBAO blur |
| `BB_VBAO_BLUR_NORMAL` | =f (default 8, 0 ignores normals): how sharply the VBAO blur stops at normal changes |
| `BB_VBAO_NORMALS` | =0: VBAO from depth only |
| `BB_VBAO_POWER` | =f (default 1.5): VBAO strength |
| `BB_VBAO_RADIUS` | =metres (default 1.5): VBAO radius |
| `BB_VBAO_THICKNESS` | =metres (default 0.4): VBAO assumed thickness |
| `BB_VERTEX_READ_COPY` | =1 (experimental): vertex data bound in place copied into VRAM for the vertices each draw reads (once per guest submission) |
| `BB_VK_RECORD_THREAD` | =0: Vulkan commands recorded on the calling thread |
| `BB_VK_RECORD_THREADS` | =N (1-8): Vulkan recording threads (default 2 from 8 hardware threads) |
| `BB_VK_SEGMENT_KB` | =N (default 64): recording segment size |
| `BB_VMA_BLOCK_MB` | =N (default 64): VMA block size |
| `BB_VMA_DEDICATED_MB` | =N (default 4): images from this size get dedicated memory |
| `BB_VRAM_FLOOR_IDLE_SECONDS` | =N (default 30): mirror floor idle time before textures make room |
| `BB_VRAM_IDLE_SECONDS` | =N (default 60): VRAM copies unused this long are given back |
| `BB_VRAM_LIMIT_MB` | =N: the driver's VRAM budget taken as at most N MiB |
| `BB_VRAM_PRESSURE_IDLE_SECONDS` | =N (default 10): under pressure, VRAM groups unused this long go |
| `BB_VRAM_PROMOTE_PERCENT` | =N (default 90): no new VRAM copies past this share of the budget |
| `BB_VULKANINFO` | path of vulkaninfo for the launcher |
| `BB_WORK_RETIRED` | =0: the recording thread and fences do not wait for retired work |

## Diagnostics (305)

Logs, dumps, traces, probes and experiments, for finding bugs. Most cost frame time; none is meant for playing.

| Variable | What it does |
|---|---|
| `BB_ARENA_VRAM_ONLY` | =1 (tests): the arena in VRAM only, as NVIDIA's sparse buffers |
| `BB_AUDIO_DUMP` | =dir: raw converted PCM per audio port |
| `BB_AUDIO_STATS` | set: audio port statistics |
| `BB_AUDIO_TRACE` | =1: Ajm instance configurations and jobs that fail |
| `BB_AVPLAYER_TRACE` | =1: the game's movie player calls with results and times |
| `BB_BARRIER_STATS` | =1: images whose subresource states cost the most |
| `BB_BBSC_MEMSCAN` | set: BBSC finds its sites by scanning memory instead of the known addresses |
| `BB_BBSC_PROBE` | =address: BBSC probe of that guest address |
| `BB_BDA_LOG` | set: GPU addresses of buffer sources, to place a GPUVM fault |
| `BB_BIG_ALLOC_STACKS` | =1: call stacks of device memory allocations of 64 MiB and more |
| `BB_BREADCRUMB_READS` | =1 / 0: breadcrumbs of T# and V# reads everywhere / nowhere (default: NVIDIA only) |
| `BB_BUFFER_STATS` | =1: how many frames behind the GPU is when a frame starts |
| `BB_CAMERA_FRAME_LOG` | set: projection of every G-buffer pass's camera with its GPU frame |
| `BB_CAMERA_INVERSE` | =own: motion vectors from the inverse of the view matrix instead of the game's stored one |
| `BB_CAMERA_LOG` | set: camera position every 100 ms |
| `BB_CAMERA_Y` | =up / down: force the camera's y direction for motion vectors |
| `BB_CAMLOG` | set: camera yaw when it changes (#325) |
| `BB_CAPTURE_DIR` | folder for frame captures (BB_CAPTURE_TRIGGER) |
| `BB_CAPTURE_TRIGGER` | =file: creating that file records the next frame for the frame analyzer |
| `BB_CE_STATS` | =1: what the constant engine's dumps feed |
| `BB_CE_WAIT_LOG` | set: constant engine waits that were not satisfied |
| `BB_CE_WATCH` | =hex address: constant RAM dumps and WRITE_DATA packets over that address |
| `BB_CLOTH_STIFFNESS` | =value: cloth stiffness bound at any frame rate |
| `BB_COMPILE_TRACE` | =1: every shader and pipeline compiled during the session |
| `BB_CONSTRAM_WAIT_LOG` | set: constant RAM dumps that waited for the GPU, sampled |
| `BB_COPY_SHADER_TRACE` | set: the copy shader's buffers every 2 s |
| `BB_COPY_WHERE` | =1: the copy shader's sources and destinations per frame, VRAM or in place (log lines) |
| `BB_DCB_STATS` | =1: structure of the graphics command buffers |
| `BB_DEBUG_MOTION` | =1: motion vectors blended over the picture |
| `BB_DISPATCH_LOG` | =1: the shape of each kind of compute dispatch, once |
| `BB_DISPATCH_START_LOG` | set: dispatches with a nonzero thread group start |
| `BB_DISPLAY_WATCH` | set: every draw into a display buffer with its sizes and state |
| `BB_DISPLAY_WATCH_TARGET` | =hex address: also every draw into that color target |
| `BB_DLSSG_CAMERA_MOTION` | =0: DLSS-G told the vectors have no camera motion |
| `BB_DLSSG_DEPTH_INVERTED` | =1: DLSS-G told the depth is the other way round |
| `BB_DLSSG_HUDLESS` | =0: DLSS-G gets no hudless frame |
| `BB_DLSSG_MATRICES` | =0: DLSS-G gets no camera matrices |
| `BB_DLSSG_MVEC_SCALE` | =f: DLSS-G motion vector scale (default 1) |
| `BB_DLSSG_PREFILL` | =0: DLSS-G outputs not filled with the real frame first (black frames when it skips) |
| `BB_DLSSG_SUBRECTS` | =0: DLSS-G gets no input sub-rectangles |
| `BB_DRAW_WATCH` | =1: draws whose viewport, targets or textures differ from the previous frame's |
| `BB_DRAW_WATCH_TEXTURES` | set: BB_DRAW_WATCH also lists bound textures |
| `BB_DUMP_DIR` | =dir: where upscaler dumps go (default out/dump) |
| `BB_DUMP_EVERY` | set: dump every frame while the trigger file exists |
| `BB_DUMP_FRAMES` | =N: frames per upscaler dump (default 8) |
| `BB_DUMP_GOT` | =offset: which host function a guest GOT slot calls |
| `BB_DUMP_ONLY` | =name,name: dump only those upscaler images |
| `BB_DUMP_SHADERS` | =1: translated shader programs written to the dump folder |
| `BB_DUMP_TRIGGER` | =file: creating it dumps the upscaler's inputs and outputs |
| `BB_DYNAMIC_STATE_ALWAYS` | =1: all dynamic state set again at every draw |
| `BB_DYNSTATE_CHECK` | =1: the scissor each command buffer really has, and who set it |
| `BB_EFFECT_DUMP` | set: the game's effect options objects every 10 s |
| `BB_EXPERIMENT_CYCLE` | =bits,bits,...:seconds: the experiment bits step through the list, over and over (test builds) |
| `BB_EXP_BARRIER_ALL` | set (experiment): a full memory barrier before every draw and dispatch |
| `BB_EXP_BARRIER_TARGET` | =hex address (experiment): a full barrier before each draw into that target |
| `BB_EXP_CONSTRAM_FINISH` | =1 (experiment): every constant RAM dump waits for all earlier GPU work |
| `BB_EXP_COPY_SRC_SNAPSHOT` | set (experiment): copy shader sources snapshotted as with the old memory model |
| `BB_EXP_CPU_READS_GPU_WAIT` | set (experiment): CPU-side reads also wait for the GPU, as before |
| `BB_EXP_DMA_UNTRACKED` | set (experiment): stages reading memory through pointers count as unlisted reads |
| `BB_EXP_EXTRA_BREAKS` | =N (experiment): the render pass ended before every Nth draw, to measure breaks |
| `BB_EXP_FINISH_BEFORE_TARGET` | =hex address (experiment): the GPU finishes everything before a draw into that target |
| `BB_EXP_FINISH_FRAMES` | set (experiment): the GPU finishes each frame before the next |
| `BB_EXP_FLIP_FINISH` | set (experiment): the flip is signalled once the GPU has run the frame |
| `BB_EXP_FLIP_LAG` | set (experiment): at most one frame in flight |
| `BB_EXP_NOTED_GPU_WAIT` | set (experiment): constant RAM waits cover every read noted on stage A, as before |
| `BB_EXP_NO_MEMORY_HOOKS` | set (experiment): no guest hooks for GPU allocations and copies |
| `BB_EXP_NO_STREAM` | set (experiment): nothing goes through the stream buffer |
| `BB_EXP_PAGED_IN_PLACE` | =1 (experiment): paged buffers bound in place |
| `BB_EXP_TRANSIT_TARGET` | =hex address (experiment): that target goes through TransferSrcOptimal |
| `BB_FENCE_DELAY_TEST` | =1 (test): every 200th fence comes 80 ms late |
| `BB_FFX_FG_DEBUG` | =1: AMD's frame generation draws its tear lines and reset marks |
| `BB_FG_DUMP_TRIGGER` | =file: creating it saves real and generated frames, raw |
| `BB_FG_LUT` | =0: generated frames without the game's display LUT |
| `BB_FG_PROFILE` | =1: GPU time of each FSR 3.1 frame generation pass |
| `BB_FILE_TRACE` | =1: every guest path the game opens or checks (mods, missing files) |
| `BB_FINAL_DUMP_TRIGGER` | =file: creating it saves the presented frame |
| `BB_FI_CHECK` | =1: frame interpolation samples drawn as their difference to the SDK's |
| `BB_FPS_PRINT` | =1: the present rate every 5 s, without BB_FRAME_STATS's cost |
| `BB_FRAME_DT_SMOOTH` | =a (0..1): the game's frame step follows the measured frame time smoothed by a |
| `BB_FRAME_GEN_ONLY` | set: only the generated frames are shown |
| `BB_FRAME_LOG` | =path (with BB_FRAME_STATS): one line per flip |
| `BB_FREE_CHECK` | =1: guest heap corruption diagnostic around the free list |
| `BB_FSR411_POSTPASS` | =lds / orig: FSR 4.1.1 post pass variant |
| `BB_FSR4_MODEL_CYCLE` | =N: the FSR 4 model switched to the next one every N frames |
| `BB_FSR4_PROFILE` | =1: FSR 4.1.1 GPU time per pass every 300 frames |
| `BB_FSR4_STATS` | =1: driver statistics of every FSR 4 pass (fsr4-bench --stats) |
| `BB_GAME_ASSERT_STOP` | =1: the game's known harmless asserts stop it, as before |
| `BB_GAME_HUGE` | =0 (tests): no huge pages for the game's image |
| `BB_GAME_MENU_TRACE` | =1: how often the game asks for the port's menu texts |
| `BB_GAME_PROFILE` | =none: no game profile (hooks, fixes), the plain runtime |
| `BB_GARLIC_VRAM_RANGE` | =begin:end: VRAM copies only for guest addresses in that range |
| `BB_GC_TRIGGER_MB` | =N: the texture collector starts at N MiB (tests) |
| `BB_GNM_CENSUS` | =1: calls per Gnm function, printed |
| `BB_GNM_CENSUS_FILE` | =path: every Gnm function's total calls written there |
| `BB_GNM_CENSUS_RANGE` | =first:last: census only for those table indices (bisecting) |
| `BB_GNM_NATIVE_ONLY` | =1: native Gnm observers without checksum analysis |
| `BB_GNM_OBSERVE` | =1: submitted command buffers checked against the hooked API calls' packets |
| `BB_GNM_RACE_STACKS` | =1: call stacks of racing Gnm command buffer writers |
| `BB_GNM_TRACE` | =offset[:count]: arguments, caller and packets of that Gnm function |
| `BB_GNM_WRITES_TRACE` | =1: recent Gnm builders overlapping a bad submitted buffer |
| `BB_GPU_LOG` | =trace / debug / info / warning / error: GPU layer log level |
| `BB_GPU_PROFILE_DRAWS` | =1: the GPU profiler times every draw |
| `BB_GPU_PROFILE_TOP` | =N: the GPU profiler prints the N heaviest labels (default 30) |
| `BB_GPU_WRITE_WATCH` | =hex address: GPU writes noted over the 64 KiB page holding it |
| `BB_GS_DEBUG` | =1 (macOS, Metal command path): geometry-shader draws write magenta to their first float attachment, without blending |
| `BB_GUEST_HOOK_TRAPS` | =1: guest hooks stay int3 traps instead of native detours |
| `BB_GUEST_WHOLE_CHUNKS` | =1 (tests): as if offset mappings failed (the NVIDIA path) |
| `BB_GX_QUERY_FAIL` | =N: every Nth GX query creation fails as with a full heap |
| `BB_HEAP_DUMP` | =addr,...: those guest objects and the ones they point to |
| `BB_HEAP_FIND` | =hex value: once, 60 s in, where the game's data holds that value |
| `BB_HEAP_HOOKS` | =0 (bisecting): the heap's ranges not reported |
| `BB_HEAP_SITES` | =1: the game's malloc replacement sites |
| `BB_HEAP_TRACE` | =1: which 16 MiB ranges the game's GPU allocators hand out, every 60 s |
| `BB_HEAP_WALK` | =core,...: walk of the game's range allocator cores |
| `BB_HLE_OFF` | =mask (experiment): 1 image copy, 2 metadata clear, 4 image clear run as real shaders |
| `BB_HLE_STATS` | set: how often the compute shader substitutions fire |
| `BB_HOST_ONLY` | =phys,... (experiment): those direct allocations in host memory only |
| `BB_HUD_DX` | =f: another HUD shift |
| `BB_IMAGE_CLEAR_LOG` | =1: why an image clear went to the real shader, once per reason |
| `BB_IMAGE_COPY_LOG` | set: also why a large copy shader was not substituted |
| `BB_IMAGE_SYNC_LOG` | set: images refreshed most per 5 s, with their flags |
| `BB_IMAGE_TRACE` | =1: as BB_FILE_TRACE=1 (game file opens and reads, on the texture trace's clock) |
| `BB_IMAGE_WATCH` | =address: what happens to the image at that address |
| `BB_IME_TEXT` | =text: the on-screen keyboard answers with that text at once (tests) |
| `BB_IMITATE_NVIDIA` | =1 (tests on AMD): the port treats this GPU as NVIDIA's |
| `BB_IMPORT_LOG` | set: lazy import requests per 10 s |
| `BB_IMPORT_TRACE` | =lo-hi (MiB): who asked for imports starting there |
| `BB_IMPORT_WRITTEN_BUDGET` | =1 (tests): written bindings under the import budget too, as before |
| `BB_INPLACE_READ_CHECK` | =1: checks of the GPU's in-place reads of game memory |
| `BB_LABEL_TRAP` | =1 / 2: with BB_FREE_CHECK, read-only label pages first |
| `BB_LANG_TABLES` | set: the sell regions' language tables (#269) |
| `BB_LAYER_CHECK` | =1: stop on a stale page table entry of the layer |
| `BB_LAYER_MARK_LOG` | =lo-hi (hex): who marks layer blocks there as the CPU's |
| `BB_LAYER_PAGED` | =0 (tests): huge bindings cut in place instead of the page table |
| `BB_LAYER_PAGED_STATS` | set: paged bindings per shader every 5 s |
| `BB_LAYER_PROBE` | =1: GPU writes kept in place because the CPU once wrote the block |
| `BB_LAYER_VERIFY` | =1 / 2: verify the layer's mirrors (2: also those a paged binding reads) |
| `BB_LAYER_WATCH_HOT` | =1 (experiment): often written blocks watched and uploaded after every write |
| `BB_LAYER_WRITE_LOG` | set: large GPU writes kept in place, and the images near them |
| `BB_LIST_UNRESOLVED` | =1 (porting another game): every unresolved import, once |
| `BB_LOG_LONG_CB_MS` | =N (macOS, Metal command path): Metal command buffers whose GPU time, or the GPU's idle time before them, reaches N ms are logged |
| `BB_LOW_INFO_TEST` | set: test of the memory queries for the game's statics (#282) |
| `BB_MEMORY_LOCK_REPORT_MS` | =N (default 20, 0 off): memory map lock holds longer than this, with the code |
| `BB_MENU_KEYS_FILE` | =file (scripted tests): menu key tokens read from it |
| `BB_MENU_POINTER_FILE` | =file (tests): menu pointer moves and clicks read from it |
| `BB_MENU_POINTER_LOG` | =1: the signals of the in-menu check, with every change |
| `BB_METALFX_DEPTH_REVERSED` | =1 (macOS): MetalFX told the depth is reversed (A/B of the conventions) |
| `BB_METALFX_INVERT_JITTER` | =1 (macOS): MetalFX gets the jitter with the other sign (A/B) |
| `BB_METALFX_INVERT_MOTION` | =1 (macOS): MetalFX gets the motion vectors negated (A/B) |
| `BB_METALFX_TEST_STALL` | =N (macOS test): the 600th frame of each of MetalFX's first N worker runs waits 10 s on the GPU |
| `BB_METAL_DESCRIPTOR_SETS` | =0 (macOS, BB_METAL_FRAME): draws reading descriptor sets go to KosmicKrisp (bisecting) |
| `BB_METAL_FENCE_CHECK` | =0 (macOS, Metal command path): fence waits not checked on the CPU (on by default: waits that could stop the GPU are counted and the first logged) |
| `BB_METAL_FLAT_LAST` | =1 (macOS, BB_METAL_FRAME): draws with flat inputs from the last vertex stay in Metal (which takes them from the first) |
| `BB_METAL_FORMAT_VARIANTS` | =0 (macOS, BB_METAL_FRAME): no depth/stencil format variants of a pipeline for the pass it draws in (bisecting) |
| `BB_METAL_FRAME_TEST` | =1 (macOS, with BB_METAL_FRAME=1): ordering self-test of KosmicKrisp's and the Metal path's writes at the renderer's start (PASS or what landed out of order) |
| `BB_METAL_GENERIC` | =0 (macOS, Metal command path): the renderer's own pipelines stay on KosmicKrisp |
| `BB_METAL_GEOMETRY` | =0 (macOS, BB_METAL_FRAME): geometry-shader draws stay on KosmicKrisp (bisecting) |
| `BB_METAL_GEOMETRY_INDIRECT` | =0 (macOS, BB_METAL_FRAME): indirect geometry-shader draws stay on KosmicKrisp |
| `BB_METAL_GPU_DIAG` | =1 (macOS, BB_METAL_FRAME): encoder status on GPU errors, labelled encoders and a signpost per draw and dispatch |
| `BB_METAL_INTEROP_TEST` | =1 (macOS): Vulkan/Metal ordering test on KosmicKrisp's queue at the renderer's start (PASS or where it failed); =2: without the render pass's barriers (a negative control) |
| `BB_METAL_KEEP_AFTER_ERROR` | =1 (macOS, BB_METAL_FRAME): encoding in Metal goes on after a GPU error (else everything goes to KosmicKrisp from then on) |
| `BB_METAL_SHADOW` | =1 (macOS, bb-gpu with the Metal command path): every game pipeline also built as a Metal pipeline state, the results counted in the log (BB_METAL_FRAME=1 does it too) |
| `BB_METAL_STRIP_RESTART` | =0 (macOS, BB_METAL_FRAME): strips drawn without primitive restart go to KosmicKrisp (as before) |
| `BB_METAL_TESS` | =0 (macOS, BB_METAL_FRAME): tessellated draws stay on KosmicKrisp (bisecting) |
| `BB_METAL_TESS_INSTANCED` | =0 (macOS, BB_METAL_FRAME): instanced tessellated draws stay on KosmicKrisp |
| `BB_METAL_TESS_PATCH_INPUTS` | =0 (macOS, Metal command path): an evaluation shader keeps its own patch constants struct (one sized unlike the control shader's leaves the pipeline to KosmicKrisp) |
| `BB_METAL_TRACE` | =1: a layer over the renderer's Vulkan calls counts them; calls per second in the log every 5 s (docs/MACOS_METAL_RENDERER.md) |
| `BB_METAL_TRANSFERS` | =0 / kinds (macOS, BB_METAL_FRAME): transfers in Metal (default 1: all); 0 none; a comma-separated list of copy, fill, update, b2i, i2b, image, clear: only those; -kind: all but those |
| `BB_METAL_WATCHDOG` | =0 (macOS, BB_METAL_FRAME): no watchdog logging KosmicKrisp submissions and presents not done after 1 s and 3 s |
| `BB_META_FIXED_BASE` | set: shader prediction from metadata at a fixed program base |
| `BB_META_NO_SYNTH` | set: shader prediction without synthesized inputs |
| `BB_META_RECIPE_FETCH` | set: shader prediction without generated fetch shaders |
| `BB_MIRROR_TRACE` | set: mirror creation and growth |
| `BB_MOD_TRACE` | =1: every file a mod replaces |
| `BB_MOTION_SELECT_LOG` | =1: each G-buffer vertex shader once with the motion vector selection |
| `BB_MOTION_TRACE` | =file: the frame after the file appears prints its motion draws |
| `BB_MSG_TRACE` | =1: the game's text lookups per 2 s by category (#255) |
| `BB_NGX_LOG` | =1 / 2: NVIDIA NGX log into the port's (2: verbose) |
| `BB_NULL_GPU` | =1: no GPU work at all (tests) |
| `BB_OBJECT_MOTION_ALL` | =1: the old broad object motion selection (A/B) |
| `BB_OCCLUSION_READ_TRACE` | =1: the game's occlusion result reads |
| `BB_OCCLUSION_SAMPLES` | =N: samples the game read, for N queries |
| `BB_OCCLUSION_TRACE` | =N: the first N occlusion events with frame and render pass |
| `BB_OVERLAY_OPEN` | set (tests): the port's menu open from the start, unfolded |
| `BB_OVERLAY_TRACE` | =1: the menu's mouse buttons as SDL and ImGui see them |
| `BB_PAD_CLOCK` | =wall: pad timestamps from the wall clock, as before |
| `BB_PAD_FILE` | =file: scripted pad input for automated runs |
| `BB_PAD_RECORD` | =file: F9 starts and stops recording the pad state into it |
| `BB_PAD_REPLAY` | =file: plays a BB_PAD_RECORD recording |
| `BB_PAD_TAP_TEST` | =ms: the right stick fully right for that long (#325) |
| `BB_PAD_TRACE` | =1 / 2: gamepad buttons and touches as read (2: every sample) |
| `BB_PAGED_COPY_CROSSING` | =0 (tests): copy programs paged for size alone are not paged |
| `BB_PAGED_VERIFY` | =1: verify every block a paged binding reads |
| `BB_PARTIAL_TRACE` | set: partial image refreshes |
| `BB_PASS_BREAK_TRACE` | =1: what ended each render pass |
| `BB_PASS_DUMP` | =addr,...: rows of those render targets at each display pass (with BB_PASS_DUMP_TRIGGER) |
| `BB_PASS_DUMP_AT` | =hex address: BB_PASS_DUMP's images also before the first draw into that target |
| `BB_PASS_DUMP_STEP` | =N (default 8): BB_PASS_DUMP every Nth row |
| `BB_PASS_DUMP_TRIGGER` | =file: creating it starts BB_PASS_DUMP |
| `BB_PENDING_STATS` | set: pending notes and granules a second, by level |
| `BB_PIN_CLEAR_LOG` | set: large releases of GPU-pinned memory that were cleared |
| `BB_PIPE_VERIFY` | =N: verify the draw pipe's results every Nth frame |
| `BB_PM4_SELFTEST` | =1: tests of command processor packets Bloodborne never sends |
| `BB_PM4_SELFTEST_STEPS` | =mask: the self test steps to run |
| `BB_POOL_WATCH` | =1: watch of the game's pool corruption (#32) |
| `BB_PRESENT_DUMP_COUNT` | =N: BB_PRESENT_DUMP_TRIGGER saves N consecutive frames |
| `BB_PRESENT_DUMP_TRIGGER` | =file: creating it saves the presented frame with the UI |
| `BB_PRESET_FILE` | =file: 'preset [output [upscaler [aspect]]]' read about once a second (scripted tests) |
| `BB_PRODUCER_CHECK` | =1: command producer checks (the launcher's crash diagnostics) |
| `BB_PROMPT_TRACE` | =1: key guide icon textures found for button prompts |
| `BB_READBACK_LOG` | =path: the newest GPU-written bindings behind readbacks |
| `BB_READ_SAMPLE` | =spec: sampling of which guest memory the CPU reads |
| `BB_RECORDER_TRACE` | =1: most frequent command buffer callers every 2000 calls |
| `BB_REGS_CHECK` | set: registers compared with the defaults after a reset |
| `BB_RESIDENCY_CHUNKS` | =0: VRAM copies of guest memory each in device memory of their own, without 64 MiB chunks or reused slots (sparse binds on drivers that may mishandle shared chunks) |
| `BB_RESIDENCY_MIX` | =1 / 2: bindings per kind every 2 s |
| `BB_RESIDENCY_REPORT` | =file: when it appears, blocks bound in place and why they have no VRAM copy |
| `BB_RESIDENCY_VRAM_LIMIT_MB` | =N (tests): pretend VRAM ends at N MiB |
| `BB_RES_STATS` | =1: where bound shaders read their resource descriptors from |
| `BB_RETINA` | =0 (macOS): the window's drawable sized in points instead of pixels |
| `BB_RE_ADVANCE` | =1: probe of Scaleform's MovieImpl::Advance |
| `BB_RE_ANALOG` | =1: probe of the analog getter |
| `BB_RE_CAMERA` | =1: camera probe |
| `BB_RE_CAMSTICK` | =1: probe of the camera stick getter |
| `BB_RE_CAM_ADD` | =radians: turns the camera by that much (probe) |
| `BB_RE_ENTRY` | =offset,...: entry probes at those guest offsets |
| `BB_RE_GX_QUERY` | =1: first GX query creations and guard events with their thread |
| `BB_RE_VIEWPORT` | =1: the game's viewport calls |
| `BB_RE_WATCH` | =offset: hardware watchpoint on that guest offset |
| `BB_RE_WATCH_ADDR` | =address: hardware watchpoint on that address |
| `BB_RE_WATCH_AT` | =N (default 3000): watchpoint set after that many pad reads |
| `BB_RE_WATCH_CAMANGLE` | =1: watch who writes the camera angle |
| `BB_RE_WATCH_CAMIN` | =1: watchpoint on the camera input |
| `BB_RE_WATCH_CAMSTATE` | =1: watch who reads the camera's aspect ratio |
| `BB_RE_WATCH_FILE` | =file: watch the hex address written in it once it appears |
| `BB_RE_WATCH_FOLLOW` | =N: follow the watched bytes through host copies N times |
| `BB_RE_WATCH_INDEX` | =N (default 62): virtual input index for BB_RE_WATCH_VIRTUAL |
| `BB_RE_WATCH_VIRTUAL` | =1: watchpoint on the virtual right stick X |
| `BB_RE_ZERO_VIRTUAL` | =1: virtual inputs fed from the right stick's X get 0 |
| `BB_RING_REASONS` | set: why small read-only bindings were not copied, per 5 s |
| `BB_ROBUSTNESS` | =0 (experiment): buffer and image robustness off |
| `BB_SAMPLE_ALL` | =1: every sampled address as module+offset |
| `BB_SAMPLE_RANGE` | =lo-hi: guest offsets the sampler reports |
| `BB_SAMPLE_REPORT_SECONDS` | =N: thread samples every N seconds |
| `BB_SAMPLE_THREAD` | =name: where that thread runs, sampled |
| `BB_SAVE_TRACE` | =1: every operation on save files |
| `BB_SCENE_DEBUG` | =file: when it appears, where the scene image is accessed |
| `BB_SCENE_NO_RESOLVE` | =file:line,...: those accesses read the native scene image |
| `BB_SCENE_SAMPLED_RESOLVE` | =0: shader reads take the native scene image as it is |
| `BB_SCENE_STENCIL_BITS` | =1 (tests): the eight-draw stencil reconstruction |
| `BB_SECTIONS` | =1: timing of the draw recording thread's sections |
| `BB_SECTIONS_SLOW` | =1 (with BB_SECTIONS): slow frames' sections at each flip |
| `BB_SEQUENCE_DUMP_COUNT` | =N (default 16): images per BB_SEQUENCE_DUMP_TRIGGER |
| `BB_SEQUENCE_DUMP_TRIGGER` | =file: creating it saves the next images shown (frame generation checks) |
| `BB_SHADER_CAPTURE` | =dir: shader inputs captured there |
| `BB_SKIP_PS` | =hash,...: draws with these pixel shaders left out |
| `BB_SKIP_TESS_DRAWS` | =1 (macOS test): draws with the game's tessellation are not drawn (missing from the picture) |
| `BB_SLOW_BIND_LOG` | =microseconds: image memory binds slower than this |
| `BB_SPEC_MATCH_CHECK` | set: the built specialization must find the same one |
| `BB_STALL_REPORT` | =0: no report when the queues stop making progress |
| `BB_STALL_THREADS` | =0: a stall report covers the command thread only |
| `BB_STREAM_STATS` | =1: what a frame streams |
| `BB_TAA_DIAGNOSTICS` | set: TAA diagnostic image dumps |
| `BB_TESS_MAX` | =N (macOS experiment): tessellation factors capped at N (less detail than the game asks for) |
| `BB_TEST_VPAD` | =1 (tests): a virtual Xbox-like gamepad |
| `BB_THD_TRACE` | =1: every The Hunter's Dream datagram and callback |
| `BB_TICK_LOG` | set: ranges marked written by the GPU most often, per 5 s |
| `BB_TILE_BACK_LOG` | set: which shader a 'Tile back' line is for |
| `BB_TIMELINE` | =file: a timeline of frames on their way to the GPU |
| `BB_TIMELINE_START` | =seconds (default 30): BB_TIMELINE starts recording this late |
| `BB_TLS_WATCH` | =offset: who writes that static TLS variable of the main thread |
| `BB_TOGGLES` | =mask: experiment bits from the start, as BB_TOGGLE_FILE without a file (the launcher's) |
| `BB_TOGGLE_FILE` | =file: experiment bits re-read every 250 ms |
| `BB_TRACE_ADDR` | =addr:size: that range traced from the start (with BB_TRACE_SIZE) |
| `BB_TRACE_CB` | =hash: how that shader's constant buffers were bound |
| `BB_TRACE_CB_LIST` | =hash,...: BB_TRACE_CB for several shaders |
| `BB_TRACE_CB_RANGE` | =lo-hi: constant buffer bindings over that range |
| `BB_TRACE_GPU_INPUTS` | =vs hash: the traced draw's buffer ranges, dumped at the display pass |
| `BB_TRACE_PIPELINE` | =vs hash: that draw's pipeline handle and push data |
| `BB_TRACE_SEMA` | set: blocking semaphore waits (deadlocks) |
| `BB_TRACE_SHADER` | =vs hash: that draw's counts and streams once a second |
| `BB_TRACE_SIZE` | =bytes: bindings over GPU-written buffers of that size, counted |
| `BB_TRACE_VERTEX_DATA` | set (with BB_TRACE_SHADER): the stream bytes whenever they change |
| `BB_TROPHY_TEST` | =id: that trophy's announcement once, nothing recorded |
| `BB_UI_DRAWS` | =file: while it exists, one frame's UI draws (another aspect ratio, or the upscaler's UI phase) logged with their targets, stencil, blend, clip rects and redirection |
| `BB_UI_DRAWS_PS` | =hex pixel shader hash: BB_UI_DRAWS also dumps that shader's draws' vertex streams, first constant buffer and indices |
| `BB_UI_TEXTURE_LOG` | =file: when it appears, the textures the UI binds over the next 2 s |
| `BB_UI_TRACE` | set: draws of frames that ended without the output-size UI |
| `BB_UI_TRIGGER_VS` | =hex hash: the vertex shader that starts the UI phase |
| `BB_UPSCALE_BEFORE_CS` | =hex hash: upscale before that compute shader |
| `BB_VERTEX_IN_PLACE` | =1: vertex data bound in place per frame, by why it has no VRAM copy, and what draws read of it (log lines) |
| `BB_VK_MARKERS` | =1: the game's debug labels as Vulkan markers |
| `BB_VK_VALIDATION` | =1: Vulkan validation layers |
| `BB_VK_VALIDATION_SYNC` | =1: Vulkan synchronization validation |
| `BB_VRAM_ACCESS_TRAP` | set: blocks moved to VRAM are closed to the CPU |
| `BB_VRAM_TRAP_RANGE` | =addr:size: BB_VRAM_ACCESS_TRAP only over that range |
| `BB_WAIT_LOG` | =file: the game's kernel waits written there |
| `BB_WAIT_TRACE` | =1: where the time waiting for GPU ticks goes |
| `BB_WARN_META_TEXTURE` | set: warn about textures read from metadata memory |
| `BB_WATCH` | =addr:size: what the translator reads from that range |
| `BB_WRITE_LOG` | =1: GPU writes to guest memory near a fault |
| `BB_WRITE_SHADOWS` | =1: ranges the game's copy shader writes in place copied into VRAM after it, small read-only bindings read there (no gain measured) |
| `BB_WRITE_SHADOW_TRACE` | =1: why BB_WRITE_SHADOWS lookups miss and who drops the copies (log lines) |
| `BB_WRITE_VERIFY` | =1: page protection to catch writes nothing announced |
| `BB_ZERO_SCAN` | =seconds: that long after the first mapping, how much direct memory is zero |

## Internal (34)

Passed between the launchers, run.py / run.sh and the game. Not meant to be set by hand.

| Variable | What it does |
|---|---|
| `BB_BUNDLED_VK_DRIVER_FILES` | the package's own Vulkan driver manifests (set by the launcher) |
| `BB_BUNDLED_VK_LAYER_PATH` | the package's own Vulkan layers, found before the host's (set by the launcher) |
| `BB_CLIENT_VERSION` | the online client's name for the server (bbport VERSION, set by run.py) |
| `BB_DATA_DIR` | the port's data folder (set by run.py) |
| `BB_DMEM_AUTO` | =1: set by tools/macos/run.sh when BB_DMEM_MB was not given: run.sh sizes direct memory again after an in-game restart |
| `BB_FSR411_OUT` | where the launcher's FSR 4.1.1 build puts its assets |
| `BB_FSR4CAP_CLEAN` | =1: the FSR 4.1.1 build removes its work folder afterwards (set by the launcher) |
| `BB_FSR4CAP_TOOLS` | the FSR 4.1.1 build's tools folder (set by the launcher) |
| `BB_FSR4CAP_WORK` | the FSR 4.1.1 build's work folder (set by the launcher) |
| `BB_GPU_DEPS` | (macOS) the native GPU process's library folder (default deps-arm64 beside bb-probe's folder) |
| `BB_GPU_PROCESS` | (macOS) the native GPU process to start (default gpu-arm64/bb-gpu in bb-probe's folder) |
| `BB_GPU_USER_DIR` | the user folder as the GPU library and the online module see it (set by the runtime) |
| `BB_IMAGE_SHAPES` | the package's image shapes list for spare images (set by run.py) |
| `BB_INTERNAL_HEIGHT` | the game's internal screen height (default 1080; set by run.py) |
| `BB_INTERNAL_WIDTH` | the game's internal screen width (default 1920; set by run.py) |
| `BB_LAUNCHER_SECTION` | the launcher page to open |
| `BB_LIVE_RES` | =1: render resolution changes live (set by run.py) |
| `BB_MODS_CONFIG` | the mods list (default <data>/mods.json) |
| `BB_MODS_DIR` | the folder of enabled mods' files (set by run.py) |
| `BB_MONITOR_SIZE` | =WxH: the monitor's size the startup patch was made for (set by the launcher) |
| `BB_OUTPUT_RES` | =WxH: the output the startup patch was made for (set by run.py) |
| `BB_PATCHES_CONFIG` | the patch choice file (default <data>/patches.json) |
| `BB_PATCHES_DIR` | the patches folder (default <data>/patches) |
| `BB_PREBUILT` | =1: packaged build, run.sh builds nothing (set by the AppImage) |
| `BB_PROBE` | path of bb-probe for run.py |
| `BB_SHADER_GAME_DIR` | the game's files the shader recipes' programs come from (set by run.py) |
| `BB_SHADER_RECIPES` | the shader recipes pack (set by run.py) |
| `BB_SKIP_NETWORK_CHOICE` | =1: no online choice at start (the launcher's) |
| `BB_STALL_THREADS_DONE` | set by the port once all threads were reported |
| `BB_STEAM_OVERLAY` | the Steam overlay's preload kept for the game (=0: off; set by the AppImage entry) |
| `BB_TIMEOUT` | =seconds: run.py stops the game after this long (tests) |
| `BB_USER_DIR` | the user folder (saves, settings; set by run.py) |
| `BB_VBLANK_HZ` | =N (default 60, 0: 480): the guest vblank rate (1000 for uncapped; set by run.py) |
| `BB_VULKAN_LIBRARY` | the Vulkan loader by path (macOS: the app's own copy, as the default search tries leaf names only) |

## Build (24)

Read by the build and packaging scripts only.

| Variable | What it does |
|---|---|
| `BB_APP` | macOS: where tools/macos/build_launcher.sh puts the app (default out/bbport.app) |
| `BB_ARCH` | =x86_64 (default) / arm64: the architecture tools/macos/setup_deps.sh builds for (arm64: the native GPU process's libraries) |
| `BB_CCACHE` | =0: the macOS build scripts without ccache |
| `BB_DEPS` | macOS: the x86-64 dependency prefix (default deps-x86_64) |
| `BB_DEPS_ARM64` | macOS: the arm64 dependency prefix the native GPU process builds with (default deps-arm64) |
| `BB_FSR4CAP_CHECKED` | internal to tools/fsr4cap: the tools were checked |
| `BB_FSR4CAP_FP8` | =1: also the FP8 FSR 4.1.1 variant through Proton's emulation; 0: never |
| `BB_FSR4CAP_ON_HOST` | set: the FSR 4.1.1 capture runs without systemd-run |
| `BB_FSR4CAP_RUNNER` | =steam: the FSR 4.1.1 capture uses Steam's runtime |
| `BB_FSR4CAP_SHELL` | internal to tools/fsr4cap: already inside the nix shell with the tools |
| `BB_FSR4CAP_SPOOF` | =0: the FSR 4.1.1 capture does not present an RDNA4 GPU to AMD's DLL |
| `BB_FSR4CAP_UMU_SHELL` | internal to tools/fsr4cap: already inside the umu nix shell |
| `BB_FSR4_DRIVER_MODELS` | FSR 4 driver models to build (default: navi2 lite) |
| `BB_FSR4_TIERS` | FSR 4 v07 asset tiers to download (tools/fetch_fsr4_assets.sh) |
| `BB_IN_NIX_SHELL` | internal to run.sh: already inside the nix shell |
| `BB_LTO` | =1: link-time optimization (packaging) |
| `BB_NET` | =OFF: no online module (shadNet) in the build (CMake option) |
| `BB_PACKAGE_FSR411` | =1: the AppImage carries FSR 4.1.1 models (never in public packages) |
| `BB_PGO` | =generate / use: profile-guided optimization (CMake option) |
| `BB_PGO_DIR` | the PGO profile folder (CMake option) |
| `BB_REBUILD` | =1: tools/macos/update.sh builds everything again |
| `BB_RELEASE` | =version: tools/macos/build_launcher.sh makes a release app (no repository path, no game icon) |
| `BB_SKIP_GPU` | set: tools/macos/build.sh builds the loader only (the GPU library already built) |
| `BB_VERSION` | the version tools/macos/package.sh packages (unless given as its first argument) |
