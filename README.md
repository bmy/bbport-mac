# bbport-mac — Bloodborne on Apple Silicon Macs

A macOS port of [bbport](https://github.com/deadinside28/bloodborne_pc) (release 0.3),
deadinside28's native port of *Bloodborne* (PS4, CUSA03173, game version 1.09). It runs the game's original
executable on an Apple Silicon Mac and renders it through Vulkan on Metal, with temporal
upscaling (AMD FSR 3.1) up to 4K.

> **No game files are included.** You need your own dump of Bloodborne (CUSA03173) with the
> 1.09 update. This project is not affiliated with Sony Interactive Entertainment,
> FromSoftware, AMD, Apple, shadPS4 or the upstream bbport project. Please report problems
> with the Mac port **here**, not to shadPS4 or to upstream bbport.

**Status: experimental, playable.** The game boots, plays, saves and loads, with sound and a
DualSense or keyboard. It has been tested on one machine so far: a MacBook Pro with an M5 Max,
macOS 27.

## Performance

On the M5 Max, at 1440p output with FSR 3.1 (Native AA): **45–60 FPS**, depending on how busy
the scene is. A 30 FPS lock holds everywhere.

The limit is not the GPU: resolution and upscaling are nearly free. The limit is the CPU cost
of each Vulkan call. The game's code is x86-64, so the whole process, including the Vulkan
driver and Apple's Metal driver, runs under Rosetta 2. Every draw passes through that
translated stack, so scenes with many draws per frame run slower. See *Known limitations*
and *Roadmap*.

Make sure **Low Power Mode is off** (battery menu, or System Settings → Battery), and keep the
Mac plugged in for consistent results.

## How it works on the Mac

- **Rosetta 2.** The game code, the loader and every library they load are built as x86-64 and
  run under Rosetta 2. The build tools themselves run natively.
- **KosmicKrisp.** Vulkan is provided by Mesa's Vulkan-on-Metal driver, built from source as
  x86-64 by the setup script, with a small patch that adds an on-disk shader cache. MoltenVK
  can't run bbport, because bbport's buffer cache relies on sparse buffers.
- **Mac-specific runtime work.** On top of upstream bbport:
  - guest thread-local storage through the pthread TSD slot (macOS has no `fs`/`gs` base API);
  - a low heap for host objects the game sees, which must sit below 1 TiB;
  - Mach/BSD differences in memory, signals, timers and threads;
  - Cocoa on the main thread;
  - Core Audio output;
  - native text input.
- **Graphics extras.** Parallel shader warm-up at startup, the FPS counter, an English in-game
  menu, and FSR 3.1 and MetalFX on Apple GPUs.

## Requirements

- An Apple Silicon Mac with **macOS 26 or later** (KosmicKrisp needs macOS 26+).
- **Rosetta 2:** `softwareupdate --install-rosetta --agree-to-license`.
- **Command Line Tools:** `xcode-select --install`. Full Xcode isn't needed.
- **[Homebrew](https://brew.sh)** for Apple Silicon.
- **Your game folder:** Bloodborne CUSA03173 *with the 1.09 update merged in*, meaning the
  update's files have replaced the base game's files. The folder with `eboot.bin` in it is the
  one you point bbport at. An unmerged 1.00 `eboot.bin` won't work with the 60 FPS patches.
  The dump must be decrypted; `eboot.bin` can be either a decrypted SELF or a plain ELF.

## Setup and build

```
git clone --recursive -b macos-port https://github.com/bmy/bbport-mac.git ~/Projects/bbport-mac/src
cd ~/Projects/bbport-mac/src
bash tools/macos/setup_deps.sh       # one-time: x86-64 dependencies and KosmicKrisp (~20-40 min)
bash tools/macos/build.sh            # the port: out/bb-probe and out/gpu/libbbgpu.dylib
bash tools/macos/build_launcher.sh   # the Mac app: out/bbport.app
open out/bbport.app
```

After that, the app's **Update** button keeps everything current: it fetches the latest
version from GitHub and rebuilds only what changed. From Terminal, the same is
`bash tools/macos/update.sh`.

## Playing

**From the app** (recommended): open `out/bbport.app` (drag it to the Dock to keep it handy),
choose the game folder, and press **Play**. Recommended settings:

- **Frame rate:** mode 60, no FPS limit, present mode FIFO.
- **Upscaler:** FSR 3.1, with either 1440p output at Native AA (sharpest) or 4K output at
  Balanced or Quality.
- **Live resolution changes:** off. Everything then renders at full resolution, and resolution
  or preset changes apply after **Apply and restart game** in the in-game menu. On, they apply
  instantly, but the game's post-processing stays at 1080p.

The app's log window shows the game's output, and every run is also saved to
`out/last-run.log` (the one before it to `out/previous-run.log`).

**From Terminal:**

```
cd ~/Projects/bbport-mac/src
BB_FPS=60 BB_UPSCALER=fsr3 BB_GAME_DIR=~/path/to/CUSA03173 bash tools/macos/run.sh
```

**First launch:** the game compiles its shaders, so the first session after a build stutters
whenever something new appears. From the second launch on, everything seen before is compiled
in parallel during a short black screen at startup.

**In-game menu:** press **F1**, **`** or **§**, or **L3+R3** on a controller. From the menu you
can:

- choose the upscaler and preset;
- set sharpening;
- set the output resolution, up to 4K;
- turn the FPS counter on or off;
- turn game effects on or off.

Settings are saved to `bbport.ini`.

**Remapping controls:** add lines to `bbport.ini`, one per PS4 input, with SDL names for keys and
gamepad buttons, for example `key.cross=Space` or `pad.circle=b` (several bindings separated by
commas). Inputs without a line keep the defaults below. To pick one of several controllers, set
`BB_GAMEPAD` to part of its name or its GUID.

| Keyboard | PS4 |
|---|---|
| W A S D | left stick |
| arrow keys | right stick |
| I J K L | D-pad |
| Space / Left Shift / E / Q | Cross / Circle / Square / Triangle |
| Enter | Options |
| 1 / 3 | L1 / R1 |
| R / F | L2 / R2 |
| Z / C | L3 / R3 |
| Tab / Backspace | touchpad |

More detail, including troubleshooting: [tools/macos/README-macos.md](tools/macos/README-macos.md).

### Useful settings

For Terminal runs; in the app, most are regular settings and the rest go in
**Developer → Extra variables**.

| Variable | Values |
|---|---|
| `BB_FPS` | `30`, `60` (recommended), `uncap` |
| `BB_UPSCALER` | `fsr3` (recommended), `taa`, `metalfx`, `off`. If unset, the in-game menu's choice applies. |
| `BB_FRAME_STATS=1` | Prints frame rate and timing breakdowns to `out/last-run.log` every 5 s |
| `BB_OBJECT_MOTION=0` | Turns off character motion vectors (on by default; without them FSR smears animated characters) |
| `BB_VK_RECORD_THREADS` | Threads recording Vulkan commands (default: 3 on this class of Mac; `1` records on one thread) |
| `BB_GAMEPAD` | Which controller to use: part of its name or its GUID |
| `BB_CAMERA_Y` | `up` or `down` forces which way the camera motion vectors treat vertical. By default it follows each frame's G-buffer viewport (the wrong one makes FSR shimmer on floors) |
| `BB_BREADCRUMBS=1` | Turns on bbport 0.3's GPU crash breadcrumbs (off on the Mac: with KosmicKrisp they blank the picture) |
| `BB_PRELOAD_THREADS` | Threads for the startup shader warm-up (default: all cores) |

## Upscalers on the Mac

- **FSR 3.1** works and is recommended. Use Native AA for anti-aliasing only, or
  Quality/Balanced to render below the output resolution and upscale.
- **TAA** works: cheaper than FSR 3.1, but softer.
- **MetalFX** (Apple) is experimental. The current version waits for the GPU every frame, so it
  costs frame rate, and it doesn't support the Native AA preset.
- **FSR 4 / 4.1.1** need GPU features that KosmicKrisp doesn't expose. They're hidden on the Mac.
- **DLSS** requires NVIDIA hardware and isn't possible on a Mac.

## Known limitations

- **Geometry shaders.** Metal has no geometry shader stage, so draws that need one are skipped
  and a few effects may be missing. The newer KosmicKrisp that emulates them (the version
  upstream shadPS4 moved to in October 2026) ran clearly slower in testing, even with them
  switched off, and drew blood on characters wrongly. The port stays on the previous version
  for now.
- **Depth of field (bbport 0.3).** With DOF on, the top of the frame shows an upside-down copy of
  the scene and horizontal streaks. It happens on Linux too and the upstream developer knows;
  until it's fixed, turn **Depth of field** off in the app's Game effects.
- **Dense scenes.** These drop below 60 FPS. The cause is CPU cost under Rosetta (see
  *Performance*), not the GPU.
- **Single test machine.** Everything so far has been tested on one Mac.

## Roadmap

1. **Tune multi-thread recording on the Mac,** so dense scenes get closer to 60. bbport 0.3
   records Vulkan commands on several threads and submits from them; how many threads suit
   KosmicKrisp under Rosetta is still to be measured.
2. **Geometry shaders:** find what makes the newer KosmicKrisp slower, report it upstream, and
   move to it once it's as fast.
3. **Move the renderer into a separate native arm64 process.** The renderer is the shadPS4-based
   GPU library plus KosmicKrisp and Metal. It would share guest memory with the game process
   through the existing shared-memory pool. This removes Rosetta from the frame-rate bottleneck,
   and it is also the first step towards a Rosetta-free port (below). A benchmark of native
   versus Rosetta KosmicKrisp decides when it's worth doing.

### Planning for the end of Rosetta

Apple has said macOS 27 is the last release with Rosetta for general use. From macOS 28 it
remains only for certain older, unmaintained games that rely on Intel-based frameworks, and
it's unclear whether bbport would qualify. Apple's statements on this differ in detail. The
game's code is x86-64, so a long-term Mac port needs its own translation:

- **Renderer.** After roadmap item 3, the renderer no longer needs Rosetta. Only the game code
  and bbport's small runtime remain x86-64.
- **Game code.** bbport targets a single, fixed executable (Bloodborne 1.09), which it already
  analyses and relinks offline. It also already bundles the Zydis x86 disassembler. That makes
  ahead-of-time recompilation of that one binary to arm64 feasible, with a small runtime
  fallback for indirect jumps that can't be resolved in advance. The main difficulty is memory
  ordering: x86 has stronger guarantees than arm64, and the hardware mode Rosetta uses for this
  isn't available to ordinary apps, so translated code needs barriers. This is a research-sized
  project.
- **Fallbacks.** An existing x86-to-arm64 translator, if one gains macOS support, or running the
  Linux build in an arm64 Linux virtual machine with such a translator. Both would likely be
  slower than today.

## Credits and licence

This port builds entirely on **[bbport](https://github.com/deadinside28/bloodborne_pc)** by
deadinside28, which in turn builds on **[shadPS4](https://github.com/shadps4-emu/shadPS4)**'s
renderer and shader recompiler. Vulkan on Metal is provided by **KosmicKrisp** (Mesa), as
packaged by [shadexternals](https://github.com/shadexternals/mesa-kosmickrisp). See the
[upstream README](docs/README-linux.md) for the full list of components and their licences,
the community game patches, and the Linux build.

Licensed under the **GNU GPL v2 or later** ([LICENSE](LICENSE)), like upstream.
