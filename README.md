# bbport-mac — Bloodborne on Apple Silicon Macs

A macOS port of [bbport](https://github.com/deadinside28/bloodborne_pc), deadinside28's
native port of *Bloodborne* (PS4, CUSA03173, game version 1.09). It runs the game's original
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

On the M5 Max, at 4K output with FSR 3.1 (Balanced):

- **60 FPS** in most areas.
- **Mid-50s** in busier scenes.
- **Lower in the densest scenes,** roughly 35–45.
- **A 30 FPS lock is rock solid everywhere.**

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

## Setup and build

```
git clone --recursive -b macos-port https://github.com/bmy/bbport-mac.git ~/Projects/bbport-mac/src
cd ~/Projects/bbport-mac/src
bash tools/macos/setup_deps.sh       # one-time: x86-64 dependencies and KosmicKrisp (~20-40 min)
bash tools/macos/build.sh            # the port: out/bb-probe and out/gpu/libbbgpu.dylib
bash tools/macos/build_launcher.sh   # optional: the Mac app, out/bbport.app
```

Run `build.sh` again after every `git pull`. Run `setup_deps.sh` again when it changes; it
skips what is already built.

## Playing

**From Terminal** (recommended settings: 60 FPS, FSR 3.1):

```
cd ~/Projects/bbport-mac/src
BB_FPS=60 BB_UPSCALER=fsr3 BB_GAME_DIR=~/path/to/CUSA03173 bash tools/macos/run.sh
```

**From the app** (work in progress): run `open out/bbport.app`, choose the game folder, pick
your settings and press **Play**.

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

| Variable | Values |
|---|---|
| `BB_FPS` | `30`, `60` (recommended), `uncap` |
| `BB_UPSCALER` | `fsr3` (recommended), `taa`, `metalfx`, `off`. If unset, the in-game menu's choice applies. |
| `BB_FRAME_STATS=1` | Prints frame rate and timing breakdowns to `out/last-run.log` every 5 s |
| `BB_OBJECT_MOTION=0` | Turns off character motion vectors (on by default; without them FSR smears animated characters) |
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
  and a few effects may be missing.
- **Dense scenes.** These drop below 60 FPS. The cause is CPU cost under Rosetta (see
  *Performance*), not the GPU.
- **Experimental two-thread recording** (`BB_COPIES_OFF_RECORDER=1`). It records Vulkan
  commands on two threads, but it can crash after switching to full screen and currently makes
  CPU and GPU wait on each other. It's off by default.
- **Single test machine.** Everything so far has been tested on one Mac.

## Roadmap

1. **Fix two-thread recording,** so dense scenes get closer to 60: rotating command pools per
   thread, and the full-screen crash.
2. **Finish and polish the Mac app.**
3. **Possibly run the renderer as a separate native arm64 process.** Only the game itself would
   stay under Rosetta; KosmicKrisp and Metal would run natively. This only goes ahead if a
   benchmark shows a large enough gain.

## Credits and licence

This port builds entirely on **[bbport](https://github.com/deadinside28/bloodborne_pc)** by
deadinside28, which in turn builds on **[shadPS4](https://github.com/shadps4-emu/shadPS4)**'s
renderer and shader recompiler. Vulkan on Metal is provided by **KosmicKrisp** (Mesa), as
packaged by [shadexternals](https://github.com/shadexternals/mesa-kosmickrisp). See the
[upstream README](docs/README-linux.md) for the full list of components and their licences,
the community game patches, and the Linux build.

Licensed under the **GNU GPL v2 or later** ([LICENSE](LICENSE)), like upstream.
