# bbport for Mac: Bloodborne on Apple Silicon

A macOS port of [bbport](https://github.com/deadinside28/bloodborne_pc), deadinside28's native
port of *Bloodborne* (PS4, CUSA03173, game version 1.09). It runs the game's original
executable on an Apple Silicon Mac and renders it through Vulkan on Metal, with FSR 3.1
upscaling up to 4K.

**[Download the latest release](https://github.com/bmy/bbport-mac/releases)** ·
**[User guide](docs/USER_GUIDE.md)** ·
**[Report a problem](https://github.com/bmy/bbport-mac/issues)**

> **No game files are included.** You need your own decrypted copy of Bloodborne (CUSA03173)
> with the 1.09 update merged in. This project is not affiliated with Sony Interactive
> Entertainment, FromSoftware, AMD, Apple, shadPS4 or the upstream bbport project. Please report
> problems with the Mac port **here**, not to shadPS4 or to upstream bbport.

## Status (October 2026)

**Experimental, playable.** Based on bbport 0.4.

- The whole game boots, plays, saves and loads, with sound, cutscenes, a DualSense or other
  controller, or the keyboard.
- **About 45–60 FPS** at 1440p with FSR 3.1 (Native AA) on an M5 Max; a 30 FPS lock holds
  everywhere. Busy areas are the slow ones.
- 60 FPS (community patch), output up to 4K, FSR 3.1, TAA and experimental MetalFX, the in-game
  settings menu, mods and shadPS4-format patches.
- Tested on **one machine** so far (MacBook Pro, M5 Max, macOS 27). Reports from other Macs are
  very welcome.

Recent work: a self-contained app download; the loading screens' item picture, which vanished
for most of each loading screen on every platform; keyboard input; a Retina window; and an
experimental **native arm64 renderer process** (below).

## Play

You need an Apple Silicon Mac with **macOS 26 or later**, **Rosetta 2**, and the **Command Line
Tools** (for Python 3). Then:

1. Download `bbport-….zip` from [Releases](https://github.com/bmy/bbport-mac/releases), open
   it, and drag **bbport.app** into Applications.
2. Open it (the first time: **System Settings ▸ Privacy & Security ▸ Open Anyway**, since the
   app isn't notarised).
3. Choose your game folder (the one with `eboot.bin`) and press **Play**.

The **[user guide](docs/USER_GUIDE.md)** covers each step, the commands to install Rosetta 2
and the Command Line Tools, every setting, the controls, where your saves are, and what to do
when something goes wrong. In the app it's under **Help ▸ bbport User Guide**.

Recommended: 2560×1440 output, frame rate 60, FSR 3.1 at Native AA, Model detail Highest. Keep
**Low Power Mode off**.

## How it works on the Mac

- **Rosetta 2.** The game's code is x86-64, so the game, the loader and every library they load
  run as x86-64 under Rosetta 2.
- **KosmicKrisp.** Vulkan runs on Mesa's Vulkan-on-Metal driver, built from source as x86-64,
  with small patches (an on-disk shader cache among them). MoltenVK can't run bbport, because
  bbport's buffer cache relies on sparse buffers.
- **Mac-specific runtime work** on top of upstream bbport:
  - guest thread-local storage through the pthread TSD slot (macOS has no `fs`/`gs` base API);
  - a low heap for host objects the game sees, which must sit below 1 TiB;
  - Mach/BSD differences in memory, signals, timers and threads;
  - Cocoa on the main thread, Core Audio output, native text input.
- **Graphics extras.** Parallel shader warm-up at startup, the FPS counter, an English in-game
  menu, and FSR 3.1 and MetalFX on Apple GPUs.
- **The app** (`launcher-macos/`, SwiftUI) is native arm64. The release zip carries the engine,
  its libraries and the driver inside the app; nothing else needs installing.

## Known limitations

- **Geometry shaders.** Metal has no geometry shader stage, so draws that need one are skipped
  and a few effects are missing. A newer KosmicKrisp emulates them, but it ran clearly slower in
  testing and drew blood on characters wrongly, so the port stays on the previous version.
- **Busy scenes run below 60 FPS.** The limit is the CPU cost of each draw going through
  bbport's renderer, KosmicKrisp and Metal, all translated by Rosetta. Resolution and upscaling
  are nearly free.
- **Occasional pop-in** of objects while an area streams in.
- **Shader compilation** on the first launch (and after updates): a few minutes of black screen
  and some stutter; later launches are quick.

## Roadmap

1. **Native renderer process** (branch `macos-native`, experimental, source builds only). The
   renderer (bbport's GPU library, KosmicKrisp and Metal's driver) runs in a separate native
   arm64 process, `bb-gpu`, sharing the game's memory with the x86-64 game process. It works,
   including keyboard, controllers, the overlay and statistics. Making it faster than the
   in-process path in busy scenes is the current work; then it becomes the default. Design and
   status: `docs/macos-native-gpu.md` on that branch.
2. **Geometry shaders**: find what makes the newer KosmicKrisp slower, report it upstream, and
   move to it once it's as fast.
3. **Pop-in**: measure whether it comes from streaming or from draw distance, and fix what can be
   fixed.
4. **More Macs**: results from M1–M4 machines and smaller GPUs.

### Planning for the end of Rosetta

Apple has said macOS 27 is the last release with Rosetta for general use; from macOS 28 it
remains only for certain older games, and it's unclear whether bbport would qualify. Since the
game's code is x86-64, a long-term Mac port needs its own translation:

- **Renderer.** With roadmap item 1, the renderer no longer needs Rosetta. Only the game code and
  bbport's small runtime remain x86-64.
- **Game code.** bbport targets one fixed executable (Bloodborne 1.09), which it already analyses
  and relinks offline, and it bundles the Zydis x86 disassembler. Ahead-of-time recompilation of
  that one binary to arm64 is feasible, with a runtime fallback for indirect jumps. The hard part
  is memory ordering: x86 guarantees more than arm64, and the hardware mode Rosetta uses for this
  isn't available to ordinary apps. This is a research-sized project.
- **Fallbacks.** An existing x86-to-arm64 translator, if one gains macOS support, or the Linux
  build in an arm64 Linux virtual machine with such a translator. Both would likely be slower.

## Building from source

For testing branches and development: [tools/macos/README-macos.md](tools/macos/README-macos.md).
In short:

```
git clone --recursive -b macos-0.4 https://github.com/bmy/bbport-mac.git ~/Projects/bbport-mac/src
cd ~/Projects/bbport-mac/src
bash tools/macos/setup_deps.sh
bash tools/macos/build.sh
bash tools/macos/build_launcher.sh
open out/bbport.app
```

A source build's app has an **Update** button that fetches and rebuilds the branch set in
bbport ▸ Settings. Releases are built by GitHub Actions from a tag
(`.github/workflows/release.yml`, `tools/macos/package.sh`).

## Credits and licence

This port builds entirely on **[bbport](https://github.com/deadinside28/bloodborne_pc)** by
deadinside28, which in turn builds on **[shadPS4](https://github.com/shadps4-emu/shadPS4)**'s
renderer and shader recompiler. Vulkan on Metal is provided by **KosmicKrisp** (Mesa), as
packaged by [shadexternals](https://github.com/shadexternals/mesa-kosmickrisp). See the
[upstream README](docs/README-linux.md) for the full list of components and their licences,
the community game patches, and the Linux build.

Licensed under the **GNU GPL v2 or later** ([LICENSE](LICENSE)), like upstream.
