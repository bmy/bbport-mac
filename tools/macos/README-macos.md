# bbport on macOS (Apple Silicon)

The game code runs as x86-64 under Rosetta 2. Vulkan runs on Metal through KosmicKrisp.
MoltenVK can't run bbport, because it has no sparse buffers. You need the game folder: Bloodborne CUSA03173, merged with
update 1.09, the one with `eboot.bin` in it.

## One-time setup

1. Install the Command Line Tools (`xcode-select --install`), Rosetta 2
   (`softwareupdate --install-rosetta --agree-to-license`) and Apple Silicon Homebrew
   (https://brew.sh).
2. Build the x86-64 dependencies into `deps-x86_64/`. This takes a while, and KosmicKrisp
   alone takes 10 to 20 minutes:

   ```
   bash tools/macos/setup_deps.sh
   ```

   Run it again after it changes. It skips whatever is already built.

## Building

```
bash tools/macos/build.sh            # the port: out/bb-probe and out/gpu/libbbgpu.dylib
bash tools/macos/build_launcher.sh   # the launcher app: out/bbport.app
```

To update later, press **Update** in the app, or run `bash tools/macos/update.sh`: it fetches
the latest version from GitHub and rebuilds only what changed (dependencies, the port, the app).
`bash tools/macos/update.sh <branch>` switches to another branch; in the app, set the branch in
bbport > Settings. If the app itself was rebuilt, quit and reopen it. The app remembers the checkout it was built from, and you can change
it in bbport > Settings (⌘,). If you have already picked the game folder in the app,
`build_launcher.sh` also uses the game's icon for the app.

## Playing

**From the app:** `open out/bbport.app`, or drag it to /Applications. Choose the game folder,
adjust the settings and press **Play** (⌘R). The Game Log window streams the output live. **Stop**
(⌘.) ends the game. Settings are saved when the game starts:

- The launcher's own settings go to the app's preferences
  (`defaults read io.github.bbport.mac`).
- Game settings go to `bbport.ini` in the checkout. The in-game menu reads and writes the same
  file, and other keys in it are kept.

**From Terminal:**

```
BB_GAME_DIR=~/Games/shadPS4/CUSA03173-109 bash tools/macos/run.sh
```

`run.sh` takes the same environment variables the app sets, for example `BB_FPS=30|60|90|uncap`,
`BB_UPSCALER=off|fsr3|taa|metalfx`, `BB_FULLSCREEN=1`.

### First launch: the shader warm-up

The first launch after a build, a driver update or a shader-cache reset compiles the game's
shaders. The window can stay black for a few minutes while that happens, which is normal. Later
launches load the cached pipelines at startup and are quick. The pipeline cache lives in the
saves folder (`user/` by default).

## Controls

A DualSense works over USB or Bluetooth. Keyboard layout:

| Keyboard | PS4 |
|---|---|
| W A S D | left stick |
| arrow keys | right stick |
| I J K L | D-pad |
| Space | Cross |
| Left Shift | Circle |
| E | Square |
| Q | Triangle |
| Enter | Options |
| 1 / 3 | L1 / R1 |
| R / F | L2 / R2 |
| Z / C | L3 / R3 |
| Tab / Backspace | touchpad |

The overlay menu (upscaler, FPS counter, effects) opens with **F1**, **`` ` ``** or **§**, or
**L3+R3** on a controller.

## Troubleshooting

- Every run's full output is saved to **`out/last-run.log`**, and the run before it to
  `out/previous-run.log`. Send it when you report a problem. The Game Log window has a button
  that opens it.
- "expected a decrypted PS4 SELF or ELF": the game dump is encrypted or the folder isn't the
  merged 1.09 one. The message shows the file and its first bytes.
- The app says "not built yet": run `bash tools/macos/build.sh`.
- `swift build` fails in `build_launcher.sh`: send the whole terminal output.
- The build fails: `out/gpu-errors.txt` and `out/loader-build.log` hold the errors.
- If it crashes after switching to full screen: make sure `BB_COPIES_OFF_RECORDER` is not set
  to 1, since experimental two-thread recording has a known full-screen crash.
- If it runs slowly: turn Low Power Mode off and plug the Mac in.
- The black screen lasts much longer than usual: check whether the log is still moving. The
  pipeline preload reports its count and time there when it finishes.
