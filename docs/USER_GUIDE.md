# bbport for Mac: user guide

How to install the app, set it up, play, and get help when something goes wrong. For what the
project is and its current state, see the [front page](../README.md).

- [What you need](#what-you-need)
- [Install](#install)
- [First launch](#first-launch)
- [The settings, section by section](#the-settings-section-by-section)
- [Recommended settings](#recommended-settings)
- [In the game](#in-the-game)
- [Where your files are](#where-your-files-are)
- [Updating](#updating)
- [Troubleshooting](#troubleshooting)
- [Reporting a problem](#reporting-a-problem)
- [Building from source](#building-from-source)

## What you need

- **A Mac with Apple Silicon** (M1 or later) and **macOS 26 or later**. So far the port has been
  tested on one machine, an M5 Max with macOS 27; reports from other Macs are very welcome.
- **Rosetta 2.** The game's own code is Intel (x86-64) code.
- **Python 3.** The app's start-up scripts use it. The Command Line Tools include it.
- **Your own copy of Bloodborne**, dumped from your own PS4: title ID **CUSA03173**, **merged
  with the 1.09 update** (the update's files copied over the base game's), and **decrypted**. The folder you choose is the one with `eboot.bin` in it. No game
  files come with the app, and none can be provided.

To install Rosetta 2 and the Command Line Tools, open **Terminal** (Applications ▸ Utilities)
and run these two commands. Each opens an installer or asks for your password; let it finish
before the next one. If they're already installed, they say so.

```
softwareupdate --install-rosetta --agree-to-license
```

```
xcode-select --install
```

## Install

1. Download the latest `bbport-….zip` from the
   [Releases page](https://github.com/bmy/bbport-mac/releases).
2. Open the zip. You get a folder with **bbport.app**, a short read-me and the licence.
3. Drag **bbport.app** into your **Applications** folder. Opened straight from Downloads, macOS
   runs it from a read-only copy and the app will ask you to move it.
4. Open it. Because the app isn't notarised by Apple (it's a free hobby project without a paid
   developer account), macOS stops it the first time with *"Apple could not verify…"*. Click
   **Done**, then open **System Settings ▸ Privacy & Security**, scroll down to the message
   about bbport, click **Open Anyway** and confirm with your password or Touch ID.

   If macOS says instead that bbport *"is damaged and can't be opened"*, it isn't damaged: run
   this in Terminal, then open it again.

   ```
   xattr -dr com.apple.quarantine /Applications/bbport.app
   ```

## First launch

1. Under **Game**, click **Choose…** next to *Game folder* and pick the folder with `eboot.bin`.
   The line below it checks the folder: a green tick reads *Bloodborne CUSA03173, version
   01.09*. An orange warning (another title ID or version) still lets you play, but expect
   problems.
2. Leave the rest as it is for now, or see [Recommended settings](#recommended-settings).
3. Click **Play** (⌘R). The **Game Log** window opens and shows what the game prints, and the
   game's window appears.

**The first launch is slow.** The game compiles its graphics shaders the first time it needs
them, so the window can stay black for a few minutes and the game stutters whenever something
new appears. This is normal. Everything compiled is kept, and from the second launch on it
loads during a short black screen at startup.

**Stop** (⌘.) ends the game. Quitting the app also stops it.

## The settings, section by section

Settings are saved when the game starts. The upscaler, resolution, effects and similar game
settings live in `bbport.ini`, which the in-game menu also edits, so changes made in either
place show up in the other.

**Game**
- *Game folder*: the folder with `eboot.bin`.
- *Saves folder*: where saves and compiled shaders go. The default is fine; change it to keep
  saves somewhere else (for example to use saves from a source build, see
  [Where your files are](#where-your-files-are)).
- *System language*: the game's language.

**Display**
- *Output resolution*: the size of the final image. 2560×1440 is a good choice on a MacBook Pro;
  3840×2160 works too.
- *Live resolution changes*: **Off** (recommended) renders everything at full resolution, and
  changes to resolution or upscaler preset apply after *Apply and restart game* in the in-game
  menu. **On** applies them instantly, but the game's post-processing then stays at 1080p.
- *Fullscreen*, *Present mode* (FIFO is VSync; KosmicKrisp has no Mailbox), *Allow HDR*.

**Frame rate**
- *Mode*: **60** is recommended. 30 is the game's own rate; 60, 90 and *Unlocked* apply
  community frame-rate patches.
- *FPS limit*: 0 follows the display's refresh rate.

**Upscaler**
- *Upscaler*: **FSR 3.1** (recommended), TAA (cheaper, softer), MetalFX (experimental, no
  Native AA preset), or Off.
- *Preset*: **Native AA** renders at the output resolution and only anti-aliases (sharpest);
  Quality, Balanced and so on render smaller and upscale (faster).
- *Sharpening* and *Sharpness*: the upscaler's sharpening pass.
- *Object motion vectors*: keeps moving characters from smearing with a temporal upscaler.
  Leave it on.
- *Show FPS*: the frame-rate counter in the corner.

**Game effects**
- *Model detail*: **Highest** always uses the most detailed version of models, which reduces
  objects visibly switching detail as you approach them; *As in the game* is the original.
- Switches for chromatic aberration, depth of field, motion blur, SSAO, the game's own
  anti-aliasing, dynamic shadows, an optional SSR reflection effect, skipping the intro videos,
  a free camera and the debug menu.

**Performance**: leave these on their defaults unless asked to change them for testing.

**Mods**: put each mod in its own folder inside the mods folder (*Show in Finder* opens it). A
mod folder holds either `dvdroot_ps4`, or the game's folders (`chr`, `parts`, `map`, …)
directly. Turn mods on or off and change their order with the arrows; when two mods change the
same file, the lower one wins.

**Third-party patches**: shadPS4-format `.xml` patch files for version 01.09, from the patches
folder. Each patch can be switched on or off.

**Developer**: switches for diagnostics (frame statistics or a GPU profile in the log), and
*Extra variables*: space-separated `NAME=value` settings for testing, for example
`BB_FILE_TRACE=1 BB_GPU_QOS=1`. Leave it empty unless you've been asked to set something.

## Recommended settings

On a recent MacBook Pro:

| Setting | Value |
|---|---|
| Output resolution | 2560×1440 |
| Live resolution changes | Off |
| Frame rate mode | 60 |
| Present mode | FIFO (VSync) |
| Upscaler | FSR 3.1, preset Native AA (or 3840×2160 with Quality or Balanced) |
| Model detail | Highest |

Turn **Low Power Mode off** (Battery in Control Centre, or System Settings ▸ Battery) and keep
the Mac plugged in: in Low Power Mode the game runs much slower.

## In the game

**The overlay menu** opens with **F1**, **`** or **§** on the keyboard, or **L3+R3** (both stick
buttons) on a controller. It has the upscaler and preset, sharpening, output resolution, the FPS
counter and the game effects. Changes to resolution or preset that need it are applied with
**Apply and restart game**, which restarts the game in place.

**Controllers**: a DualSense or other controller works over USB or Bluetooth. With several
connected, set `BB_GAMEPAD=` followed by part of the controller's name in *Extra variables*.

**Keyboard**:

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

**Remapping**: add lines to `bbport.ini` (Help ▸ Show the Data Folder), one per PS4 input, using
SDL's names for keys and controller buttons: for example `key.cross=Space` or `pad.circle=b`.
Several bindings for one input are separated by commas. Inputs without a line keep the defaults
above.

## Where your files are

The app writes nothing inside itself. Everything it keeps is in the **data folder**:

```
~/Library/Application Support/bbport/
    bbport.ini          game settings (also edited by the in-game menu)
    user/               saves and compiled shaders (unless you chose another saves folder)
    mods/, patches/     your mods and third-party patches
    out/last-run.log    everything the last run printed; out/previous-run.log is the one before
```

**Help ▸ Show the Data Folder** opens it in Finder. The launcher's own preferences (game folder,
switches) are kept by macOS for the app.

**Coming from a source build?** Your saves are in the `user/` folder of the checkout. Either
copy that folder into the data folder, or point *Saves folder* at it. A source checkout and the
app can share one saves folder, but don't run both at once.

## Updating

**Updates** in the app's window opens the Releases page. Download the new zip and replace
`bbport.app` in Applications with the new one; your settings and saves stay where they are.
After an update, the first launch compiles shaders again.

## Troubleshooting

- **The window stays black for a long time.** On the first launch, or after an update, this is
  the shader compilation: watch the Game Log, which keeps moving. If it stops for minutes with
  no new lines, see *Reporting a problem*.
- **It runs slowly.** Turn Low Power Mode off and plug the Mac in. Close other heavy apps. Busy
  areas run below 60 FPS on some Macs; 30 or a lower output resolution helps.
- **"Rosetta 2 seems to be missing" or "Python 3 seems to be missing"** under *Before you
  play*: run the matching command from [What you need](#what-you-need).
- **"expected a decrypted PS4 SELF or ELF"** in the log: the game files are still encrypted, or
  the folder isn't the merged 1.09 one.
- **"No eboot.bin in the folder"**: choose the folder that directly contains `eboot.bin`.
- **The game crashes or freezes**: note what you were doing, then see *Reporting a problem*. If
  it happens again at the same place, try *Two-stage GPU pipeline: Off* under Performance.
- **Some effects are missing.** Metal has no geometry shaders, so a few effects that need them
  aren't drawn yet (see *Known limitations* on the front page).

## Reporting a problem

Report problems with the Mac version on this project's
[Issues page](https://github.com/bmy/bbport-mac/issues) (Help ▸ Report a Problem…), not to
shadPS4 or to the upstream bbport project. Please include:

- your Mac model, macOS version and the app's version (bbport ▸ About bbport);
- what happened and where in the game;
- the log: **Help ▸ Show the Last Run's Log**, and attach `last-run.log` (if you've started the
  game again since, `previous-run.log` is the one from before).

## Building from source

Developers and testers can build everything from a checkout instead, including the experimental
branches (such as `macos-native`, which runs the renderer as a separate native arm64 process).
See [tools/macos/README-macos.md](../tools/macos/README-macos.md). A source build's app has an
**Update** button that fetches and rebuilds the branch set in bbport ▸ Settings.
