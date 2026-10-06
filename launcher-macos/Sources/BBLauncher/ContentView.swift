// SPDX-License-Identifier: GPL-2.0-or-later
// The main window: the Linux launcher's settings page as a grouped form, with Play below.
import AppKit
import Combine
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var model: LauncherModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(spacing: 0) {
            Form {
                if !model.repoCheck.allowsPlay {
                    RepositoryProblemSection()
                }
                GameSection()
                DisplaySection()
                FrameRateSection()
                UpscalerSection()
                EffectsSection()
                Group {
                    PerformanceSection()
                    ModsSection()
                    PatchesSection()
                    DeveloperSection()
                }
            }
            .formStyle(.grouped)
            .toggleStyle(.switch)

            Divider()
            PlayBar(showLog: { openWindow(id: "log") })
        }
        .frame(minWidth: 600, minHeight: 560)
        .navigationTitle("Bloodborne")
        .navigationSubtitle(model.statusText)
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            model.refreshChecks()
        }
    }
}

// MARK: - Play bar

struct PlayBar: View {
    @EnvironmentObject var model: LauncherModel
    let showLog: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Label {
                    Text(model.statusText)
                        .lineLimit(2)
                } icon: {
                    statusIcon
                }
                .font(.headline)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            Button {
                model.update()
                if model.isRunning { showLog() }
            } label: {
                Label("Update", systemImage: "arrow.down.circle")
            }
            .controlSize(.large)
            .disabled(!model.canUpdate)
            .help("Get the latest version from GitHub and rebuild what changed")
            Button(action: showLog) {
                Label("Log", systemImage: "text.alignleft")
            }
            .controlSize(.large)
            if model.isRunning {
                Button(role: .destructive) {
                    model.stop()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(minWidth: 70)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .controlSize(.large)
                .keyboardShortcut(".", modifiers: .command)
                .disabled(model.state == .stopping)
            } else {
                Button {
                    model.play()
                    if model.isRunning { showLog() }
                } label: {
                    Label("Play", systemImage: "play.fill")
                        .frame(minWidth: 70)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut("r", modifiers: .command)
                .disabled(!model.canPlay)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(.bar)
    }

    private var detail: String {
        switch model.state {
        case .running where model.job == .update, .stopping where model.job == .update:
            return "Fetching from GitHub and rebuilding. A new graphics driver takes 15–25 minutes; "
                + "otherwise a minute or two."
        case let .exited(status, bySignal) where model.job == .update:
            if status == 0 && !bySignal {
                return "Up to date. If the log says the Mac app was rebuilt, quit and reopen it."
            }
            return model.lastProblem ?? "Details are in the log."
        case .running, .stopping:
            return "The first launch after a build or a shader-cache reset compiles shaders: "
                + "a black window for a few minutes is expected."
        case let .exited(status, bySignal) where status != 0 && !bySignal:
            return model.lastProblem ?? "Details are in the log and in out/last-run.log."
        default:
            return "Settings are saved when the game starts. In game, the overlay menu is on F1, ` or § (L3+R3 on a controller)."
        }
    }

    @ViewBuilder private var statusIcon: some View {
        switch model.state {
        case .running:
            Image(systemName: "gamecontroller.fill").foregroundStyle(.green)
        case .stopping:
            Image(systemName: "hourglass").foregroundStyle(.orange)
        case let .exited(status, bySignal):
            if status == 0 || bySignal {
                Image(systemName: "checkmark.circle").foregroundStyle(.secondary)
            } else {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
            }
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
        case .idle:
            if model.canPlay {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            } else {
                Image(systemName: "exclamationmark.circle").foregroundStyle(.orange)
            }
        }
    }
}

// MARK: - Shared rows

/// A status line with a coloured symbol, for the folder checks.
struct CheckLabel: View {
    let check: Check

    var body: some View {
        Label {
            Text(check.message)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        } icon: {
            switch check.level {
            case .ok:
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            case .warning:
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            case .error:
                Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
            }
        }
    }
}

/// A folder row: title, path, Choose…, Show in Finder and an optional reset.
struct FolderRow: View {
    let title: String
    let path: String
    var placeholder = "Not chosen"
    let choose: () -> Void
    let reveal: () -> Void
    /// A custom folder that can go back to the default.
    var canReset = false
    var reset: () -> Void = {}

    var body: some View {
        LabeledContent {
            HStack(spacing: 6) {
                Button("Choose…", action: choose)
                Button(action: reveal) {
                    Image(systemName: "folder")
                }
                .help("Show in Finder")
                .disabled(path.isEmpty)
                if canReset {
                    Button(action: reset) {
                        Image(systemName: "arrow.uturn.backward")
                    }
                    .help("Back to the default folder")
                }
            }
        } label: {
            Text(title)
            Text(path.isEmpty ? placeholder : path)
                .truncationMode(.middle)
                .textSelection(.enabled)
        }
    }
}

// MARK: - Sections

struct RepositoryProblemSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            CheckLabel(check: model.repoCheck)
            HStack {
                SettingsLink {
                    Text("Set the Repository…")
                }
                Button("Check Again") { model.refreshChecks() }
            }
        } header: {
            Text("bbport repository")
        }
    }
}

struct GameSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            FolderRow(title: "Game folder (CUSA03173)", path: model.prefs.gameFolder,
                      choose: { model.chooseGameFolder() },
                      reveal: { model.reveal(URL(fileURLWithPath: model.prefs.gameFolder, isDirectory: true)) })
            CheckLabel(check: model.gameCheck)
            FolderRow(title: "Saves folder",
                      path: model.prefs.userFolder.isEmpty ? "Default: \(model.userFolderURL.path)" : model.prefs.userFolder,
                      choose: { model.chooseUserFolder() },
                      reveal: { model.reveal(model.userFolderURL) },
                      canReset: !model.prefs.userFolder.isEmpty,
                      reset: { model.prefs.userFolder = "" })
            Picker("System language", selection: $model.prefs.language) {
                ForEach(Catalog.languages) { Text($0.label).tag($0.value) }
            }
        } header: {
            Text("Game")
        } footer: {
            Text("The folder with eboot.bin: the base game merged with update 1.09. Saves and the pipeline cache live in the saves folder.")
                .foregroundStyle(.secondary)
        }
    }
}

struct DisplaySection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            Picker(selection: $model.game.outputRes) {
                ForEach(Catalog.outputResolutions) { Text($0.label).tag($0.value) }
            } label: {
                Text("Output resolution")
                Text("The upscaler fills the frame at this size")
            }
            Picker(selection: $model.game.liveResolution) {
                ForEach(Catalog.liveResolution) { Text($0.label).tag($0.value) }
            } label: {
                Text("Live resolution changes")
                Text("No restart needed, but slower on weaker GPUs")
            }
            Toggle("Fullscreen", isOn: $model.prefs.fullscreen)
            Picker(selection: $model.prefs.presentMode) {
                ForEach(Catalog.presentModes) { Text($0.label).tag($0.value) }
            } label: {
                Text("Present mode")
                Text("KosmicKrisp has no Mailbox and uses FIFO instead")
            }
            Toggle("Allow HDR", isOn: $model.prefs.hdr)
        } header: {
            Text("Display")
        }
    }
}

struct FrameRateSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            Picker(selection: $model.prefs.fpsMode) {
                ForEach(Catalog.fpsModes) { Text($0.label).tag($0.value) }
            } label: {
                Text("Mode")
                Text("Which frame rate patch to apply to the game (BB_FPS)")
            }
            LabeledContent {
                HStack(spacing: 4) {
                    TextField("FPS limit", value: $model.prefs.fpsLimit, format: .number)
                        .labelsHidden()
                        .multilineTextAlignment(.trailing)
                        .frame(width: 56)
                    Stepper("FPS limit", value: $model.prefs.fpsLimit, in: 0...480, step: 5)
                        .labelsHidden()
                }
            } label: {
                Text("FPS limit")
                Text("0: the display refresh rate (up to 120 Hz); set a number for another limit")
            }
        } header: {
            Text("Frame rate")
        } footer: {
            Text("30 is the setting validated on macOS so far; 60, 90 and unlocked apply the community frame-rate patches.")
                .foregroundStyle(.secondary)
        }
    }
}

struct UpscalerSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        let upscaler = model.game.upscaler
        Section {
            Picker(selection: $model.game.upscaler) {
                ForEach(Catalog.upscalers) { Text($0.label).tag($0.value) }
            } label: {
                Text("Upscaler")
                Text(hint(for: upscaler))
            }
            Picker("Preset", selection: $model.game.preset) {
                ForEach(Catalog.presets) { Text($0.label).tag($0.value) }
            }
            .disabled(upscaler == "off" || upscaler == "taa")
            Toggle("Sharpening (RCAS)", isOn: $model.game.sharpen)
                .disabled(upscaler == "off")
            LabeledContent("Sharpness") {
                HStack {
                    Slider(value: $model.game.sharpness, in: 0...2, step: 0.05)
                        .frame(minWidth: 140)
                    Text(String(format: "%.2f", model.game.sharpness))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            .disabled(upscaler == "off" || !model.game.sharpen)
            Toggle(isOn: $model.game.objectMotion) {
                Text("Object motion vectors")
                Text("Less ghosting on characters; costs about 10% FPS")
            }
            Toggle("Show FPS", isOn: $model.game.showFPS)
        } header: {
            Text("Upscaler")
        } footer: {
            VStack(alignment: .leading, spacing: 4) {
                if let replaced = model.game.replacedUpscaler {
                    Text("bbport.ini asked for \(replaced), which macOS cannot run; FSR 3.1 is selected instead.")
                }
                Text("Stored in bbport.ini and passed as BB_UPSCALER; in game, change it in the overlay menu.")
            }
            .foregroundStyle(.secondary)
        }
    }

    private func hint(for upscaler: String) -> String {
        switch upscaler {
        case "fsr3": return "AMD FSR 3.1 temporal upscaling"
        case "taa": return "Anti-aliasing at the output resolution, no FSR model"
        case "metalfx": return "Apple MetalFX temporal upscaling. Being added on a separate branch: a build without it runs with the upscaler off."
        default: return "No upscaling or temporal anti-aliasing"
        }
    }
}

struct EffectsSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            Picker("Model detail", selection: $model.game.modelLOD) {
                ForEach(Catalog.modelDetail) { Text($0.label).tag($0.value) }
            }
            ForEach($model.game.effects) { $effect in
                Toggle(isOn: $effect.isOn) {
                    Text(effect.title)
                    if let note = effect.note {
                        Text(note)
                    }
                }
            }
        } header: {
            Text("Game effects")
        } footer: {
            Text("Game patches, applied at start.")
                .foregroundStyle(.secondary)
        }
    }
}

struct PerformanceSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            Picker(selection: $model.prefs.drawPipe) {
                ForEach(Catalog.drawPipe) { Text($0.label).tag($0.value) }
            } label: {
                Text("Two-stage GPU pipeline")
                Text("20–30% faster; turn off if unstable")
            }
            Picker(selection: $model.prefs.preupload) {
                ForEach(Catalog.preupload) { Text($0.label).tag($0.value) }
            } label: {
                Text("Background upload of game data")
                Text("Full can mean fewer hitches when areas load")
            }
            Picker("GPU data readbacks by the CPU", selection: $model.prefs.readbacks) {
                ForEach(Catalog.readbacks) { Text($0.label).tag($0.value) }
            }
        } header: {
            Text("Performance")
        }
    }
}

struct ModsSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            Toggle("Load mods", isOn: $model.prefs.modsEnabled)
            FolderRow(title: "Mods folder", path: model.modsFolderURL.path,
                      choose: { model.chooseModsFolder() },
                      reveal: { model.reveal(model.modsFolderURL) },
                      canReset: !model.prefs.modsFolder.isEmpty,
                      reset: { model.prefs.modsFolder = "" })
            if model.mods.isEmpty {
                Text("No mods")
                    .foregroundStyle(.secondary)
            }
            ForEach($model.mods) { $mod in
                HStack {
                    Text(mod.name)
                    Spacer()
                    Button {
                        model.moveMod(mod.name, by: -1)
                    } label: {
                        Image(systemName: "chevron.up")
                    }
                    .buttonStyle(.borderless)
                    .help("Load earlier")
                    Button {
                        model.moveMod(mod.name, by: 1)
                    } label: {
                        Image(systemName: "chevron.down")
                    }
                    .buttonStyle(.borderless)
                    .help("Load later")
                    Toggle(mod.name, isOn: $mod.isOn)
                        .labelsHidden()
                }
            }
            .disabled(!model.prefs.modsEnabled)
            Button("Refresh the List") {
                model.saveModProfile()
                model.refreshMods()
            }
        } header: {
            Text("Mods")
        } footer: {
            Text("Extract each mod into its own folder (with dvdroot_ps4, or chr/, parts/, … directly). When files collide, the mod lower in the list wins. Applied at start.")
                .foregroundStyle(.secondary)
        }
    }
}

struct PatchesSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            FolderRow(title: "Patches folder", path: model.patchesFolderURL.path,
                      choose: { model.choosePatchesFolder() },
                      reveal: { model.reveal(model.patchesFolderURL) },
                      canReset: !model.prefs.patchesFolder.isEmpty,
                      reset: { model.prefs.patchesFolder = "" })
            if model.patches.isEmpty {
                Text("No patches")
                    .foregroundStyle(.secondary)
            }
            ForEach($model.patches) { $patch in
                Toggle(isOn: $patch.isOn) {
                    Text(patch.title)
                    Text(patch.subtitle)
                }
                .help(patch.note ?? "")
            }
            Button("Refresh the List") {
                model.savePatchProfile()
                model.refreshPatches()
            }
        } header: {
            Text("Third-party patches")
        } footer: {
            Text("shadPS4-format XML patches for version 01.09 from the patches folder. Applied at start.")
                .foregroundStyle(.secondary)
        }
    }
}

struct DeveloperSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            Toggle(isOn: $model.prefs.frameStats) {
                Text("Frame statistics in the log")
                Text("BB_FRAME_STATS")
            }
            Toggle(isOn: $model.prefs.gpuProfile) {
                Text("GPU profile in the log")
                Text("BB_GPU_PROFILE")
            }
            Toggle(isOn: $model.prefs.vkValidation) {
                Text("Vulkan validation layers")
                Text("Much slower; needs the validation layers installed")
            }
            TextField(text: $model.prefs.extraEnv, prompt: Text("NAME=value NAME2=value")) {
                Text("Extra variables")
            }
        } header: {
            Text("Developer")
        } footer: {
            Text("Extra variables are space-separated and override the settings above.")
                .foregroundStyle(.secondary)
        }
    }
}
