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
        .navigationTitle("血源诅咒")
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
                Label("更新", systemImage: "arrow.down.circle")
            }
            .controlSize(.large)
            .disabled(!model.canUpdate)
            .help("从 GitHub 获取最新版本并重新构建变更部分")
            Button(action: showLog) {
                Label("日志", systemImage: "text.alignleft")
            }
            .controlSize(.large)
            if model.isRunning {
                Button(role: .destructive) {
                    model.stop()
                } label: {
                    Label("停止", systemImage: "stop.fill")
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
                    Label("启动游戏", systemImage: "play.fill")
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
            return "正在从 GitHub 获取更新并重新构建。新图形驱动约需 15–25 分钟；"
                + "其他更新约需一两分钟。"
        case let .exited(status, bySignal) where model.job == .update:
            if status == 0 && !bySignal {
                return "已更新。如果日志提示启动器已重新构建，请退出并重新打开。"
            }
            return model.lastProblem ?? "详情请查看日志。"
        case .running, .stopping:
            return "构建或清空着色器缓存后的首次启动需要编译着色器："
                + "可能会黑屏几分钟，请耐心等待。"
        case let .exited(status, bySignal) where status != 0 && !bySignal:
            return model.lastProblem ?? "详情请查看日志或 out/last-run.log。"
        default:
            return "启动游戏时保存设置。游戏内按 F1、` 或 § 打开设置面板（手柄按 L3+R3）。"
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
    var placeholder = "未选择"
    let choose: () -> Void
    let reveal: () -> Void
    /// A custom folder that can go back to the default.
    var canReset = false
    var reset: () -> Void = {}

    var body: some View {
        LabeledContent {
            HStack(spacing: 6) {
                Button("选择…", action: choose)
                Button(action: reveal) {
                    Image(systemName: "folder")
                }
                .help("在访达中显示")
                .disabled(path.isEmpty)
                if canReset {
                    Button(action: reset) {
                        Image(systemName: "arrow.uturn.backward")
                    }
                    .help("恢复默认目录")
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
                    Text("设置项目目录…")
                }
                Button("重新检查") { model.refreshChecks() }
            }
        } header: {
            Text("bbport 项目目录")
        }
    }
}

struct GameSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            FolderRow(title: "游戏目录（CUSA03173）", path: model.prefs.gameFolder,
                      choose: { model.chooseGameFolder() },
                      reveal: { model.reveal(URL(fileURLWithPath: model.prefs.gameFolder, isDirectory: true)) })
            CheckLabel(check: model.gameCheck)
            FolderRow(title: "存档目录",
                      path: model.prefs.userFolder.isEmpty ? "默认：\(model.userFolderURL.path)" : model.prefs.userFolder,
                      choose: { model.chooseUserFolder() },
                      reveal: { model.reveal(model.userFolderURL) },
                      canReset: !model.prefs.userFolder.isEmpty,
                      reset: { model.prefs.userFolder = "" })
            Picker("游戏语言", selection: $model.prefs.language) {
                ForEach(Catalog.languages) { Text($0.label).tag($0.value) }
            }
        } header: {
            Text("游戏")
        } footer: {
            Text("选择包含 eboot.bin、已合并 1.09 更新的游戏目录。存档和管线缓存保存在存档目录中。")
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
                Text("输出分辨率")
                Text("超分辨率按此尺寸输出画面")
            }
            Picker(selection: $model.game.liveResolution) {
                ForEach(Catalog.liveResolution) { Text($0.label).tag($0.value) }
            } label: {
                Text("实时切换分辨率")
                Text("无需重启，但较弱的 GPU 可能变慢")
            }
            Toggle("全屏", isOn: $model.prefs.fullscreen)
            Picker(selection: $model.prefs.presentMode) {
                ForEach(Catalog.presentModes) { Text($0.label).tag($0.value) }
            } label: {
                Text("画面呈现模式")
                Text("KosmicKrisp 不支持 Mailbox，会改用 FIFO")
            }
            Toggle("启用 HDR", isOn: $model.prefs.hdr)
        } header: {
            Text("显示")
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
                Text("帧率模式")
                Text("选择游戏使用的帧率补丁（BB_FPS）")
            }
            LabeledContent {
                HStack(spacing: 4) {
                    TextField("帧率上限", value: $model.prefs.fpsLimit, format: .number)
                        .labelsHidden()
                        .multilineTextAlignment(.trailing)
                        .frame(width: 56)
                    Stepper("帧率上限", value: $model.prefs.fpsLimit, in: 0...480, step: 5)
                        .labelsHidden()
                }
            } label: {
                Text("帧率上限")
                Text("0 表示跟随屏幕刷新率（最高 120 Hz）；其他数值为指定上限")
            }
        } header: {
            Text("帧率")
        } footer: {
            Text("目前已在 macOS 验证 30 FPS；60、90 和解除限制使用社区帧率补丁。")
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
                Text("超分辨率与抗锯齿")
                Text(hint(for: upscaler))
            }
            Picker("质量预设", selection: $model.game.preset) {
                ForEach(Catalog.presets) { Text($0.label).tag($0.value) }
            }
            .disabled(upscaler == "off" || upscaler == "taa")
            Toggle("锐化（RCAS）", isOn: $model.game.sharpen)
                .disabled(upscaler == "off")
            LabeledContent("锐化强度") {
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
                Text("物体运动矢量")
                Text("减少角色拖影；帧率可能下降约 10%")
            }
            Toggle("显示帧率", isOn: $model.game.showFPS)
        } header: {
            Text("超分辨率与抗锯齿")
        } footer: {
            VStack(alignment: .leading, spacing: 4) {
                if let replaced = model.game.replacedUpscaler {
                    Text("bbport.ini 指定了 macOS 不支持的 \(replaced)，已改用 FSR 3.1。")
                }
                Text("保存到 bbport.ini；游戏运行时可在设置面板中调整。")
            }
            .foregroundStyle(.secondary)
        }
    }

    private func hint(for upscaler: String) -> String {
        switch upscaler {
        case "fsr3": return "AMD FSR 3.1 时域超分辨率"
        case "taa": return "在输出分辨率下进行抗锯齿，不使用 FSR 模型"
        case "metalfx": return "Apple MetalFX 时域超分辨率。正在独立分支中开发；未包含此功能的版本会关闭超分辨率。"
        default: return "关闭超分辨率和时域抗锯齿"
        }
    }
}

struct EffectsSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            Picker("模型细节", selection: $model.game.modelLOD) {
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
            Text("游戏画面效果")
        } footer: {
            Text("通过游戏补丁实现，启动时生效。")
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
                Text("两阶段 GPU 管线")
                Text("可提升约 20–30% 性能；不稳定时关闭")
            }
            Picker("CPU 回读 GPU 数据", selection: $model.prefs.readbacks) {
                ForEach(Catalog.readbacks) { Text($0.label).tag($0.value) }
            }
        } header: {
            Text("性能")
        }
    }
}

struct ModsSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            Toggle("加载模组", isOn: $model.prefs.modsEnabled)
            FolderRow(title: "模组目录", path: model.modsFolderURL.path,
                      choose: { model.chooseModsFolder() },
                      reveal: { model.reveal(model.modsFolderURL) },
                      canReset: !model.prefs.modsFolder.isEmpty,
                      reset: { model.prefs.modsFolder = "" })
            if model.mods.isEmpty {
                Text("暂无模组")
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
                    .help("提前加载")
                    Button {
                        model.moveMod(mod.name, by: 1)
                    } label: {
                        Image(systemName: "chevron.down")
                    }
                    .buttonStyle(.borderless)
                    .help("延后加载")
                    Toggle(mod.name, isOn: $mod.isOn)
                        .labelsHidden()
                }
            }
            .disabled(!model.prefs.modsEnabled)
            Button("刷新列表") {
                model.saveModProfile()
                model.refreshMods()
            }
        } header: {
            Text("模组")
        } footer: {
            Text("每个模组解压到独立目录（包含 dvdroot_ps4，或直接包含 chr/、parts/ 等）。文件冲突时，列表靠下的模组优先。启动时生效。")
                .foregroundStyle(.secondary)
        }
    }
}

struct PatchesSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            FolderRow(title: "补丁目录", path: model.patchesFolderURL.path,
                      choose: { model.choosePatchesFolder() },
                      reveal: { model.reveal(model.patchesFolderURL) },
                      canReset: !model.prefs.patchesFolder.isEmpty,
                      reset: { model.prefs.patchesFolder = "" })
            if model.patches.isEmpty {
                Text("暂无补丁")
                    .foregroundStyle(.secondary)
            }
            ForEach($model.patches) { $patch in
                Toggle(isOn: $patch.isOn) {
                    Text(patch.title)
                    Text(patch.subtitle)
                }
                .help(patch.note ?? "")
            }
            Button("刷新列表") {
                model.savePatchProfile()
                model.refreshPatches()
            }
        } header: {
            Text("第三方补丁")
        } footer: {
            Text("从补丁目录加载适用于 01.09 的 shadPS4 格式 XML 补丁，启动时生效。")
                .foregroundStyle(.secondary)
        }
    }
}

struct DeveloperSection: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Section {
            Toggle(isOn: $model.prefs.frameStats) {
                Text("在日志中记录帧统计")
                Text("BB_FRAME_STATS")
            }
            Toggle(isOn: $model.prefs.gpuProfile) {
                Text("在日志中记录 GPU 性能数据")
                Text("BB_GPU_PROFILE")
            }
            Toggle(isOn: $model.prefs.vkValidation) {
                Text("Vulkan 验证层")
                Text("会显著降低速度；需要先安装验证层")
            }
            TextField(text: $model.prefs.extraEnv, prompt: Text("NAME=value NAME2=value")) {
                Text("额外环境变量")
            }
        } header: {
            Text("开发者选项")
        } footer: {
            Text("多个变量用空格分隔，会覆盖上方对应设置。")
                .foregroundStyle(.secondary)
        }
    }
}
