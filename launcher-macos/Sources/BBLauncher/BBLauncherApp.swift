// SPDX-License-Identifier: GPL-2.0-or-later
// bbport launcher for macOS: picks the game folder, edits bbport.ini and the start-up
// environment, and runs tools/macos/run.sh (the Mac counterpart of launcher/bbport_launcher.py).
import AppKit
import SwiftUI

@main
struct BBLauncherApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("血源诅咒", id: "main") {
            ContentView()
                .environmentObject(LauncherModel.shared)
        }
        .defaultSize(width: 700, height: 880)

        Window("运行日志", id: "log") {
            LogView()
                .environmentObject(LauncherModel.shared)
        }
        .defaultSize(width: 900, height: 560)

        Settings {
            LauncherSettingsView()
                .environmentObject(LauncherModel.shared)
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Also come to the front when started as a bare executable (swift run).
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        LauncherModel.shared.prepareForQuit()
    }
}

/// Settings (⌘,): where the bbport repository is.
struct LauncherSettingsView: View {
    @EnvironmentObject var model: LauncherModel

    var body: some View {
        Form {
            Section {
                LabeledContent("构建时记录的目录") {
                    Text(model.embeddedRepoPath ?? "无")
                        .textSelection(.enabled)
                        .truncationMode(.middle)
                }
                TextField(text: $model.prefs.repoOverride, prompt: Text("使用构建时记录的目录")) {
                    Text("自定义目录")
                }
                HStack {
                    Button("选择…") { model.chooseRepository() }
                    Button("恢复构建时记录的目录") { model.prefs.repoOverride = "" }
                        .disabled(model.prefs.repoOverride.isEmpty)
                }
                CheckLabel(check: model.repoCheck)
            } header: {
                Text("bbport 项目目录")
            } footer: {
                Text("选择包含 tools/macos/run.sh 的项目目录。构建脚本会记录构建时使用的项目路径。")
                    .foregroundStyle(.secondary)
            }
            Section {
                TextField(text: $model.prefs.branch, prompt: Text("保留当前分支")) {
                    Text("更新分支")
                }
            } header: {
                Text("更新")
            } footer: {
                Text("更新时从 GitHub 获取此分支并重新构建。macos-port 为主分支，其他分支用于测试。留空表示保留当前分支。")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 560, height: 440)
    }
}
