// SPDX-License-Identifier: GPL-2.0-or-later
// bbport launcher for macOS: picks the game folder, edits bbport.ini and the start-up
// environment, and runs tools/macos/run.sh (the Mac counterpart of launcher/bbport_launcher.py).
import AppKit
import SwiftUI

@main
struct BBLauncherApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("Bloodborne", id: "main") {
            ContentView()
                .environmentObject(LauncherModel.shared)
        }
        .defaultSize(width: 700, height: 880)

        Window("Game Log", id: "log") {
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
                LabeledContent("Embedded at build") {
                    Text(model.embeddedRepoPath ?? "None")
                        .textSelection(.enabled)
                        .truncationMode(.middle)
                }
                TextField(text: $model.prefs.repoOverride, prompt: Text("Use the embedded path")) {
                    Text("Override")
                }
                HStack {
                    Button("Choose…") { model.chooseRepository() }
                    Button("Use the Embedded Path") { model.prefs.repoOverride = "" }
                        .disabled(model.prefs.repoOverride.isEmpty)
                }
                CheckLabel(check: model.repoCheck)
            } header: {
                Text("bbport repository")
            } footer: {
                Text("The folder with tools/macos/run.sh. tools/macos/build_launcher.sh embeds the checkout it was run from.")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 560, height: 320)
    }
}
