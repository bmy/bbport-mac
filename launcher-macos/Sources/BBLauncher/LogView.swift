// SPDX-License-Identifier: GPL-2.0-or-later
// The game's output, live. Lines are lazy rows so a long log stays cheap to update.
import AppKit
import SwiftUI

struct LogView: View {
    @EnvironmentObject var model: LauncherModel
    @State private var follow = true

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(model.logLines) { line in
                        Text(line.text.isEmpty ? " " : line.text)
                            .font(.system(size: 11, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(line.id)
                    }
                }
                .padding(8)
            }
            .defaultScrollAnchor(.bottom)
            .background(Color(nsColor: .textBackgroundColor))
            .onChange(of: model.logLines.last?.id) { _, newID in
                guard follow, let newID else { return }
                proxy.scrollTo(newID, anchor: .bottom)
            }
            .overlay {
                if model.logLines.isEmpty {
                    Text("Press Play: the game's output appears here.")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(minWidth: 480, minHeight: 240)
        .navigationTitle("Game Log")
        .navigationSubtitle(model.statusText)
        .toolbar {
            ToolbarItemGroup {
                Toggle(isOn: $follow) {
                    Label("Follow Output", systemImage: "arrow.down.to.line")
                }
                .help("Keep the newest output in view")
                Button {
                    model.copyLog()
                } label: {
                    Label("Copy Log", systemImage: "doc.on.doc")
                }
                .help("Copy the whole log")
                Button {
                    NSWorkspace.shared.open(model.lastRunLogURL)
                } label: {
                    Label("Open last-run.log", systemImage: "doc.text")
                }
                .help("out/last-run.log: the full output of the last run (send this when reporting a problem)")
                .disabled(!FileManager.default.fileExists(atPath: model.lastRunLogURL.path))
                Button {
                    model.clearLog()
                } label: {
                    Label("Clear", systemImage: "trash")
                }
                .help("Clear the log view")
                .disabled(model.isRunning)
            }
        }
    }
}
