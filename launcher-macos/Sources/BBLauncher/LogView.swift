// SPDX-License-Identifier: GPL-2.0-or-later
// The game's output, live. Lines are lazy rows so a long log stays cheap to update.
import AppKit
import SwiftUI

struct LogView: View {
    @EnvironmentObject var model: LauncherModel
    // Not @State: in the macOS 27 SDK @State is a macro whose plugin ships only with Xcode, so
    // a Command Line Tools build fails; the flag lives in the model instead.

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
                guard model.followLog, let newID else { return }
                proxy.scrollTo(newID, anchor: .bottom)
            }
            .overlay {
                if model.logLines.isEmpty {
                    Text("点击“启动游戏”后，运行日志会显示在这里。")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(minWidth: 480, minHeight: 240)
        .navigationTitle("运行日志")
        .navigationSubtitle(model.statusText)
        .toolbar {
            ToolbarItemGroup {
                Toggle(isOn: $model.followLog) {
                    Label("跟随最新日志", systemImage: "arrow.down.to.line")
                }
                .help("自动滚动到最新日志")
                Button {
                    model.copyLog()
                } label: {
                    Label("复制日志", systemImage: "doc.on.doc")
                }
                .help("复制全部日志")
                Button {
                    NSWorkspace.shared.open(model.lastRunLogURL)
                } label: {
                    Label("打开 last-run.log", systemImage: "doc.text")
                }
                .help("out/last-run.log：上次运行的完整日志，反馈问题时可提供此文件")
                .disabled(!FileManager.default.fileExists(atPath: model.lastRunLogURL.path))
                Button {
                    model.clearLog()
                } label: {
                    Label("清空", systemImage: "trash")
                }
                .help("清空日志显示")
                .disabled(model.isRunning)
            }
        }
    }
}
