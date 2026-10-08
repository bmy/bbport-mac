// SPDX-License-Identifier: GPL-2.0-or-later
// Launcher state: settings, checks, mods and patches, and the running game (Process).
import AppKit
import Foundation

struct ModEntry: Identifiable, Equatable {
    let name: String
    var isOn: Bool
    var id: String { name }
}

struct PatchEntry: Identifiable, Equatable {
    let key: String
    let title: String
    let subtitle: String
    let note: String?
    let enabledByDefault: Bool
    var isOn: Bool
    var id: String { key }
}

struct LogLine: Identifiable {
    let id: Int
    let text: String
}

/// What the started process is: the game (run.sh) or an update (tools/macos/update.sh).
enum Job: Equatable {
    case game
    case update
}

enum RunState: Equatable {
    case idle
    case running
    case stopping
    case exited(status: Int32, bySignal: Bool)
    case failed(String)
}

@MainActor
final class LauncherModel: ObservableObject {
    static let shared = LauncherModel()

    static let maxLogLines = 5000

    @Published var prefs: LauncherPrefs {
        didSet { prefsChanged(from: oldValue) }
    }
    /// bbport.ini values edited by the form; written when the game starts.
    @Published var game = GameSettings()
    @Published var mods: [ModEntry] = []
    @Published var patches: [PatchEntry] = []

    @Published private(set) var repoCheck = Check(level: .error, message: "")
    @Published private(set) var gameCheck = Check(level: .error, message: "")
    @Published private(set) var state: RunState = .idle
    @Published private(set) var job: Job = .game
    @Published private(set) var logLines: [LogLine] = []
    /// Log window: keep scrolled to the newest line.
    @Published var followLog = true
    /// The last "STOP:" or error line, shown when the game exits with an error.
    @Published private(set) var lastProblem: String? = nil

    /// BBRepoPath from Info.plist, written by tools/macos/build_launcher.sh.
    let embeddedRepoPath: String?

    private var process: Process?
    private var collector: OutputCollector?
    private var pollTask: Task<Void, Never>?
    private var exitSeen: Date?
    private var nextLineID = 0
    private var ini = IniFile()
    /// The repository bbport.ini, mods and patches were last read from.
    private var loadedRepo: URL?
    private var iconFolder = ""

    private init() {
        embeddedRepoPath = Bundle.main.object(forInfoDictionaryKey: "BBRepoPath") as? String
        prefs = LauncherPrefs.load(from: UserDefaults.standard)
        refreshChecks()
        reloadRepositoryFiles()
    }

    // MARK: - Paths

    /// Settings override, else the path embedded at build time, else a repository above the app.
    var repoURL: URL? {
        if !prefs.repoOverride.isEmpty {
            return URL(fileURLWithPath: (prefs.repoOverride as NSString).expandingTildeInPath, isDirectory: true)
        }
        if let embeddedRepoPath, !embeddedRepoPath.isEmpty {
            let url = URL(fileURLWithPath: embeddedRepoPath, isDirectory: true)
            if isRepository(url) { return url }
        }
        return findRepository(above: Bundle.main.bundleURL) ?? embeddedRepoPath.map { URL(fileURLWithPath: $0, isDirectory: true) }
    }

    /// Generated files, saves, bbport.ini, mods.json and patches.json live in the repository
    /// (run.sh's data directory when BB_DATA_DIR is unset).
    private var dataURL: URL { repoURL ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true) }

    /// Nothing is written outside a real checkout (a mistyped override, no embedded path).
    private var hasRepository: Bool { repoURL.map(isRepository) ?? false }

    var iniURL: URL { dataURL.appendingPathComponent("bbport.ini") }
    var lastRunLogURL: URL { dataURL.appendingPathComponent("out/last-run.log") }

    var userFolderURL: URL { folder(prefs.userFolder, default: "user") }
    var modsFolderURL: URL { folder(prefs.modsFolder, default: "mods") }
    var patchesFolderURL: URL { folder(prefs.patchesFolder, default: "patches") }
    private var modsConfigURL: URL { dataURL.appendingPathComponent("mods.json") }
    private var patchesConfigURL: URL { dataURL.appendingPathComponent("patches.json") }

    private func folder(_ custom: String, default name: String) -> URL {
        if custom.isEmpty { return dataURL.appendingPathComponent(name, isDirectory: true) }
        return URL(fileURLWithPath: (custom as NSString).expandingTildeInPath, isDirectory: true)
    }

    // MARK: - State

    var isRunning: Bool { process != nil }

    var canPlay: Bool {
        !isRunning && repoCheck.allowsPlay && gameCheck.allowsPlay
    }

    var canUpdate: Bool {
        !isRunning && hasRepository
    }

    var statusText: String {
        switch state {
        case .idle:
            if !repoCheck.allowsPlay { return repoCheck.message }
            if !gameCheck.allowsPlay { return "Choose the game folder" }
            return "Ready"
        case .running:
            return job == .update ? "Updating…" : "Running"
        case .stopping:
            return "Stopping…"
        case let .exited(status, bySignal):
            if bySignal { return "Stopped (signal \(status))" }
            if job == .update {
                return status == 0 ? "Updated" : "The update failed: see the log"
            }
            return status == 0 ? "The game exited" : "The game exited with code \(status): see the log"
        case let .failed(message):
            return "Could not start: \(message)"
        }
    }

    private func prefsChanged(from old: LauncherPrefs) {
        prefs.save(to: UserDefaults.standard)
        if prefs.repoOverride != old.repoOverride {
            refreshChecks()
            if repoURL != loadedRepo { reloadRepositoryFiles() }
            return
        }
        if prefs.gameFolder != old.gameFolder { refreshGameCheck() }
        if prefs.modsFolder != old.modsFolder { refreshMods() }
        if prefs.patchesFolder != old.patchesFolder { refreshPatches() }
    }

    /// bbport.ini and the mod and patch lists of the current repository.
    private func reloadRepositoryFiles() {
        loadedRepo = repoURL
        reloadIni()
        refreshMods()
        refreshPatches()
    }

    /// Cheap checks, also run when the app becomes active (a build may have finished meanwhile).
    func refreshChecks() {
        repoCheck = checkRepository(repoURL)
        refreshGameCheck()
    }

    private func refreshGameCheck() {
        gameCheck = checkGameFolder(prefs.gameFolder)
        // The game's own icon for the Dock while no bundle icon was built from it.
        if gameCheck.allowsPlay && prefs.gameFolder != iconFolder {
            iconFolder = prefs.gameFolder
            let icon = URL(fileURLWithPath: prefs.gameFolder).appendingPathComponent("sce_sys/icon0.png")
            if let image = NSImage(contentsOf: icon) {
                NSApplication.shared.applicationIconImage = image
            }
        }
    }

    func reloadIni() {
        ini = IniFile.load(iniURL)
        game = GameSettings(ini: ini.values)
    }

    private func saveIni() throws {
        guard hasRepository else { return }
        ini = IniFile.load(iniURL)  // the in-game menu may have rewritten it
        let text = ini.text(applying: game.iniValues)
        try text.write(to: iniURL, atomically: true, encoding: .utf8)
        ini = IniFile.load(iniURL)
    }

    // MARK: - Folders

    func chooseGameFolder() {
        if let path = chooseFolder(message: "Choose the game folder (with eboot.bin)", startingAt: prefs.gameFolder) {
            prefs.gameFolder = path
        }
    }

    func chooseUserFolder() {
        if let path = chooseFolder(message: "Choose the saves folder", startingAt: userFolderURL.path) {
            prefs.userFolder = path
        }
    }

    func chooseModsFolder() {
        saveModProfile()
        if let path = chooseFolder(message: "Choose the mods folder", startingAt: modsFolderURL.path) {
            prefs.modsFolder = path
        }
    }

    func choosePatchesFolder() {
        savePatchProfile()
        if let path = chooseFolder(message: "Choose the patches folder", startingAt: patchesFolderURL.path) {
            prefs.patchesFolder = path
        }
    }

    func chooseRepository() {
        if let path = chooseFolder(message: "Choose the bbport repository (with tools/macos/run.sh)",
                                   startingAt: repoURL?.path) {
            prefs.repoOverride = path
        }
    }

    private func chooseFolder(message: String, startingAt path: String?) -> String? {
        let panel = NSOpenPanel()
        panel.message = message
        panel.prompt = "Choose"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        if let path, !path.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
        }
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        return url.path
    }

    /// Opens a folder in Finder, creating it first so that a new saves or mods folder opens.
    func reveal(_ url: URL) {
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.open(url)
    }

    // MARK: - Mods (mods.json: {"order": [...], "disabled": [...]})

    func refreshMods() {
        let profile = readJSON(modsConfigURL)
        let order = profile["order"] as? [String] ?? []
        let disabled = Set(profile["disabled"] as? [String] ?? [])
        let available = discoverMods(in: modsFolderURL)
        let availableSet = Set(available)
        var seen = Set<String>()
        var names: [String] = []
        // Saved order first; new folders are appended alphabetically (mods.py selected()).
        for name in order + available where availableSet.contains(name) && !seen.contains(name) {
            seen.insert(name)
            names.append(name)
        }
        mods = names.map { ModEntry(name: $0, isOn: !disabled.contains($0)) }
    }

    func moveMod(_ name: String, by offset: Int) {
        guard let index = mods.firstIndex(where: { $0.name == name }) else { return }
        let target = index + offset
        guard mods.indices.contains(target) else { return }
        mods.swapAt(index, target)
        saveModProfile()
    }

    func saveModProfile() {
        guard hasRepository else { return }
        writeJSON(["order": mods.map(\.name), "disabled": mods.filter { !$0.isOn }.map(\.name)], to: modsConfigURL)
    }

    // MARK: - Third-party patches (patches.json: the switches that differ from isEnabled)

    func refreshPatches() {
        let profile = readJSON(patchesConfigURL)
        let enabled = Set(profile["enabled"] as? [String] ?? [])
        let disabled = Set(profile["disabled"] as? [String] ?? [])
        let builtIn = dataURL.appendingPathComponent("patches/Bloodborne.xml")
        patches = discoverPatches(in: patchesFolderURL, builtIn: builtIn).map { meta in
            var subtitle = String(meta.key.split(separator: "/", maxSplits: 1).first ?? "")
            if let author = meta.author { subtitle += " · Author: \(author)" }
            let isOn = enabled.contains(meta.key) || (meta.enabledByDefault && !disabled.contains(meta.key))
            return PatchEntry(key: meta.key, title: meta.name, subtitle: subtitle, note: meta.note,
                              enabledByDefault: meta.enabledByDefault, isOn: isOn)
        }
    }

    func savePatchProfile() {
        guard hasRepository else { return }
        let profile = readJSON(patchesConfigURL)
        let shown = Set(patches.map(\.key))
        // Patches of files not listed now (another folder) keep their choice.
        var enabled = Set((profile["enabled"] as? [String] ?? []).filter { !shown.contains($0) })
        var disabled = Set((profile["disabled"] as? [String] ?? []).filter { !shown.contains($0) })
        for patch in patches where patch.isOn != patch.enabledByDefault {
            if patch.isOn { enabled.insert(patch.key) } else { disabled.insert(patch.key) }
        }
        writeJSON(["enabled": enabled.sorted(), "disabled": disabled.sorted()], to: patchesConfigURL)
    }

    private func readJSON(_ url: URL) -> [String: Any] {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return object
    }

    private func writeJSON(_ object: [String: [String]], to url: URL) {
        guard var data = try? JSONSerialization.data(withJSONObject: object,
                                                     options: [.prettyPrinted, .sortedKeys]) else { return }
        data.append(0x0A)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }

    // MARK: - Environment for tools/macos/run.sh

    /// Every variable the launcher controls is set explicitly (or removed), so the result does
    /// not depend on run.sh's defaults or on the environment the app was opened from.
    func environment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let p = prefs
        // Apps opened from Finder get a minimal PATH; run.sh needs Homebrew's python3.
        let path = env["PATH"].flatMap { $0.isEmpty ? nil : $0 } ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + path
        env["PYTHONUNBUFFERED"] = "1"
        // run.sh derives these from bbport.ini (output_res, live_resolution).
        for key in ["BB_RENDER_RES", "BB_OUTPUT_RES", "BB_AUTO_RENDER_RES"] {
            env.removeValue(forKey: key)
        }
        env["BB_CONFIG"] = iniURL.path
        env["BB_GAME_DIR"] = (p.gameFolder as NSString).expandingTildeInPath
        env["BB_USER_DIR"] = userFolderURL.path
        env["BB_MODS_DIR"] = modsFolderURL.path
        env["BB_MODS_CONFIG"] = modsConfigURL.path
        env["BB_MODS_ENABLED"] = p.modsEnabled ? "1" : "0"
        env["BB_PATCHES_DIR"] = patchesFolderURL.path
        env["BB_PATCHES_CONFIG"] = patchesConfigURL.path
        env["BB_LANGUAGE"] = p.language
        env["BB_FULLSCREEN"] = p.fullscreen ? "1" : "0"
        env["BB_PRESENT_MODE"] = p.presentMode
        env["BB_HDR"] = p.hdr ? "1" : "0"
        env["BB_FPS"] = p.fpsMode
        // The ini's upscaler is overridden by BB_UPSCALER at start; both carry the same value.
        env["BB_UPSCALER"] = game.upscaler
        let limit = min(max(p.fpsLimit, 0), 480)
        if limit > 0 { env["BB_FPS_LIMIT"] = String(limit) } else { env.removeValue(forKey: "BB_FPS_LIMIT") }
        if p.drawPipe.isEmpty { env.removeValue(forKey: "BB_DRAW_PIPE") } else { env["BB_DRAW_PIPE"] = p.drawPipe }
        if p.readbacks.isEmpty { env.removeValue(forKey: "BB_READBACKS") } else { env["BB_READBACKS"] = p.readbacks }
        if p.preupload.isEmpty { env.removeValue(forKey: "BB_PREUPLOAD") } else { env["BB_PREUPLOAD"] = p.preupload }
        // Set only when on: parts of the port test these for presence, not value ("0" is on).
        for (key, on) in [("BB_NATIVE_GPU", p.nativeGpu),
                          ("BB_FRAME_STATS", p.frameStats), ("BB_GPU_PROFILE", p.gpuProfile),
                          ("BB_VK_VALIDATION", p.vkValidation)] {
            if on { env[key] = "1" } else { env.removeValue(forKey: key) }
        }
        for item in p.extraEnv.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" }) {
            guard let eq = item.firstIndex(of: "="), eq != item.startIndex else { continue }
            env[String(item[..<eq])] = String(item[item.index(after: eq)...])
        }
        return env
    }

    // MARK: - Running the game

    func play() {
        guard process == nil else { return }
        refreshChecks()
        guard let repo = repoURL, repoCheck.allowsPlay, gameCheck.allowsPlay else { return }
        saveModProfile()
        savePatchProfile()
        do {
            try saveIni()
        } catch {
            state = .failed("cannot write \(iniURL.path): \(error.localizedDescription)")
            return
        }

        let env = environment()
        let summary = ["BB_GAME_DIR", "BB_FPS", "BB_UPSCALER", "BB_FULLSCREEN", "BB_PRESENT_MODE"]
            .map { "\($0)=\(env[$0] ?? "")" }.joined(separator: " ")
        start(.game, arguments: ["tools/macos/run.sh"], in: repo, env: env,
              echo: "$ \(summary) bash tools/macos/run.sh")
    }

    /// Fetches the branch from GitHub and rebuilds what changed (tools/macos/update.sh). An empty
    /// branch preference keeps the current one.
    func update() {
        guard process == nil, let repo = repoURL, hasRepository else { return }
        let branch = prefs.branch.trimmingCharacters(in: .whitespaces)
        var arguments = ["tools/macos/update.sh"]
        if !branch.isEmpty { arguments.append(branch) }
        start(.update, arguments: arguments, in: repo, env: environment(),
              echo: "$ bash " + arguments.joined(separator: " "))
    }

    private func start(_ job: Job, arguments: [String], in repo: URL, env: [String: String], echo: String) {
        logLines = []
        lastProblem = nil
        exitSeen = nil
        self.job = job
        appendLines(["$ cd \(repo.path)", echo])

        let pipe = Pipe()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = arguments
        process.currentDirectoryURL = repo
        process.environment = env
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = pipe
        process.standardError = pipe
        // The reader owns a duplicate of the read end (closed by the reader at end of file).
        let readFD = dup(pipe.fileHandleForReading.fileDescriptor)
        do {
            try process.run()
        } catch {
            if readFD >= 0 { close(readFD) }
            state = .failed(error.localizedDescription)
            appendLines(["Could not start: \(error.localizedDescription)"])
            return
        }
        let collector = OutputCollector()
        if readFD >= 0 {
            pumpOutput(from: readFD, into: collector)
        } else {
            collector.finish()
        }
        self.process = process
        self.collector = collector
        state = .running
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 100_000_000)
                guard let self else { return }
                if self.poll() { return }
            }
        }
    }

    /// SIGTERM; SIGKILL after 3 s. tools/macos/run.sh and run.sh exec into bb-probe (or forward
    /// TERM to it when mods are merged), so the game is the process we started.
    func stop() {
        guard let process, process.isRunning else { return }
        state = .stopping
        process.terminate()
        let pid = process.processIdentifier
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            guard let self, let current = self.process, current.processIdentifier == pid, current.isRunning else {
                return
            }
            kill(pid, SIGKILL)
        }
    }

    /// The app is quitting: ask the game to stop, as the Linux launcher does on close.
    func prepareForQuit() {
        if process == nil {
            saveModProfile()
            savePatchProfile()
            try? saveIni()
        }
        process?.terminate()
    }

    /// Moves new output into the log; true once the game has exited and the log is complete.
    private func poll() -> Bool {
        guard let collector, let process else { return true }
        let (lines, ended) = collector.drain()
        if !lines.isEmpty { appendLines(lines) }
        if process.isRunning { return false }
        if !ended {
            // A leftover child (tee, a Python step) may still hold the pipe: wait briefly.
            let now = Date()
            if exitSeen == nil { exitSeen = now }
            if let seen = exitSeen, now.timeIntervalSince(seen) < 2 { return false }
        }
        let bySignal = process.terminationReason == .uncaughtSignal
        let status = process.terminationStatus
        self.process = nil
        self.collector = nil
        pollTask = nil
        state = .exited(status: status, bySignal: bySignal)
        if job == .update {
            appendLines(["", status == 0 && !bySignal ? "— update finished —" : "— the update stopped (code \(status)) —"])
            // New code may bring new checks, mods or patches.
            refreshChecks()
            reloadRepositoryFiles()
            return true
        }
        appendLines(["", bySignal ? "— the game was stopped (signal \(status)) —" : "— the game exited (code \(status)) —"])
        // Pick up changes made in the in-game menu (it saves bbport.ini).
        reloadIni()
        return true
    }

    private func appendLines(_ lines: [String]) {
        var added: [LogLine] = []
        added.reserveCapacity(lines.count)
        for line in lines {
            if line.contains("STOP:") || line.contains("Traceback") || line.range(of: "error:", options: .caseInsensitive) != nil {
                lastProblem = line
            }
            added.append(LogLine(id: nextLineID, text: line))
            nextLineID += 1
        }
        var all = logLines
        all.append(contentsOf: added)
        if all.count > Self.maxLogLines {
            all.removeFirst(all.count - Self.maxLogLines)
        }
        logLines = all
    }

    func clearLog() {
        logLines = []
    }

    func copyLog() {
        let text = logLines.map(\.text).joined(separator: "\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
