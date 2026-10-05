// SPDX-License-Identifier: GPL-2.0-or-later
// Game folder and repository checks, and the mod / third-party patch discovery of
// scripts/mods.py and scripts/patches.py (the launcher's lists; run.sh does the real work).
import Foundation

struct Check: Equatable {
    enum Level { case ok, warning, error }
    var level: Level
    var message: String
    /// Play stays available with warnings (the check may be stricter than the port).
    var allowsPlay: Bool { level != .error }
}

private func isDirectory(_ url: URL) -> Bool {
    var directory: ObjCBool = false
    return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && directory.boolValue
}

private func isFile(_ url: URL) -> Bool {
    var directory: ObjCBool = false
    return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && !directory.boolValue
}

private func entries(of folder: URL) -> [URL] {
    (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
}

// MARK: - Game folder

/// Title IDs of Bloodborne (scripts/patches.py BLOODBORNE_IDS).
let bloodborneTitleIDs: Set<String> = ["CUSA00207", "CUSA00208", "CUSA00900", "CUSA01363", "CUSA03173", "CUSA03023"]

/// String and integer values of a PS4 param.sfo (scripts/prepare.py sfo()).
func readSFO(_ url: URL) -> [String: String]? {
    guard let data = try? Data(contentsOf: url), data.count >= 20 else { return nil }
    let bytes = [UInt8](data)
    guard bytes[0] == 0, bytes[1] == 0x50, bytes[2] == 0x53, bytes[3] == 0x46 else { return nil }
    func u16(_ offset: Int) -> Int {
        Int(bytes[offset]) | (Int(bytes[offset + 1]) << 8)
    }
    func u32(_ offset: Int) -> Int {
        let low = Int(bytes[offset]) | (Int(bytes[offset + 1]) << 8)
        let high = (Int(bytes[offset + 2]) << 16) | (Int(bytes[offset + 3]) << 24)
        return low | high
    }
    let keyTable = u32(8)
    let dataTable = u32(12)
    let count = u32(16)
    var result: [String: String] = [:]
    for index in 0..<count {
        let entry = 20 + index * 16
        guard entry + 16 <= bytes.count else { break }
        let keyStart = keyTable + u16(entry)
        let format = u16(entry + 2)
        let size = u32(entry + 4)
        let dataStart = dataTable + u32(entry + 12)
        guard keyStart < bytes.count, dataStart + size <= bytes.count else { continue }
        let keyEnd = bytes[keyStart...].firstIndex(of: 0) ?? bytes.count
        let key = String(decoding: bytes[keyStart..<keyEnd], as: UTF8.self)
        if format == 0x0404 {
            if size >= 4 { result[key] = String(u32(dataStart)) }
        } else {
            var value = Array(bytes[dataStart..<(dataStart + size)])
            while value.last == 0 { value.removeLast() }
            result[key] = String(decoding: value, as: UTF8.self)
        }
    }
    return result
}

/// The Linux launcher only requires eboot.bin; the title and version are reported as warnings.
func checkGameFolder(_ path: String) -> Check {
    if path.isEmpty {
        return Check(level: .error, message: "Not chosen: pick the folder with eboot.bin")
    }
    let folder = URL(fileURLWithPath: path, isDirectory: true)
    if !isDirectory(folder) {
        return Check(level: .error, message: "The folder does not exist")
    }
    if !isFile(folder.appendingPathComponent("eboot.bin")) {
        return Check(level: .error, message: "No eboot.bin in the folder")
    }
    guard let sfo = readSFO(folder.appendingPathComponent("sce_sys/param.sfo")) else {
        return Check(level: .warning, message: "eboot.bin found; sce_sys/param.sfo is missing or unreadable")
    }
    let title = sfo["TITLE_ID"] ?? "?"
    let version = sfo["APP_VER"] ?? "?"
    if !bloodborneTitleIDs.contains(title) {
        return Check(level: .warning, message: "eboot.bin found, but the title is \(title), not Bloodborne")
    }
    if version != "01.09" {
        return Check(level: .warning,
                     message: "\(title) version \(version) found; the port expects 01.09 (the game merged with update 1.09)")
    }
    return Check(level: .ok, message: "Bloodborne \(title), version \(version)")
}

// MARK: - Repository

/// The repository root: has tools/macos/run.sh.
func isRepository(_ url: URL) -> Bool {
    isFile(url.appendingPathComponent("tools/macos/run.sh"))
}

/// A repository above `start` (the app bundle sits in <repo>/out when not moved).
func findRepository(above start: URL) -> URL? {
    var url = start.standardizedFileURL
    for _ in 0..<6 {
        if isRepository(url) { return url }
        let parent = url.deletingLastPathComponent()
        if parent.path == url.path { break }
        url = parent
    }
    return nil
}

func checkRepository(_ url: URL?) -> Check {
    guard let url else {
        return Check(level: .error,
                     message: "bbport repository not found: rebuild with tools/macos/build_launcher.sh or set it in Settings")
    }
    if !isRepository(url) {
        return Check(level: .error, message: "\(url.path) has no tools/macos/run.sh: set the repository in Settings")
    }
    if !FileManager.default.isExecutableFile(atPath: url.appendingPathComponent("out/bb-probe").path) {
        return Check(level: .error, message: "The port is not built yet: run bash tools/macos/build.sh in Terminal")
    }
    if !isFile(url.appendingPathComponent("deps-x86_64/env.sh")) {
        return Check(level: .warning, message: "deps-x86_64/env.sh is missing: run bash tools/macos/setup_deps.sh")
    }
    return Check(level: .ok, message: url.path)
}

// MARK: - Mods (scripts/mods.py)

/// Top-level folders of the game's dvdroot_ps4.
private let gameFolders: Set<String> = ["action", "chr", "event", "facegen", "font", "map", "menu", "movie",
                                        "msg", "mtd", "obj", "other", "param", "paramdef", "parts", "remo",
                                        "script", "sfx", "shader", "sound"]

/// `name` inside `folder`, matched case-insensitively (mods.py child()).
private func child(_ folder: URL, _ name: String) -> URL {
    let lowered = name.lowercased()
    return entries(of: folder).first { $0.lastPathComponent.lowercased() == lowered }
        ?? folder.appendingPathComponent(name)
}

/// mods.py content_root() != None: a layout run.sh accepts.
private func hasModLayout(_ folder: URL, depth: Int = 0) -> Bool {
    guard depth < 8, isDirectory(folder) else { return false }
    for wrapper in ["", "app0", "CUSA03173"] {
        let base = wrapper.isEmpty ? folder : child(folder, wrapper)
        if isDirectory(child(base, "dvdroot_ps4")) { return true }
    }
    let visible = entries(of: folder).filter { !$0.lastPathComponent.hasPrefix(".") }
    let folders = visible.filter(isDirectory)
    if !folders.isEmpty && folders.allSatisfy({ gameFolders.contains($0.lastPathComponent.lowercased()) }) {
        return true
    }
    let ignored: Set<String> = ["txt", "md", "jpg", "png", "ini"]
    let files = visible.filter { isFile($0) && !ignored.contains($0.pathExtension.lowercased()) }
    if folders.count == 1 && files.isEmpty {
        return hasModLayout(folders[0], depth: depth + 1)
    }
    return false
}

/// Mod folder names, sorted as mods.py discover() does.
func discoverMods(in root: URL) -> [String] {
    guard isDirectory(root) else { return [] }
    let names = entries(of: root).filter { isDirectory($0) && hasModLayout($0) }.map(\.lastPathComponent)
    return names.sorted { ($0.lowercased(), $0) < ($1.lowercased(), $1) }
}

// MARK: - Third-party patches (scripts/patches.py external_patches)

struct PatchMetadata {
    let key: String
    let name: String
    let author: String?
    let note: String?
    let enabledByDefault: Bool
}

/// Collects <ID> texts and <Metadata> attributes of a shadPS4 patch file.
private final class PatchFileReader: NSObject, XMLParserDelegate {
    var ids: Set<String> = []
    var metadata: [[String: String]] = []
    private var inID = false
    private var idText = ""

    func parser(_ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?,
                qualifiedName qName: String?, attributes attributeDict: [String: String] = [:]) {
        if elementName == "ID" {
            inID = true
            idText = ""
        } else if elementName == "Metadata" {
            metadata.append(attributeDict)
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        if inID { idText += string }
    }

    func parser(_ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?,
                qualifiedName qName: String?) {
        if elementName == "ID" {
            inID = false
            let id = idText.trimmingCharacters(in: .whitespacesAndNewlines)
            if !id.isEmpty { ids.insert(id) }
        }
    }
}

/// Patches for version 01.09's eboot.bin in directory/*.xml, keyed "<file name>/<patch name>"
/// (the patches.json selection). The repository's own patches/Bloodborne.xml is skipped.
func discoverPatches(in directory: URL, builtIn: URL) -> [PatchMetadata] {
    guard isDirectory(directory) else { return [] }
    let skip = builtIn.resolvingSymlinksInPath().path
    let files = entries(of: directory)
        .filter { $0.pathExtension == "xml" && isFile($0) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    var found: [PatchMetadata] = []
    for file in files where file.resolvingSymlinksInPath().path != skip {
        guard let parser = XMLParser(contentsOf: file) else { continue }
        let reader = PatchFileReader()
        parser.delegate = reader
        guard parser.parse() else { continue }
        if !reader.ids.isEmpty && reader.ids.isDisjoint(with: bloodborneTitleIDs) { continue }
        for meta in reader.metadata where meta["AppVer"] == "01.09" && (meta["AppElf"] ?? "eboot.bin") == "eboot.bin" {
            let name = meta["Name"] ?? "None"
            found.append(PatchMetadata(
                key: "\(file.lastPathComponent)/\(name)",
                name: name,
                author: meta["Author"].flatMap { $0.isEmpty ? nil : $0 },
                note: meta["Note"].flatMap { $0.isEmpty ? nil : $0.replacingOccurrences(of: "\\n", with: "\n") },
                enabledByDefault: (meta["isEnabled"] ?? "false").lowercased() == "true"))
        }
    }
    return found
}
