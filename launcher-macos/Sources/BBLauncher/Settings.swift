// SPDX-License-Identifier: GPL-2.0-or-later
// Settings shown by the launcher. Mirrors launcher/bbport_launcher.py: launcher-only state lives
// in UserDefaults (the Linux launcher's settings.json), game settings in bbport.ini, which the
// port and its in-game menu also read and write (gpu/shim/bbport_settings.cpp).
import Foundation

/// A picker entry: the stored value and its English label.
struct Choice<Value: Hashable & Sendable>: Identifiable, Sendable {
    let value: Value
    let label: String
    var id: Value { value }
}

/// A game effect switched by a community patch at start (bbport.ini key, launcher EFFECTS).
struct EffectDefinition: Sendable {
    let key: String
    let title: String
    let defaultOn: Bool
    let note: String?
}

enum Catalog {
    // Upscalers offered on macOS (bbport.ini "upscaler", BB_UPSCALER). FSR 4 and 4.1.1 need
    // FP8/WMMA features the Mac Vulkan drivers lack; the port falls back to FSR 3.1 for them.
    static let upscalers: [Choice<String>] = [
        Choice(value: "off", label: "Off"),
        Choice(value: "fsr3", label: "FSR 3.1"),
        Choice(value: "taa", label: "TAA (native anti-aliasing)"),
        Choice(value: "metalfx", label: "MetalFX"),
    ]
    static let defaultUpscaler = "off"

    static let presets: [Choice<Int>] = [
        Choice(value: 0, label: "Native AA"),
        Choice(value: 1, label: "Quality (x1.5)"),
        Choice(value: 2, label: "Balanced (x1.7)"),
        Choice(value: 3, label: "Performance (x2)"),
        Choice(value: 4, label: "Ultra Performance (x3)"),
    ]
    // Linux defaults to 4 for FSR 4; FSR 3.1 and MetalFX look much better at Quality.
    static let defaultPreset = 1

    static let outputResolutions: [Choice<String>] = [
        Choice(value: "1280x720", label: "1280×720"),
        Choice(value: "1920x1080", label: "1920×1080"),
        Choice(value: "2560x1440", label: "2560×1440"),
        Choice(value: "3840x2160", label: "3840×2160"),
    ]

    static let liveResolution: [Choice<String>] = [
        Choice(value: "auto", label: "Auto (by GPU)"),
        Choice(value: "0", label: "Off (faster)"),
        Choice(value: "1", label: "On"),
    ]

    static let modelDetail: [Choice<String>] = [
        Choice(value: "0", label: "As in the game"),
        Choice(value: "-2", label: "Highest (-2)"),
        Choice(value: "1", label: "Lower (1)"),
        Choice(value: "2", label: "Lowest (2)"),
    ]

    static let effects: [EffectDefinition] = [
        EffectDefinition(key: "effect_chromatic_aberration", title: "Chromatic aberration", defaultOn: true, note: nil),
        EffectDefinition(key: "effect_dof", title: "Depth of field (DoF)", defaultOn: true, note: nil),
        EffectDefinition(key: "effect_motion_blur", title: "Motion blur", defaultOn: true, note: nil),
        EffectDefinition(key: "effect_ssao", title: "SSAO", defaultOn: true, note: nil),
        EffectDefinition(key: "effect_game_aa", title: "The game's own anti-aliasing", defaultOn: true, note: nil),
        EffectDefinition(key: "effect_dynamic_shadows", title: "Dynamic light shadows", defaultOn: true, note: nil),
        EffectDefinition(key: "effect_ssr", title: "SSR reflections (not in the original game)", defaultOn: false, note: nil),
        EffectDefinition(key: "skip_intro", title: "Skip the intro videos", defaultOn: false, note: nil),
        EffectDefinition(key: "debug_camera", title: "Free camera (Cross + L3 / Space + Z)", defaultOn: false, note: nil),
        EffectDefinition(key: "debug_menu", title: "Debug menu (left touchpad / Tab; needs fonts)", defaultOn: false,
                         note: "Install DbgFont14h.ccm and DbgFont14h.tpf into dvdroot_ps4/font from Nexus mod #253"),
    ]

    static let fpsModes: [Choice<String>] = [
        Choice(value: "uncap", label: "Unlocked (patch)"),
        Choice(value: "60", label: "60"),
        Choice(value: "90", label: "90"),
        Choice(value: "30", label: "30 (as on PS4)"),
    ]
    // 30 is what macOS has been validated with so far (tools/macos/run.sh's default).
    static let defaultFPSMode = "30"

    static let presentModes: [Choice<String>] = [
        Choice(value: "Mailbox", label: "Mailbox"),
        Choice(value: "Fifo", label: "FIFO (VSync)"),
        Choice(value: "FifoRelaxed", label: "FIFO Relaxed"),
        Choice(value: "Immediate", label: "Immediate"),
    ]

    static let drawPipe: [Choice<String>] = [
        Choice(value: "", label: "Auto (8+ threads)"),
        Choice(value: "1", label: "On"),
        Choice(value: "0", label: "Off (more stable)"),
    ]

    static let readbacks: [Choice<String>] = [
        Choice(value: "", label: "Relaxed (default)"),
        Choice(value: "0", label: "Off"),
        Choice(value: "2", label: "Precise"),
    ]

    static let languages: [Choice<String>] = [
        Choice(value: "1", label: "English"),
        Choice(value: "8", label: "Russian"),
        Choice(value: "0", label: "Japanese"),
        Choice(value: "2", label: "French"),
        Choice(value: "3", label: "Spanish"),
        Choice(value: "4", label: "German"),
        Choice(value: "5", label: "Italian"),
    ]

    static let vkDrivers: [Choice<String>] = [
        Choice(value: "kosmickrisp", label: "KosmicKrisp (default)"),
        Choice(value: "moltenvk", label: "MoltenVK"),
    ]

    static func contains<Value: Hashable & Sendable>(_ choices: [Choice<Value>], _ value: Value) -> Bool {
        choices.contains { $0.value == value }
    }
}

/// Launcher-only settings (UserDefaults, one key each so `defaults read io.github.bbport.mac`
/// shows them). Empty folder paths mean the default inside the repository.
struct LauncherPrefs: Equatable {
    var gameFolder = ""
    var repoOverride = ""
    var vkDriver = "kosmickrisp"
    var userFolder = ""
    var modsFolder = ""
    var modsEnabled = true
    var patchesFolder = ""
    var language = "1"
    var fullscreen = false
    var hdr = false
    var presentMode = "Mailbox"
    var fpsMode = Catalog.defaultFPSMode
    var fpsLimit = 0
    var drawPipe = ""
    var readbacks = ""
    var frameStats = false
    var gpuProfile = false
    var vkValidation = false
    var extraEnv = ""

    private enum Key {
        static let gameFolder = "gameFolder"
        static let repoOverride = "repoOverride"
        static let vkDriver = "vkDriver"
        static let userFolder = "userFolder"
        static let modsFolder = "modsFolder"
        static let modsEnabled = "modsEnabled"
        static let patchesFolder = "patchesFolder"
        static let language = "language"
        static let fullscreen = "fullscreen"
        static let hdr = "hdr"
        static let presentMode = "presentMode"
        static let fpsMode = "fpsMode"
        static let fpsLimit = "fpsLimit"
        static let drawPipe = "drawPipe"
        static let readbacks = "readbacks"
        static let frameStats = "frameStats"
        static let gpuProfile = "gpuProfile"
        static let vkValidation = "vkValidation"
        static let extraEnv = "extraEnv"
    }

    static func load(from defaults: UserDefaults) -> LauncherPrefs {
        var p = LauncherPrefs()
        func text(_ key: String, _ fallback: String) -> String { defaults.string(forKey: key) ?? fallback }
        func flag(_ key: String, _ fallback: Bool) -> Bool { defaults.object(forKey: key) as? Bool ?? fallback }
        p.gameFolder = text(Key.gameFolder, p.gameFolder)
        p.repoOverride = text(Key.repoOverride, p.repoOverride)
        p.vkDriver = text(Key.vkDriver, p.vkDriver)
        p.userFolder = text(Key.userFolder, p.userFolder)
        p.modsFolder = text(Key.modsFolder, p.modsFolder)
        p.modsEnabled = flag(Key.modsEnabled, p.modsEnabled)
        p.patchesFolder = text(Key.patchesFolder, p.patchesFolder)
        p.language = text(Key.language, p.language)
        p.fullscreen = flag(Key.fullscreen, p.fullscreen)
        p.hdr = flag(Key.hdr, p.hdr)
        p.presentMode = text(Key.presentMode, p.presentMode)
        p.fpsMode = text(Key.fpsMode, p.fpsMode)
        p.fpsLimit = defaults.object(forKey: Key.fpsLimit) as? Int ?? p.fpsLimit
        p.drawPipe = text(Key.drawPipe, p.drawPipe)
        p.readbacks = text(Key.readbacks, p.readbacks)
        p.frameStats = flag(Key.frameStats, p.frameStats)
        p.gpuProfile = flag(Key.gpuProfile, p.gpuProfile)
        p.vkValidation = flag(Key.vkValidation, p.vkValidation)
        p.extraEnv = text(Key.extraEnv, p.extraEnv)
        // Values from an older version that the pickers no longer offer.
        if !Catalog.contains(Catalog.vkDrivers, p.vkDriver) { p.vkDriver = "kosmickrisp" }
        if !Catalog.contains(Catalog.languages, p.language) { p.language = "1" }
        if !Catalog.contains(Catalog.presentModes, p.presentMode) { p.presentMode = "Mailbox" }
        if !Catalog.contains(Catalog.fpsModes, p.fpsMode) { p.fpsMode = Catalog.defaultFPSMode }
        if !Catalog.contains(Catalog.drawPipe, p.drawPipe) { p.drawPipe = "" }
        if !Catalog.contains(Catalog.readbacks, p.readbacks) { p.readbacks = "" }
        return p
    }

    func save(to defaults: UserDefaults) {
        defaults.set(gameFolder, forKey: Key.gameFolder)
        defaults.set(repoOverride, forKey: Key.repoOverride)
        defaults.set(vkDriver, forKey: Key.vkDriver)
        defaults.set(userFolder, forKey: Key.userFolder)
        defaults.set(modsFolder, forKey: Key.modsFolder)
        defaults.set(modsEnabled, forKey: Key.modsEnabled)
        defaults.set(patchesFolder, forKey: Key.patchesFolder)
        defaults.set(language, forKey: Key.language)
        defaults.set(fullscreen, forKey: Key.fullscreen)
        defaults.set(hdr, forKey: Key.hdr)
        defaults.set(presentMode, forKey: Key.presentMode)
        defaults.set(fpsMode, forKey: Key.fpsMode)
        defaults.set(fpsLimit, forKey: Key.fpsLimit)
        defaults.set(drawPipe, forKey: Key.drawPipe)
        defaults.set(readbacks, forKey: Key.readbacks)
        defaults.set(frameStats, forKey: Key.frameStats)
        defaults.set(gpuProfile, forKey: Key.gpuProfile)
        defaults.set(vkValidation, forKey: Key.vkValidation)
        defaults.set(extraEnv, forKey: Key.extraEnv)
    }
}

/// One effect switch as edited in the form.
struct EffectSetting: Identifiable, Equatable {
    let key: String
    let title: String
    let note: String?
    var isOn: Bool
    var id: String { key }
}

/// The bbport.ini keys the launcher edits (the Linux launcher's INI_DEFAULTS); other keys stay.
struct GameSettings: Equatable {
    var upscaler = Catalog.defaultUpscaler
    var preset = Catalog.defaultPreset
    var sharpen = true
    var sharpness = 0.5
    var objectMotion = true
    var showFPS = true
    var outputRes = "1920x1080"
    var modelLOD = "0"
    var liveResolution = "0"
    var effects: [EffectSetting] = Catalog.effects.map {
        EffectSetting(key: $0.key, title: $0.title, note: $0.note, isOn: $0.defaultOn)
    }
    /// An upscaler from the ini that macOS does not offer (FSR 4 set on Linux or in the menu).
    var replacedUpscaler: String?

    init() {}

    init(ini values: [String: String]) {
        if let raw = values["upscaler"] {
            if Catalog.contains(Catalog.upscalers, raw) {
                upscaler = raw
            } else if raw == "fsr4" || raw == "fsr411" {
                // What the port itself does when FSR 4 is unsupported.
                upscaler = "fsr3"
                replacedUpscaler = raw
            }
        }
        if let raw = values["preset"], let value = Int(raw), Catalog.contains(Catalog.presets, value) {
            preset = value
        }
        sharpen = GameSettings.flag(values["sharpen"], sharpen)
        if let raw = values["sharpness"], let value = Double(raw) {
            sharpness = min(max(value, 0), 2)
        }
        objectMotion = GameSettings.flag(values["object_motion"], objectMotion)
        showFPS = GameSettings.flag(values["show_fps"], showFPS)
        if let raw = values["output_res"], Catalog.contains(Catalog.outputResolutions, raw) {
            outputRes = raw
        }
        if let raw = values["model_lod"], Catalog.contains(Catalog.modelDetail, raw) {
            modelLOD = raw
        }
        if let raw = values["live_resolution"], Catalog.contains(Catalog.liveResolution, raw) {
            liveResolution = raw
        }
        for index in effects.indices {
            effects[index].isOn = GameSettings.flag(values[effects[index].key], effects[index].isOn)
        }
    }

    private static func flag(_ value: String?, _ fallback: Bool) -> Bool {
        guard let value else { return fallback }
        return value == "1"
    }

    /// The edited keys in the Linux launcher's order (missing ones are appended in this order).
    var iniValues: [(key: String, value: String)] {
        var out: [(key: String, value: String)] = [
            (key: "upscaler", value: upscaler),
            (key: "preset", value: String(preset)),
            (key: "sharpen", value: sharpen ? "1" : "0"),
            (key: "sharpness", value: String(format: "%.2f", sharpness)),
            (key: "object_motion", value: objectMotion ? "1" : "0"),
            (key: "show_fps", value: showFPS ? "1" : "0"),
            (key: "output_res", value: outputRes),
            (key: "model_lod", value: modelLOD),
            (key: "live_resolution", value: liveResolution),
        ]
        for effect in effects {
            out.append((key: effect.key, value: effect.isOn ? "1" : "0"))
        }
        return out
    }
}

/// bbport.ini as the Linux launcher reads and writes it: `key=value` lines, `#` comments; edited
/// keys are rewritten in place, missing ones appended, everything else kept.
struct IniFile {
    var lines: [String] = []
    var values: [String: String] = [:]

    static func load(_ url: URL) -> IniFile {
        var ini = IniFile()
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return ini }
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        ini.lines = lines.map { $0.hasSuffix("\r") ? String($0.dropLast()) : $0 }
        for line in ini.lines {
            guard let setting = IniFile.split(line) else { continue }
            ini.values[setting.key] = setting.value
        }
        return ini
    }

    /// (key, value) of a setting line, nil for comments and other lines.
    static func split(_ line: String) -> (key: String, value: String)? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.hasPrefix("#"), let eq = line.firstIndex(of: "=") else { return nil }
        let key = line[..<eq].trimmingCharacters(in: .whitespaces)
        let value = line[line.index(after: eq)...].trimmingCharacters(in: .whitespaces)
        return (key: key, value: value)
    }

    func text(applying edits: [(key: String, value: String)]) -> String {
        var pending = edits
        var out: [String] = []
        for line in lines {
            if let setting = IniFile.split(line), let index = pending.firstIndex(where: { $0.key == setting.key }) {
                out.append("\(setting.key)=\(pending[index].value)")
                // A key listed twice in the file gets the new value both times, as in Python.
                continue
            }
            out.append(line)
        }
        let written = Set(out.compactMap { IniFile.split($0)?.key })
        if lines.isEmpty {
            out.append("# bbport settings (in-game menu: F1, ` or § / L3+R3)")
        }
        pending.removeAll { written.contains($0.key) }
        for edit in pending {
            out.append("\(edit.key)=\(edit.value)")
        }
        return out.joined(separator: "\n") + "\n"
    }
}
