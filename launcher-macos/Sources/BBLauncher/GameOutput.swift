// SPDX-License-Identifier: GPL-2.0-or-later
// Reads the game's merged stdout/stderr off the main thread. The model drains complete lines on
// the main actor a few times a second, so a chatty log costs one view update per drain.
import Foundation

/// Bytes received from the pipe, shared between the reader thread and the main actor.
final class OutputCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var pending: [UInt8] = []
    private var ended = false

    func append(_ bytes: ArraySlice<UInt8>) {
        lock.lock()
        pending.append(contentsOf: bytes)
        lock.unlock()
    }

    func finish() {
        lock.lock()
        ended = true
        lock.unlock()
    }

    /// Complete lines received so far (and the unterminated rest once the pipe has closed).
    func drain() -> (lines: [String], ended: Bool) {
        lock.lock()
        let isEnded = ended
        var chunk: [UInt8] = []
        if isEnded {
            chunk = pending
            pending.removeAll()
        } else if let newline = pending.lastIndex(of: 0x0A) {
            chunk = Array(pending[...newline])
            pending.removeSubrange(...newline)
        }
        lock.unlock()
        guard !chunk.isEmpty else { return ([], isEnded) }
        // Split bytes, not Characters: "\r\n" is a single Character in Swift.
        var lines = chunk.split(separator: 0x0A, omittingEmptySubsequences: false).map { part -> String in
            var bytes = part
            if bytes.last == 0x0D { bytes.removeLast() }
            return cleanLogLine(String(decoding: bytes, as: UTF8.self))
        }
        if chunk.last == 0x0A { lines.removeLast() }
        return (lines, isEnded)
    }
}

/// Drops terminal colour codes (the port's fmt/spdlog style output).
private func cleanLogLine(_ line: String) -> String {
    guard line.contains("\u{1B}") else { return line }
    return line.replacingOccurrences(of: "\u{1B}\\[[0-9;?]*[ -/]*[@-~]", with: "", options: .regularExpression)
}

/// Reads `fd` until end of file on a background queue, then closes it. The descriptor is the
/// reader's own (dup), so the Pipe object can go away while a straggling child still writes.
func pumpOutput(from fd: Int32, into collector: OutputCollector) {
    DispatchQueue.global(qos: .utility).async {
        var buffer = [UInt8](repeating: 0, count: 65_536)
        while true {
            let count = buffer.withUnsafeMutableBytes { raw in
                read(fd, raw.baseAddress, raw.count)
            }
            if count > 0 {
                collector.append(buffer[0..<count])
            } else if count < 0 && errno == EINTR {
                continue
            } else {
                break
            }
        }
        close(fd)
        collector.finish()
    }
}
