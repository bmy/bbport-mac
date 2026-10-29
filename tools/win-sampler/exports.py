# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Names sampler.exe samples in system DLLs by their nearest export (no PDBs needed).

    python exports.py SAMPLES.txt MODULE_NAME [TOP]"""
import bisect
import collections
import struct
import sys


def exports(path):
    data = open(path, "rb").read()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dirs = opt + (112 if magic == 0x20B else 96)
    exp_rva, _ = struct.unpack_from("<II", data, dirs)
    sec_table = opt + struct.unpack_from("<H", data, pe + 20)[0]
    sections = []
    for i in range(nsec):
        s = sec_table + 40 * i
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, s + 8)
        sections.append((va, max(vsize, rsize), raw))

    def off(rva):
        for va, size, raw in sections:
            if va <= rva < va + size:
                return raw + rva - va
        raise ValueError(hex(rva))

    e = off(exp_rva)
    nfuncs, nnames, funcs, names, ords = struct.unpack_from("<IIIII", data, e + 20)
    result = []
    for i in range(nnames):
        name_rva = struct.unpack_from("<I", data, off(names) + 4 * i)[0]
        ordinal = struct.unpack_from("<H", data, off(ords) + 2 * i)[0]
        func_rva = struct.unpack_from("<I", data, off(funcs) + 4 * ordinal)[0]
        name = data[off(name_rva):data.index(b"\0", off(name_rva))].decode()
        result.append((func_rva, name))
    result.sort()
    return result


def main():
    samples_path, module = sys.argv[1], sys.argv[2].lower()
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    path = None
    counts = collections.Counter()
    total = 0
    for line in open(samples_path, encoding="utf-8", errors="replace"):
        if line.startswith("module ") and line.rstrip().lower().endswith("\\" + module):
            path = line.split(" ", 2)[2].strip()
        if line.startswith("sample "):
            parts = line.split()
            where = parts[2]
            count = parts[4] if parts[3].startswith("caller+") else parts[3]
            total += int(count)
            name, _, offset = where.rpartition("+")
            if name.lower() == module:
                counts[int(offset, 16)] += int(count)
    table = exports(path)
    rvas = [r for r, _ in table]
    named = collections.Counter()
    for rva, count in counts.items():
        i = bisect.bisect_right(rvas, rva) - 1
        named[f"{table[i][1]}+{rva - table[i][0]:#x}" if i >= 0 else hex(rva)] += count
    by_export = collections.Counter()
    for key, count in named.items():
        by_export[key.split("+")[0]] += count
    print(f"{module}: {sum(counts.values())} of {total} samples ({path})")
    for name, count in by_export.most_common(top):
        print(f"{100 * count / total:6.2f}%  {name}")


if __name__ == "__main__":
    main()
