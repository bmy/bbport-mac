# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Groups sampler.exe output by function (and by caller chain-free inline frame).

    python symbolize.py SAMPLES.txt EXE [TOP]

EXE is the module the samples name (with its debug info); other modules are summed by name.
Needs llvm-symbolizer on PATH (MSYS2 CLANG64)."""
import collections
import struct
import subprocess
import sys


def image_base(path):
    with open(path, "rb") as f:
        data = f.read(4096)
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    magic = struct.unpack_from("<H", data, pe + 24)[0]
    return struct.unpack_from("<Q", data, pe + 24 + 24)[0] if magic == 0x20B else \
        struct.unpack_from("<I", data, pe + 24 + 28)[0]


def main():
    samples_path, exe = sys.argv[1], sys.argv[2]
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    exe_name = exe.replace("/", "\\").split("\\")[-1].lower()
    base = image_base(exe)
    own = collections.Counter()
    other = collections.Counter()
    callers = collections.Counter()  # (module, caller offset) of samples outside the exe
    total = 0
    for line in open(samples_path, encoding="utf-8", errors="replace"):
        if not line.startswith("sample "):
            continue
        parts = line.split()
        where = parts[2]
        count = int(parts[4] if parts[3].startswith("caller+") else parts[3])
        total += count
        module, _, offset = where.rpartition("+")
        if module.lower() == exe_name:
            own[int(offset, 16)] += count
        else:
            other[module] += count
            if parts[3].startswith("caller+"):
                callers[(module, int(parts[3][7:], 16))] += count
    addresses = list(own) + [a for _, a in callers]
    query = "\n".join(hex(base + a) for a in addresses) + "\n"
    out = subprocess.run(["llvm-symbolizer", "--obj=" + exe, "--functions=linkage", "--demangle",
                          "--no-inlines"], input=query, capture_output=True, text=True).stdout
    blocks = out.strip().split("\n\n")
    names = {}
    for address, block in zip(addresses, blocks):
        lines = block.splitlines()
        names[address] = (lines[0] if lines else "?",
                          (lines[1] if len(lines) > 1 else "?").replace("\\", "/").split("/")[-1])
    by_caller = collections.Counter()
    for (module, address), count in callers.items():
        function, location = names[address]
        by_caller[(module, function, location)] += count
    blocks = blocks[:len(own)]
    by_function = collections.Counter()
    by_line = collections.Counter()
    for address, block in zip(addresses, blocks):
        lines = block.splitlines()
        function = lines[0] if lines else "?"
        location = lines[1] if len(lines) > 1 else "?"
        by_function[function] += own[address]
        by_line[(function, location.split("\\")[-1].split("/")[-1])] += own[address]
    print(f"{total} samples; {sum(own.values())} in {exe_name}")
    for module, count in other.most_common(15):
        print(f"{100 * count / total:6.2f}%  [{module}]")
    print("-- functions")
    for function, count in by_function.most_common(top):
        print(f"{100 * count / total:6.2f}%  {function[:150]}")
    print("-- lines")
    for (function, location), count in by_line.most_common(top):
        print(f"{100 * count / total:6.2f}%  {location:40s} {function[:90]}")
    print("-- other modules by caller (first return address into the exe)")
    for (module, function, location), count in by_caller.most_common(top):
        print(f"{100 * count / total:6.2f}%  {module:16s} {location:36s} {function[:90]}")


if __name__ == "__main__":
    main()
