#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""FSR 4.1.2 from AMD's driver DLL (amdxcffx64.dll, Adrenalin 26.10.1+), one model at a time.

    driver_model.py info <amdxcffx64.dll>
    driver_model.py force <amdxcffx64.dll> <out.dll> <fp8|int8|navi2|lite>

The driver DLL holds four FSR 4 providers: ffxProvider_FSR4 (FP8, RDNA4), ffxProvider_FSR4_Int8
(RDNA3), ffxProvider_FSR4_Int8_Navi2 (RDNA2) and ffxProvider_FSR4Lite_Int8 (the Steam Deck's and
RDNA 3.5 APUs). It takes the first, in that order, whose device check (the provider's third
virtual function: the GPU family and revision the driver reports) accepts the GPU. "force" writes
a copy in which that check answers yes for the wanted provider and no for the others, so that
fsr4cap.exe records that model on this GPU (the passes run on any GPU with INT8 dot products; the
FP8 one needs FP8 matrices, or vkd3d-proton's FP16 emulation of them). The copy is the user's own,
for the recording only; nothing of AMD's is shipped.

The providers are found by their RTTI names (type descriptor -> complete object locator ->
vtable), so other builds of the DLL work as long as the classes keep their names.
Exit status: 0; 2 not a driver DLL with these providers.
"""
import re
import struct
import sys

PROVIDERS = {
    "fp8": ".?AVffxProvider_FSR4@@",
    "int8": ".?AVffxProvider_FSR4_Int8@@",
    "navi2": ".?AVffxProvider_FSR4_Int8_Navi2@@",
    "lite": ".?AVffxProvider_FSR4Lite_Int8@@",
}
CHECK_SLOT = 2  # the vtable entry of the device check
YES = b"\xb0\x01\xc3"  # mov al, 1; ret
NO = b"\x30\xc0\xc3"   # xor al, al; ret


class Pe:
    def __init__(self, data):
        self.data = data
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            raise ValueError("not a PE file")
        sections = struct.unpack_from("<H", data, pe + 6)[0]
        optional = struct.unpack_from("<H", data, pe + 20)[0]
        self.base = struct.unpack_from("<Q", data, pe + 24 + 24)[0]
        self.sections = [struct.unpack_from("<IIII", data, pe + 24 + optional + 40 * i + 8)
                         for i in range(sections)]

    def rva(self, offset):
        for vsize, va, rsize, raw in self.sections:
            if raw <= offset < raw + rsize:
                return va + offset - raw
        return None

    def offset(self, va):
        rva = va - self.base
        for vsize, sva, rsize, raw in self.sections:
            if sva <= rva < sva + max(vsize, rsize):
                return raw + rva - sva
        return None

    def check(self, name):
        """File offset of a provider's device check, or None."""
        data = self.data
        at = data.find(name.encode() + b"\0")
        if at < 16:
            return None
        descriptor = self.rva(at - 16)  # TypeDescriptor: vftable pointer, spare, name
        for m in re.finditer(re.escape(struct.pack("<I", descriptor)), data):
            locator = m.start() - 12  # signature 1, offset, cd offset, type descriptor, ...
            if locator < 0 or struct.unpack_from("<I", data, locator)[0] != 1:
                continue
            if struct.unpack_from("<I", data, locator + 20)[0] != self.rva(locator):
                continue
            pointer = struct.pack("<Q", self.base + self.rva(locator))
            for v in re.finditer(re.escape(pointer), data):
                vtable = v.start() + 8
                target = struct.unpack_from("<Q", data, vtable + 8 * CHECK_SLOT)[0]
                code = self.offset(target)
                if code is not None:
                    return code
        return None


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("info", "force"):
        sys.exit(__doc__)
    data = bytearray(open(sys.argv[2], "rb").read())
    try:
        pe = Pe(bytes(data))
    except (ValueError, struct.error) as e:
        print(f"{sys.argv[2]}: {e}")
        sys.exit(2)
    checks = {model: pe.check(name) for model, name in PROVIDERS.items()}
    if sys.argv[1] == "info":
        for model, at in checks.items():
            print(f"{model}: {'found' if at is not None else 'missing'}")
        sys.exit(0 if all(at is not None for at in checks.values()) else 2)
    want = sys.argv[4]
    if want not in PROVIDERS or any(at is None for at in checks.values()):
        print(f"cannot force {want}: providers {checks}")
        sys.exit(2)
    for model, at in checks.items():
        data[at:at + 3] = YES if model == want else NO
    open(sys.argv[3], "wb").write(data)
    print(f"{sys.argv[3]}: FSR 4 model {want}")


if __name__ == "__main__":
    main()
