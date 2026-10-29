# The community's "Uncap FPS++" constants (Bloodborne.xml, addresses +0x400000) the port does not
# redirect (bbport_frame_rate.cpp Redirects): their original values and the code reading them
# RIP-relative. usage: python patchconsts.py eboot.elf Bloodborne.xml bbport_frame_rate.cpp
import re
import struct
import sys

data = open(sys.argv[1], 'rb').read()
xml = open(sys.argv[2], encoding='utf-8').read()
src = open(sys.argv[3], encoding='utf-8').read()
block = xml[xml.index('Name="Uncap FPS++"'):]
block = block[:block.index('</PatchList>')]
patched = {}
for addr, value in re.findall(r'Address="0x([0-9A-Fa-f]+)" Value="([0-9A-Fa-f]{8})"', block):
    a = int(addr, 16) - 0x400000
    if a >= 0x4600000:  # data, not code
        patched[a] = struct.unpack('<f', bytes.fromhex(value))[0]
used = {int(c, 16) for c in re.findall(r'\{0x[0-9a-f]+, (0x[0-9a-f]+), LIVE', src)}
missing = sorted(a for a in patched if a not in used)
refs = {a: [] for a in missing}
for i in range(1, 0x4600000 - 5):
    if data[i] & 0xC7 != 0x05:
        continue
    disp = int.from_bytes(data[i + 1:i + 5], 'little', signed=True)
    for extra in (0, 1, 4):
        t = i + 5 + extra + disp
        if t in refs:
            refs[t].append(i)
for a in missing:
    original = struct.unpack('<f', data[a:a + 4])[0]
    sites = ' '.join(f'{s:#x}' for s in refs[a][:8])
    print(f'{a:#x}: original {original:.6g} -> patch {patched[a]:.6g}; read near {sites}')
