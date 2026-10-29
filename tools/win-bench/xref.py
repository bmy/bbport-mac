# RIP-relative references (lea/mov with a disp32 to the target) in the eboot image: who uses an
# address. usage: python xref.py eboot.elf 0xTARGET [0xTARGET ...]
import sys

data = open(sys.argv[1], 'rb').read()
targets = {int(t, 16) for t in sys.argv[2:]}
text_end = 0x4600000  # code below the read-only data
hits = []
# Any instruction byte followed by a modrm with mod=00 rm=101 (RIP-relative), disp32 after it:
# check every position i as the modrm byte; the next instruction starts at i+5 (no immediate).
for i in range(1, min(len(data), text_end) - 5):
    if data[i] & 0xC7 != 0x05:
        continue
    disp = int.from_bytes(data[i + 1:i + 5], 'little', signed=True)
    for extra in (0, 1, 4):  # no immediate, imm8, imm32 after the operand
        target = i + 5 + extra + disp
        if target in targets:
            hits.append((i, target, extra))
for i, target, extra in hits:
    print(f'modrm at {i:#x} -> {target:#x} (imm {extra})')
