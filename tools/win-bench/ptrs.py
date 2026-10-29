# 8-byte pointers to the given code addresses anywhere in the image (vtables): file offsets.
# usage: python ptrs.py eboot.elf 0xADDR [0xADDR ...]
import sys

data = open(sys.argv[1], 'rb').read()
for t in sys.argv[2:]:
    needle = int(t, 16).to_bytes(8, 'little')
    at = data.find(needle)
    hits = []
    while at != -1 and len(hits) < 10:
        hits.append(at)
        at = data.find(needle, at + 1)
    print(t, ' '.join(f'{h:#x}' for h in hits))
