# Float constants read RIP-relative by SSE/AVX instructions in an address range of the eboot:
# disassembly from llvm-objdump (stdin), values from the image. usage:
#   llvm-objdump -d --no-show-raw-insn --start-address=A --stop-address=B eboot.elf | python floats.py eboot.elf
import re
import struct
import sys

data = open(sys.argv[1], 'rb').read()
pattern = re.compile(r'^\s*([0-9a-f]+):\s+(v?(?:mul|add|sub|div|mov|max|min|comi|ucomi|cmp)\w*ss)\s+.*#\s+0x([0-9a-f]+)')
for line in sys.stdin:
    m = pattern.match(line)
    if not m:
        continue
    site, op, target = int(m.group(1), 16), m.group(2), int(m.group(3), 16)
    if target + 4 > len(data):
        continue
    value = struct.unpack('<f', data[target:target + 4])[0]
    print(f'{site:#x} {op:10s} [{target:#x}] = {value:.6g}')
