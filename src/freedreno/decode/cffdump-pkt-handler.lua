-- SPDX-License-Identifier: MIT

-- `r` is predefined in the environment and is the equivalent of rnn.init(<gpu>)

local SCRATCH = {}
for i = 1, 8 do
	SCRATCH[i] = 0
end

function CP_REG_RMW(pkt, size)
	local dst_reg		= pkt[0].DST_REG
	local dst_scratch	= pkt[0].DST_SCRATCH
	local rotate		= pkt[0].ROTATE
	local src1_add		= pkt[0].SRC1_ADD
	local src1_is_reg	= pkt[0].SRC1_IS_REG
	local src0_is_reg	= pkt[0].SRC0_IS_REG
	local src0 		= pkt[1].SRC0
	local src1 		= pkt[2].SRC1

	local dst = regs.val(dst_reg)
	local dst_reg_str = string.format("%s", rnn.regname(r, dst_reg))
	if dst_scratch then
		dst_reg_str = string.format("SCRATCH[%s]", dst_reg)
	end

	local src0_str = string.format("0x%08x", src0)
	if src0_is_reg then
		src0_str = string.format("%s", rnn.regname(r, src0))
		src0 = regs.val(src0)
	end

	local src1_str = string.format("0x%08x", src1)
	if src1_is_reg then
		src1_str = string.format("%s", rnn.regname(r, src1))
		src1 = regs.val(src1)
	end

	local result = dst & src0
	result = (result << rotate) | result >> (32 - rotate)

	local op_str
	if src1_add then
		op_str = '+'
		result = result + src1
	else
		op_str = '|'
		result = result | src1
	end

	result = (dst &~ 0xFFFFFFFF) | result & 0xFFFFFFFF

	if dst_scratch then
		SCRATCH[dst_reg + 1] = result
	else
		priv.reg_set(dst_reg, result)
	end

	return string.format("%s = ((%s & %s) rot_l %d) %s %s\n",
		             dst_reg_str, dst_reg_str, src0_str, rotate,
		             op_str, src1_str)
end

function CP_MEM_WRITE(pkt, size)
	local addr = pkt.ADDR

	for i = 2, size - 1 do
		dbg("write: %x %x\n", addr, pkt[i])
		bos.write(addr, pkt[i])
		addr = addr + 4
	end
end

function CP_REG_TO_MEM(pkt, size)
	local reg = pkt.REG
	local cnt = pkt.CNT
	local addr = pkt.DEST

	-- note: CNT in units of dwords even if IS_64B, so ignoring
	-- ACCUMULATE we can just do the simple thing of copying
	-- however many dwords
	if pkt.ACCUMULATE then
		io.stderr:write("WARNING: Write with ACCUMULATE is not emulated.")
	end

	for i = 0, cnt do
		dbg("val: %x\n", regs.val(reg))
		bos.write(addr, regs.val(reg))
		reg = reg + 1
		addr = addr + 4
	end
end

function CP_MEM_TO_REG(pkt, size)
	local reg = pkt.REG
	local cnt = pkt.CNT
	local addr = pkt.SRC

	for i = 0, cnt do
		local val = bos[addr]

		-- If the memory address is not available, there is not much
		-- we can do.  Just bail.
		if not val then
			dbg("address not available: %x\n", addr)
			return
		end

		dbg("val: %x\n", val)

		if pkt.SHIFT_BY_2 then
			val = val << 2
		end

		priv.reg_set(reg, val)

		reg = reg + 1
		addr = addr + 4
	end
end

function CP_SCRATCH_WRITE(pkt, size)
	local idx = pkt.SCRATCH + 1

	for i = 1, size - 1 do
		dbg("SCRATCH[%d] <- %x\n", idx, pkt[i])
		SCRATCH[idx] = pkt[i]
		idx = idx + 1
	end
end

function CP_REG_TO_SCRATCH(pkt, size)
	local reg = pkt.REG
	local idx = pkt.SCRATCH + 1
	local cnt = pkt.CNT
	for i = 0, cnt do
		SCRATCH[idx + i] = regs.val(reg + i)
	end
end

function CP_SCRATCH_TO_REG(pkt, size)
	local reg = pkt.REG
	local idx = pkt.SCRATCH + 1
	local cnt = pkt.CNT
	for i = 0, cnt do
		priv.reg_set(reg + i, SCRATCH[idx + i])
	end
end
