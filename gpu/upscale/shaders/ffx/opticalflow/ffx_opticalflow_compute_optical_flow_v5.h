// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FFX_OPTICALFLOW_COMPUTE_OPTICAL_FLOW_V5_H
#define FFX_OPTICALFLOW_COMPUTE_OPTICAL_FLOW_V5_H

#define CompareSize (4 * 2)
#define BlockSizeY 8
#define BlockSizeX 8
#define ThreadCount (4 * 16)
#define SearchRadiusX (8)
#define SearchRadiusY (8)
#define BlockCount 2

#define SearchBufferSizeX ((CompareSize + SearchRadiusX*2)/4)
#define SearchBufferSizeY  (CompareSize + SearchRadiusY*2)

FFX_GROUPSHARED FfxUInt32 pixels[CompareSize][CompareSize / 4];
#if BB_OF_PACKED_SAD
// bbport: the search window's words already split into even bytes (0, 2) and odd bytes (1, 3),
// as packed 16-bit lanes (CalculateQSads2): split once per word here, not by every thread.
FFX_GROUPSHARED FfxUInt32 searchEven[SearchBufferSizeY * SearchBufferSizeX];
FFX_GROUPSHARED FfxUInt32 searchOdd[SearchBufferSizeY * SearchBufferSizeX];
#else
FFX_GROUPSHARED FfxUInt32 searchBuffer[1][SearchBufferSizeY * SearchBufferSizeX];
#endif
#define bankBreaker 1
FFX_GROUPSHARED FfxUInt32 sadMapBuffer[4][SearchRadiusY * 2][(SearchRadiusX * 2) / 4 + bankBreaker];

#define MaxWaves 2
FFX_GROUPSHARED FfxUInt32 sWaveSad[MaxWaves];
FFX_GROUPSHARED FfxUInt32 sWaveMin[MaxWaves];

FfxUInt32 BlockSad64(FfxUInt32 blockSadSum, FfxInt32 iLocalIndex, FfxInt32 iLaneToBlockId, FfxInt32 block)
{
    if (iLaneToBlockId != block)
    {
        blockSadSum = 0u;
    }
    blockSadSum = ffxWaveSum(blockSadSum);

    if (ffxWaveLaneCount() == 32)
    {
        FfxInt32 waveId = iLocalIndex >> 5u;
        if (ffxWaveIsFirstLane())
        {
            sWaveSad[waveId] = blockSadSum;
        }
        FFX_GROUP_MEMORY_BARRIER;
        blockSadSum += sWaveSad[waveId ^ 1];
    }

    return blockSadSum;
}

FfxUInt32 SadMapMinReduction256(FfxInt32x2 iSearchId, FfxInt32 iLocalIndex)
{
    FfxUInt32 min01 = ffxMin(sadMapBuffer[0][iSearchId.y][iSearchId.x], sadMapBuffer[1][iSearchId.y][iSearchId.x]);
    FfxUInt32 min23 = ffxMin(sadMapBuffer[2][iSearchId.y][iSearchId.x], sadMapBuffer[3][iSearchId.y][iSearchId.x]);
    FfxUInt32 min0123 = ffxMin(min01, min23);
    min0123 = ffxWaveMin(min0123);

    if (ffxWaveLaneCount() == 32)
    {
        FfxInt32 waveId = iLocalIndex >> 5u;

        if (ffxWaveIsFirstLane())
        {
            sWaveMin[waveId] = min0123;
        }
        FFX_GROUP_MEMORY_BARRIER;
        min0123 = ffxMin(min0123, sWaveMin[waveId ^ 1]);
    }

    return min0123;
}

void LoadSearchBuffer(FfxInt32 iLocalIndex, FfxInt32x2 iPxPosShifted)
{
    FfxInt32 baseX = (iPxPosShifted.x - SearchRadiusX);
    FfxInt32 baseY = (iPxPosShifted.y - SearchRadiusY);

    for (FfxInt32 id = iLocalIndex; id < SearchBufferSizeX * SearchBufferSizeY; id += ThreadCount)
    {
        FfxInt32 idx = id % SearchBufferSizeX;
        FfxInt32 idy = id / SearchBufferSizeX;
        FfxInt32 x = baseX + idx * 4;
        FfxInt32 y = baseY + idy;
#if BB_OF_PACKED_SAD
        const FfxUInt32 w = LoadSecondImagePackedLuma(FfxInt32x2(x, y));
        searchEven[id] = w & 0x00ff00ffu;
        searchOdd[id] = (w >> 8) & 0x00ff00ffu;
#else
        searchBuffer[0][id] = LoadSecondImagePackedLuma(FfxInt32x2(x, y));
#endif
    }
    FFX_GROUP_MEMORY_BARRIER;
}

#if BB_OF_PACKED_SAD
FfxUInt32 BbLaneSum(u16vec2 v)
{
    return FfxUInt32(v.x) + FfxUInt32(v.y);
}
// Adds max(a, b) over the 4 byte offsets of the 8-byte window (a0, a1) against b to mx0..mx3,
// from the even/odd byte halves of a0 (L0, H0) and a1 (L1, H1), and b's bytes to sb.
#define BB_OF_QMAX(L0, H0, L1, H1, b_word)                                                    \
    {                                                                                         \
        FfxUInt32 bw = (b_word);                                                              \
        u16vec2 bL = unpack16(bw & 0x00ff00ffu), bH = unpack16((bw >> 8) & 0x00ff00ffu);      \
        u16vec2 l0 = unpack16(L0), h0 = unpack16(H0), l1 = unpack16(L1), h1 = unpack16(H1);  \
        u16vec2 mL = unpack16((L0 >> 16) | (L1 << 16)), mH = unpack16((H0 >> 16) | (H1 << 16)); \
        mx0 += max(l0, bL) + max(h0, bH);                                                     \
        mx1 += max(h0, bL) + max(mL, bH);                                                     \
        mx2 += max(mL, bL) + max(mH, bH);                                                     \
        mx3 += max(mH, bL) + max(l1, bH);                                                     \
        sb += bL + bH;                                                                        \
    }
#endif

FfxUInt32x4 CalculateQSads2(FfxInt32x2 iSearchId)
{
    FfxUInt32x4 sad = ffxBroadcast4(0u);

#if FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION == 1

    FfxInt32 idx = iSearchId.y * 6 + iSearchId.x;

    sad = msad4(pixels[0][0], FfxUInt32x2(searchBuffer[0][idx],     searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[0][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[1][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[1][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[2][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[2][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[3][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[3][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[4][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[4][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[5][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[5][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[6][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[6][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[7][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[7][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);

#elif BB_OF_PACKED_SAD
    // bbport: the same sums as QSad with fewer operations. |a - b| = 2 max(a, b) - a - b, and the
    // sums of a (per window) and b (per block) are linear, so they are taken once per row as
    // whole words rather than per offset. Bytes go in packed 16-bit lanes: each 32-bit word
    // split into its even bytes (0, 2) and odd bytes (1, 3); QSad's 4 byte offsets are then
    // lane-aligned words of those halves. Every lane sum stays below 2^16 (16 words of 2 bytes
    // at 255 each), and the result is bit-identical.
    u16vec2 mx0 = u16vec2(0us), mx1 = u16vec2(0us), mx2 = u16vec2(0us), mx3 = u16vec2(0us), sb = u16vec2(0us);
    u16vec2 sL0 = u16vec2(0us), sH0 = u16vec2(0us), sL1 = u16vec2(0us), sH1 = u16vec2(0us), sL2 = u16vec2(0us), sH2 = u16vec2(0us);
    FfxInt32 rowOffset = iSearchId.y * SearchBufferSizeX + iSearchId.x;
    for (FfxInt32 dy = 0; dy < CompareSize; dy++, rowOffset += SearchBufferSizeX)
    {
        FfxUInt32 L0 = searchEven[rowOffset], L1 = searchEven[rowOffset + 1], L2 = searchEven[rowOffset + 2];
        FfxUInt32 H0 = searchOdd[rowOffset], H1 = searchOdd[rowOffset + 1], H2 = searchOdd[rowOffset + 2];
        sL0 += unpack16(L0); sH0 += unpack16(H0);
        sL1 += unpack16(L1); sH1 += unpack16(H1);
        sL2 += unpack16(L2); sH2 += unpack16(H2);
        BB_OF_QMAX(L0, H0, L1, H1, pixels[dy][0]);
        BB_OF_QMAX(L1, H1, L2, H2, pixels[dy][1]);
    }
    // Window sums per offset: an aligned word (X >> 16) | (Y << 16) has the lanes (X.y, Y.x).
    const FfxUInt32x2 l0 = FfxUInt32x2(sL0), h0 = FfxUInt32x2(sH0), l1 = FfxUInt32x2(sL1), h1 = FfxUInt32x2(sH1);
    const FfxUInt32x2 l2 = FfxUInt32x2(sL2), h2 = FfxUInt32x2(sH2);
    const FfxUInt32 mL01 = l0.y + l1.x, mH01 = h0.y + h1.x, mL12 = l1.y + l2.x, mH12 = h1.y + h2.x;
    const FfxUInt32x4 windowSum = FfxUInt32x4(l0.x + l0.y + h0.x + h0.y + l1.x + l1.y + h1.x + h1.y,
                                              h0.x + h0.y + mL01 + h1.x + h1.y + mL12,
                                              mL01 + mH01 + mL12 + mH12,
                                              mH01 + l1.x + l1.y + mH12 + l2.x + l2.y);
    const FfxUInt32 blockSum = BbLaneSum(sb);
    sad = 2u * FfxUInt32x4(BbLaneSum(mx0), BbLaneSum(mx1), BbLaneSum(mx2), BbLaneSum(mx3)) - windowSum - blockSum;
#else
    for (FfxInt32 dy = 0; dy < CompareSize; dy++)
    {
        FfxInt32 rowOffset = (iSearchId.y + dy) * SearchBufferSizeX;
        FfxUInt32 a0 = searchBuffer[0][rowOffset + iSearchId.x];
        FfxUInt32 a1 = searchBuffer[0][rowOffset + iSearchId.x + 1];
        FfxUInt32 a2 = searchBuffer[0][rowOffset + iSearchId.x + 2];
        sad += QSad(a0, a1, pixels[dy][0]);
        sad += QSad(a1, a2, pixels[dy][1]);
    }
#endif

    return sad;
}

FfxUInt32x2 abs_2(FfxInt32x2 val)
{
    FfxInt32x2 tmp = val;
    FfxInt32x2 mask = tmp >> 31;
    FfxUInt32x2 res = (tmp + mask) ^ mask;
    return res;
}

FfxUInt32 EncodeSearchCoord(FfxInt32x2 coord)
{
#if FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS == 1
    FfxUInt32x2 absCoord = FfxUInt32x2(abs_2(coord - 8));
    return FfxUInt32(absCoord.y << 12) | FfxUInt32(absCoord.x << 8) | FfxUInt32(coord.y << 4) | FfxUInt32(coord.x);
#else //FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS == 1
    return FfxUInt32(coord.y << 8) | FfxUInt32(coord.x);
#endif //FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS == 1
}

FfxInt32x2 DecodeSearchCoord(FfxUInt32 bits)
{
#if FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS == 1
    FfxInt32 dx = FfxInt32(bits & 0xfu) - SearchRadiusX;
    FfxInt32 dy = FfxInt32((bits >> 4) & 0xfu) - SearchRadiusY;

    return FfxInt32x2(dx, dy);
#else
    FfxInt32 dx = FfxInt32(bits & 0xffu) - SearchRadiusX;
    FfxInt32 dy = FfxInt32((bits >> 8) & 0xffu) - SearchRadiusY;

    return FfxInt32x2(dx, dy);
#endif
}

void PrepareSadMap(FfxInt32x2 iSearchId, FfxUInt32x4 qsad)
{
    sadMapBuffer[0][iSearchId.y][iSearchId.x] = (qsad.x << 16) | EncodeSearchCoord(FfxInt32x2(iSearchId.x * 4 + 0, iSearchId.y));
    sadMapBuffer[1][iSearchId.y][iSearchId.x] = (qsad.y << 16) | EncodeSearchCoord(FfxInt32x2(iSearchId.x * 4 + 1, iSearchId.y));
    sadMapBuffer[2][iSearchId.y][iSearchId.x] = (qsad.z << 16) | EncodeSearchCoord(FfxInt32x2(iSearchId.x * 4 + 2, iSearchId.y));
    sadMapBuffer[3][iSearchId.y][iSearchId.x] = (qsad.w << 16) | EncodeSearchCoord(FfxInt32x2(iSearchId.x * 4 + 3, iSearchId.y));
    FFX_GROUP_MEMORY_BARRIER;
}


uint ABfe(uint src, uint off, uint bits) { uint mask = (1u << bits) - 1u; return (src >> off) & mask; }
uint ABfi(uint src, uint ins, uint mask) { return (ins & mask) | (src & (~mask)); }
uint ABfiM(uint src, uint ins, uint bits) { uint mask = (1u << bits) - 1u; return (ins & mask) | (src & (~mask)); }
void MapThreads(in FfxInt32x2 iGroupId, in FfxInt32 iLocalIndex,
                out FfxInt32x2 iSearchId, out FfxInt32x2 iPxPos, out FfxInt32 iLaneToBlockId)
{
    iSearchId = FfxInt32x2(ABfe(iLocalIndex, 0u, 2u), ABfe(iLocalIndex, 2u, 4u));
    iLaneToBlockId = FfxInt32(ABfe(iLocalIndex, 1u, 1u) | (ABfe(iLocalIndex, 5u, 1u) << 1u));
    iPxPos = (iGroupId << 4u) + iSearchId * FfxInt32x2(4, 1);
}

void ComputeOpticalFlowAdvanced(FfxInt32x2 iGlobalId, FfxInt32x2 iLocalId, FfxInt32x2 iGroupId, FfxInt32 iLocalIndex)
{
    FfxInt32x2 iSearchId;
    FfxInt32x2 iPxPos;
    FfxInt32 iLaneToBlockId;
    MapThreads(iGroupId, iLocalIndex, iSearchId, iPxPos, iLaneToBlockId);

    FfxInt32x2 currentOFPos = iPxPos >> 3u;

    if (IsSceneChanged())
    {
        if ((iSearchId.y & 0x7) == 0 && (iSearchId.x & 0x1) == 0)
        {
            StoreOpticalFlow(currentOFPos, FfxInt32x2(0, 0));
        }

        return;
    }

    const FfxBoolean bUsePredictionFromPreviousLevel = (OpticalFlowPyramidLevel() != OpticalFlowPyramidLevelCount() - 1);

#if BB_OF_LEVEL0_PREDICT
    // bbport (BB_OF_FAST): the finest level is not searched. Its vectors stay the coarser level's,
    // scaled up into this level's buffer by the scale pass before this one: a 16-pixel block's
    // motion at 2-pixel steps instead of an 8-pixel block's, for most of the flow's cost.
    if (OpticalFlowPyramidLevel() == 0 && bUsePredictionFromPreviousLevel)
    {
        return;
    }
#endif

    FfxUInt32 packedLuma_4blocks = LoadFirstImagePackedLuma(iPxPos);

#if FFX_LOCAL_SEARCH_FALLBACK == 1
    FfxUInt32 prevPackedLuma_4blocks = LoadSecondImagePackedLuma(iPxPos);
    FfxUInt32 sad_4blocks = Sad(packedLuma_4blocks, prevPackedLuma_4blocks);
#endif //FFX_LOCAL_SEARCH_FALLBACK

    FfxInt32x2 ofGroupOffset = iGroupId << 1u;
    FfxInt32x2 pixelGroupOffset = iGroupId << 4u;

    FfxInt32x2 blockId;
    for (blockId.y = 0; blockId.y < BlockCount; blockId.y++)
    {
        for (blockId.x = 0; blockId.x < BlockCount; blockId.x++)
        {
            FfxInt32x2 currentVector = LoadRwOpticalFlow(ofGroupOffset + blockId);
            if (!bUsePredictionFromPreviousLevel)
            {
                currentVector = FfxInt32x2(0, 0);
            }

            if (iLaneToBlockId == blockId.y * 2 + blockId.x)
            {
                pixels[iSearchId.y & 0x7][iSearchId.x & 0x1] = packedLuma_4blocks;
            }

            LoadSearchBuffer(iLocalIndex, pixelGroupOffset + blockId * 8 + currentVector);

            FfxUInt32x4 qsad = CalculateQSads2(iSearchId);

            PrepareSadMap(iSearchId, qsad);
            FfxUInt32 minSad = SadMapMinReduction256(iSearchId, iLocalIndex);

            FfxInt32x2 minSadCoord = DecodeSearchCoord(minSad);
            FfxInt32x2 newVector = currentVector + minSadCoord;

#if FFX_LOCAL_SEARCH_FALLBACK == 1
            FfxUInt32 blockSadSum = BlockSad64(sad_4blocks, iLocalIndex, iLaneToBlockId, blockId.x + blockId.y * 2);
            if (OpticalFlowPyramidLevel() == 0 && blockSadSum <= (minSad >> 16u))
            {
                newVector = FfxInt32x2(0, 0);
            }
#endif //FFX_LOCAL_SEARCH_FALLBACK

            {
                StoreOpticalFlow(ofGroupOffset + blockId, newVector);
            }
        }
    }
}

#endif // FFX_OPTICALFLOW_COMPUTE_OPTICAL_FLOW_V5_H
