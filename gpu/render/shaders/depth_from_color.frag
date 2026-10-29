// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: writes a multisampled depth image from a color image that holds one sample per
// channel (a game's depth resolve read back as color): sample N takes channel N.
#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D samples;
layout(location = 0) in vec2 uv;

void main() {
    const ivec2 texel = ivec2(uv * vec2(textureSize(samples, 0)));
    gl_FragDepth = texelFetch(samples, texel, 0)[gl_SampleID];
}
