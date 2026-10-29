// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: copies the source texel under each fragment (MSAA_SOURCE: the same sample), for
// copies between images Vulkan cannot copy directly (sample counts, depth and color).
#version 450
#extension GL_EXT_samplerless_texture_functions : require

#ifdef MSAA_SOURCE
layout(set = 0, binding = 0) uniform texture2DMS source;
#else
layout(set = 0, binding = 0) uniform texture2D source;
#endif

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main() {
#ifdef MSAA_SOURCE
    color = texelFetch(source, ivec2(gl_FragCoord.xy), gl_SampleID);
#else
    color = texelFetch(source, ivec2(gl_FragCoord.xy), 0);
#endif
}
