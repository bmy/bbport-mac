// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: one triangle covering the viewport (vertices 0..2, no buffers), with texture
// coordinates 0..1 over the visible part.
#version 450

layout(location = 0) out vec2 uv;

void main() {
    // (0,0), (2,0), (0,2) in texture space: the corner beyond the screen is clipped.
    uv = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1) * 2.0;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
