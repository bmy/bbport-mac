// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game's cameras as the 4x4 clip-space matrices DLSS Frame Generation takes
// (row-major, row vectors: clip = view * m; D3D clip space, +x right, +y up, depth 0 at the near
// plane, w = view z). Header only, without Vulkan: tests/test_camera_matrices.cpp checks it.
//
// Bloodborne's camera (CameraMotion): a 3x4 world-to-view matrix (rows [R | t], column
// vectors), its inverse, and a projection of four numbers {x scale, y scale, z scale, z offset}:
// clip = (xs x, ys y, zs z + zo, z). The y scale carries the sign of the G-buffer viewport so
// that the motion shaders' ndc has +y down the screen; D3D's clip space has +y up, so it is
// negated here.
#pragma once

#include <array>
#include <cmath>

namespace Vulkan::CameraMatrices {

using Matrix4 = std::array<float, 16>;
using Affine = std::array<float, 12>;
using Projection = std::array<float, 4>;

/// a * b for 3x4 affine matrices (rows [R | t], column vectors).
inline Affine MultiplyAffine(const Affine& a, const Affine& b) {
    Affine out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            float v = c == 3 ? a[r * 4 + 3] : 0.0f;
            for (int k = 0; k < 3; ++k) {
                v += a[r * 4 + k] * b[k * 4 + c];
            }
            out[r * 4 + c] = v;
        }
    }
    return out;
}

/// a * b for 4x4 row-major matrices.
inline Matrix4 Multiply(const Matrix4& a, const Matrix4& b) {
    Matrix4 out{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float v = 0.0f;
            for (int k = 0; k < 4; ++k) {
                v += a[r * 4 + k] * b[k * 4 + c];
            }
            out[r * 4 + c] = v;
        }
    }
    return out;
}

/// The game's projection as a D3D row-vector matrix: clip = (xs x, -ys y, zs z + zo, z).
inline Matrix4 ProjectionMatrix(const Projection& p) {
    return {p[0], 0, 0, 0, 0, -p[1], 0, 0, 0, 0, p[2], 1, 0, 0, p[3], 0};
}

/// Its inverse (view = clip * m, homogeneous).
inline Matrix4 InverseProjectionMatrix(const Projection& p) {
    return {1 / p[0], 0, 0, 0, 0, -1 / p[1], 0, 0, 0, 0, 0, 1 / p[3], 0, 0, 1, -p[2] / p[3]};
}

/// A 3x4 affine matrix (rows [R | t], column vectors) as a 4x4 row-vector matrix.
inline Matrix4 RowVectors(const Affine& a) {
    return {a[0], a[4], a[8], 0, a[1], a[5], a[9], 0, a[2], a[6], a[10], 0, a[3], a[7], a[11], 1};
}

inline bool Usable(const Projection& p) {
    return p[0] != 0.0f && p[1] != 0.0f && p[3] != 0.0f && std::isfinite(p[0]) &&
           std::isfinite(p[1]) && std::isfinite(p[2]) && std::isfinite(p[3]);
}

/// The projection of the current camera, and the transforms from its clip space to the previous
/// camera's and back (current clip -> current view -> world -> previous view -> previous clip).
/// False when either camera is unusable.
inline bool ClipTransforms(const Affine& view, const Affine& inv_view, const Projection& proj,
                           const Affine& prev_view, const Affine& prev_inv_view,
                           const Projection& prev_proj, Matrix4& view_to_clip,
                           Matrix4& clip_to_prev_clip, Matrix4& prev_clip_to_clip) {
    if (!Usable(proj) || !Usable(prev_proj)) {
        return false;
    }
    view_to_clip = ProjectionMatrix(proj);
    clip_to_prev_clip = Multiply(Multiply(InverseProjectionMatrix(proj),
                                          RowVectors(MultiplyAffine(prev_view, inv_view))),
                                 ProjectionMatrix(prev_proj));
    prev_clip_to_clip = Multiply(Multiply(InverseProjectionMatrix(prev_proj),
                                          RowVectors(MultiplyAffine(view, prev_inv_view))),
                                 ProjectionMatrix(proj));
    return true;
}

} // namespace Vulkan::CameraMatrices
