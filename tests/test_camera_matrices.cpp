// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// upscale/camera_matrices.h: the clip-space matrices DLSS Frame Generation gets from the game's
// cameras. No GPU: a point is taken through the matrices and compared with the direct way.
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "gpu/upscale/camera_matrices.h"

using namespace Vulkan::CameraMatrices;

namespace {

std::array<float, 4> Apply(const std::array<float, 4>& v, const Matrix4& m) {
    std::array<float, 4> out{};
    for (int c = 0; c < 4; ++c) {
        for (int k = 0; k < 4; ++k) {
            out[c] += v[k] * m[k * 4 + c];
        }
    }
    return out;
}

std::array<float, 3> ApplyAffine(const Affine& m, const std::array<float, 3>& p) {
    return {m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
            m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]};
}

/// A camera at `eye` turned by `yaw` about y (view looks along +z), as the game's 3x4 matrices.
void Camera(const std::array<float, 3>& eye, float yaw, Affine& view, Affine& inv_view) {
    const float c = std::cos(yaw), s = std::sin(yaw);
    // World to view: R^T (p - eye), with R the camera's rotation.
    const std::array<float, 9> r = {c, 0, s, 0, 1, 0, -s, 0, c}; // columns: right, up, forward
    // R^T rows are R's columns.
    const std::array<std::array<float, 3>, 3> rt = {{{r[0], r[3], r[6]}, {r[1], r[4], r[7]},
                                                      {r[2], r[5], r[8]}}};
    for (int row = 0; row < 3; ++row) {
        const float t = -(rt[row][0] * eye[0] + rt[row][1] * eye[1] + rt[row][2] * eye[2]);
        view[row * 4 + 0] = rt[row][0];
        view[row * 4 + 1] = rt[row][1];
        view[row * 4 + 2] = rt[row][2];
        view[row * 4 + 3] = t;
        inv_view[row * 4 + 0] = r[row * 3 + 0];
        inv_view[row * 4 + 1] = r[row * 3 + 1];
        inv_view[row * 4 + 2] = r[row * 3 + 2];
        inv_view[row * 4 + 3] = eye[row];
    }
}

bool Near(float a, float b, float eps = 1e-3f) {
    return std::abs(a - b) <= eps * std::max(1.0f, std::max(std::abs(a), std::abs(b)));
}

} // namespace

int main() {
    // The game's projection: 16:9, vertical fov ~ 60 degrees, near 0.1, infinite far (zs = 1,
    // depth = 1 - near / z), y scale negative (ndc +y down for the motion shaders).
    const float ys = 1.0f / std::tan(0.5236f);
    const Projection proj{ys / (16.0f / 9.0f), -ys, 1.0f, -0.1f};
    const Projection prev_proj{ys / (16.0f / 9.0f), -ys, 1.0f, -0.1f};
    Affine view{}, inv_view{}, prev_view{}, prev_inv_view{};
    Camera({1.0f, 2.0f, -3.0f}, 0.3f, view, inv_view);
    Camera({0.5f, 2.0f, -3.5f}, 0.25f, prev_view, prev_inv_view);

    Matrix4 view_to_clip{}, clip_to_prev{}, prev_to_clip{};
    assert(ClipTransforms(view, inv_view, proj, prev_view, prev_inv_view, prev_proj, view_to_clip,
                          clip_to_prev, prev_to_clip));

    // The projection: a view point maps to D3D clip space (+y up: the game's negative y scale
    // is negated), w = view z, depth 0 at the near plane.
    {
        const auto clip = Apply({1.0f, 0.5f, 4.0f, 1.0f}, view_to_clip);
        assert(Near(clip[3], 4.0f));
        assert(Near(clip[0], proj[0] * 1.0f));
        assert(Near(clip[1], ys * 0.5f)); // positive: up
        assert(Near(clip[2] / clip[3], 1.0f - 0.1f / 4.0f));
        const auto near = Apply({0.0f, 0.0f, 0.1f, 1.0f}, view_to_clip);
        assert(Near(near[2] / near[3], 0.0f));
    }
    // The inverse projection undoes it (homogeneous).
    {
        const auto clip = Apply({-0.7f, 0.3f, 7.0f, 1.0f}, view_to_clip);
        const auto back = Apply(clip, InverseProjectionMatrix(proj));
        assert(Near(back[0] / back[3], -0.7f) && Near(back[1] / back[3], 0.3f) &&
               Near(back[2] / back[3], 7.0f));
    }
    // A world point through the current clip space to the previous one lands where the
    // previous camera's projection puts it directly, and comes back.
    {
        const std::array<float, 3> world{2.0f, 1.5f, 6.0f};
        const auto cur = ApplyAffine(view, world);
        const auto prev = ApplyAffine(prev_view, world);
        const auto cur_clip = Apply({cur[0], cur[1], cur[2], 1.0f}, view_to_clip);
        const auto prev_clip = Apply(cur_clip, clip_to_prev);
        const auto expected = Apply({prev[0], prev[1], prev[2], 1.0f}, ProjectionMatrix(prev_proj));
        for (int i = 0; i < 3; ++i) {
            assert(Near(prev_clip[i] / prev_clip[3], expected[i] / expected[3]));
        }
        const auto again = Apply(prev_clip, prev_to_clip);
        for (int i = 0; i < 3; ++i) {
            assert(Near(again[i] / again[3], cur_clip[i] / cur_clip[3]));
        }
    }
    // A camera that did not move: identities (up to rounding).
    {
        Matrix4 a{}, b{}, c{};
        assert(ClipTransforms(view, inv_view, proj, view, inv_view, proj, a, b, c));
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 4; ++col) {
                assert(Near(b[r * 4 + col], r == col ? 1.0f : 0.0f, 1e-4f));
                assert(Near(c[r * 4 + col], r == col ? 1.0f : 0.0f, 1e-4f));
            }
        }
    }
    // No camera yet: refused.
    {
        Matrix4 a{}, b{}, c{};
        assert(!ClipTransforms(view, inv_view, {}, prev_view, prev_inv_view, prev_proj, a, b, c));
    }
    std::puts("PASS: camera clip-space matrices for DLSS Frame Generation");
}
