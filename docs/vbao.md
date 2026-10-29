# GT-VBAO in place of the game's SSAO

Setting `vbao` (launcher, or the overlay menu next to the game effects; applies at once). Needs
the game's SSAO on (`effect_ssao`): VBAO replaces the result of the game's AO passes, so the game's
lighting uses it as its own and it darkens ambient light only.

## The game's AO: two paths

The game picks one per run or area (both seen on 2026-10-12, 2560x1440, same settings).

Temporal path:

| Pass | Pixel shader | What |
|---|---|---|
| 1 | `01ae5388` | linear depth R32F, full size -> half size |
| 2 | `f7229091` | copy of the previous frame's AO (history; a generic copy shader) |
| 3 | `83da8467` | SSAO: half-size depth, a G-buffer target, 64x64 noise, history -> RGBA8 half size |
| 4 | `c45c51cb` | depth-aware upsample -> **R8Unorm, scene size**: the final AO |

Half-size path: `01ae5388`, then `cbbca5dc` (no history) -> **RGBA8 half size**, read directly.

Either AO target is read by `a291c498` (merged into the RGBA8 light/shadow mask) and `f2531727`
(lighting into the two B10G11R11 targets); both use only its red channel, and the lighting
multiplies by it (1 = unoccluded; decompiled from the pipeline cache's SPIR-V). With SSAO off
both read a 1x1 texture and `f2531727` becomes `b59b4c4e`.

## The port

`Vbao::OnAoPass` (vk_vbao.cpp) runs right after the draw with pixel shader `c45c51cb` or
`cbbca5dc`, at the AO target's size (the scene size, or half of it). Compute `vbao.comp` reads
the scene depth (`CameraMotion::Depth`) and projection (`CameraMotion::SceneProjection`),
takes the normals from the G-buffer the game's SSAO reads (its texture 1: world normals, unorm
xyz, rotated into view space with `CameraMotion::SceneView`; they carry the normal maps, which
is where the game's SSAO gets its fine detail) and falls back to normals from depth where they
disagree with the surface. It writes the raw AO, the view depth and the normal into a scratch
RGBA16F image. 3 slices x 8 steps per side, 32 sectors; the pattern rotates per frame only while a
temporal upscaler accumulates (else it is fixed). `vbao_blur.comp` then denoises it (5x5
Gaussian, taps weighted down by their depth and normal differences) into the R8 target, or (variant
`OUTPUT_RGBA`) the red channel of the RGBA8 target.

The game's own SSAO passes still run (skipping them would save their cost).

## Tuning

- `BB_VBAO_RADIUS` (metres, 1.5), `BB_VBAO_THICKNESS` (metres, 0.4), `BB_VBAO_POWER` (1.5).
- `BB_VBAO_BLUR`: denoise taps per side (2 = 5x5, at most 4; 0 = the raw AO),
  `BB_VBAO_BLUR_DEPTH`: relative depth difference that stops the blur (0.05),
  `BB_VBAO_BLUR_NORMAL`: exponent on the cosine between normals (8; 0 = ignores normals).
- `BB_VBAO_NORMALS=0`: normals from depth only (no normal-map detail).
- AO-only view: overlay menu "Show AO only (debug)" (`debug_view=3`). At the display pass,
  `Vbao::OnDisplayPass` draws this frame's AO target (VBAO, or the game's SSAO with VBAO off) as
  grey over the frame (`vbao_view.comp`). With a scaled upscaler session the upscaler's own output
  is shown instead of the game's frame: there the view draws into it right after the upscale
  (`TemporalUpscaler::on_upscaled`, before the UI).
