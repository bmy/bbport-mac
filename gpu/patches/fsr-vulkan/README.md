# Local changes to gpu/third_party/fsr-vulkan

The submodule points at upstream FireBurn/FSR-Vulkan (`c64f093`). `build.sh` applies the
patches here to its working tree when they are not applied yet:

- `0001-...`: `BB_FSR4_PROFILE` (GPU time per FSR 4 pass) and `BB_FSR4_STATS` (driver
  statistics of each pass) in the FSR 4 v07 provider.
- `0002-...`: the FFX Vulkan backend never picks a protected memory type unless it asked for one.
  On RADV PHOENIX (Ryzen Z1 Extreme: ROG Ally, Legion Go) it picked one for FSR 3.1's
  device-local resources, and the scene was white or full of colored noise (#85). Also sent to
  FireBurn/FSR-Vulkan#1.
