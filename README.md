# mesa-kosmickrisp

Provides CMake build logic for compiling KosmicKrisp and the Vulkan loader on macOS, for either ARM64 or x86_64, on an ARM64 host.

## Building

Install the required dependencies, e.g. with brew and pip:
```
brew install cmake meson ninja pkg-config llvm spirv-tools spirv-llvm-translator
pip3 install --break-system-packages mako packaging pyyaml
```

Then build using CMake, replacing the architecture with your desired target:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build --parallel$(sysctl -n hw.ncpu)
```

