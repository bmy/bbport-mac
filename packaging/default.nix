# Packaged port: the prebuilt game binaries (build.sh first), the start-up scripts and the GTK4
# launcher, with their whole Nix closure and Mesa's Vulkan drivers. `bash packaging/appimage.sh`
# turns it into an AppImage (Steam Deck); `nix-build packaging` alone gives result/bin/bbport.
{ pkgs ? import <nixpkgs> { }
  # Store paths the prebuilt binaries load libraries from (their RUNPATHs), written by
  # appimage.sh. Nix only finds references to its inputs, and the binaries were built outside.
, runtimePaths ? (if builtins.pathExists ./runtime-paths.nix then import ./runtime-paths.nix else [ ])
}:
let
  lib = pkgs.lib;
  root = ./..;
  # FSR 4.1.1 models are extracted from AMD's DLLs: never in a public package. BB_PACKAGE_FSR411=1
  # (appimage.sh runs nix with --impure) bundles the local fsr4_411 for one's own devices.
  fsr411 = builtins.getEnv "BB_PACKAGE_FSR411" == "1";
  assetDirs = [ "scripts" "patches" "fsr4_shaders" "launcher" "tools/fsr4cap" ]
    ++ lib.optional fsr411 "fsr4_411";
  # Only what the package needs (the tree also holds builds, profiles and captures).
  wanted = [
    "run.sh" "out" "out/bb-probe" "out/bb-gpu-capabilities" "out/gpu" "out/gpu/libbbgpu.so" "tools"
  ] ++ assetDirs;
  # DLSS (a build with DLSS_SDK_ROOT): the bridge, NVIDIA's runtime library and its license, as
  # NVIDIA's RTX SDK license allows in an application. Without them the launcher lets the player
  # choose NVIDIA's library (the bridge is still needed).
  dlssFiles = [ "out/libbbport_dlss.so" "out/NVIDIA-DLSS-LICENSE.txt" "out/gpu/libbbnet.so" ];
  isNgx = rel: lib.hasPrefix "out/libnvidia-ngx-dlss.so." rel;
  src = builtins.path {
    name = "bbport-src";
    path = root;
    filter = path: type:
      let rel = lib.removePrefix (toString root + "/") (toString path);
      in builtins.elem rel wanted || builtins.elem rel dlssFiles || isNgx rel
        || lib.any (dir: lib.hasPrefix (dir + "/") rel) assetDirs;
  };
  python = pkgs.python3.withPackages (ps: [ ps.pygobject3 ]);
  # FSR 4.1.1 from the user's own AMD DLL (the launcher's button, tools/fsr4cap/build_assets.sh):
  # its two tools built here, so that the package needs no compiler and no network for it. Proton
  # (GE-Proton 10+) and Steam's runtime come from the user's Steam. On NixOS the recording runs
  # on the host (systemd-run --user, below on PATH; umu-launcher from the host's nix-shell).
  dxilSpirv = pkgs.stdenv.mkDerivation {
    pname = "dxil-spirv";
    version = "0-unstable-7dc5278";
    src = pkgs.fetchFromGitHub {
      owner = "HansKristian-Work";
      repo = "dxil-spirv";
      rev = "7dc52786cdb1f53c54dd5c5c2698dff00ea5f0a3";
      fetchSubmodules = true;
      hash = "sha256-vdSMp6QusmVTGpcl9jr+wQUQy5HLWwjr/5TI8ARMWRI=";
    };
    patches = [ ../tools/fsr4cap/dxil-spirv-class-bindings.patch ];
    nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python3 ];
    # The CLI and the shared library it links (its own install rules set the RUNPATH).
  };
  fsr4capExe = pkgs.runCommand "fsr4cap.exe" {
    nativeBuildInputs = [ pkgs.pkgsCross.mingwW64.buildPackages.gcc ];
  } ''
    f=${../tools/fsr4cap}
    x86_64-w64-mingw32-gcc -std=c11 -O1 -Wall -I$f/ffx/api/include -I$f/ffx/upscalers/include \
      $f/fsr4cap.c $f/capture.c $f/rootsig.c -o $out -ld3d12 -ldxguid -static
  '';
  # Mesa's Vulkan drivers come with the package (RADV and ANV); bbport_vulkan.py adds the host
  # NVIDIA ICD and only its vendor libraries (matching the host's kernel module). Only the two
  # drivers, not the whole Mesa: its OpenGL drivers and LLVM were ~800 MB of the package. RADV
  # without LLVM (it compiles with ACO; LLVM was only an optional backend); the drivers' one
  # reference to Mesa's store path (share/drirc.d, per-application tweaks: none for this game)
  # is removed.
  mesaVulkan = (pkgs.mesa.override {
    galliumDrivers = [ "radeonsi" "iris" "llvmpipe" "zink" ];
    # Dozen (microsoft-experimental) only because the recipe expects its spirv2dxil output.
    vulkanDrivers = [ "amd" "intel" "microsoft-experimental" ];
  }).overrideAttrs (old: {
    mesonFlags = old.mesonFlags ++ [ "-Damd-use-llvm=false" ];
  });
  vulkanDrivers = pkgs.runCommand "bbport-vulkan-drivers" {
    nativeBuildInputs = [ pkgs.removeReferencesTo pkgs.jq ];
  } ''
    mkdir -p $out/lib $out/share/vulkan/icd.d
    for name in radeon intel; do
      cp ${mesaVulkan}/lib/libvulkan_$name.so $out/lib/
      chmod u+w $out/lib/libvulkan_$name.so
      remove-references-to -t ${mesaVulkan} $out/lib/libvulkan_$name.so
      jq ".ICD.library_path = \"$out/lib/libvulkan_$name.so\"" \
        ${mesaVulkan}/share/vulkan/icd.d/''${name}_icd.x86_64.json > $out/share/vulkan/icd.d/''${name}_icd.x86_64.json
    done
  '';
  icds = lib.concatMapStringsSep ":" (name: "${vulkanDrivers}/share/vulkan/icd.d/${name}")
    [ "radeon_icd.x86_64.json" "intel_icd.x86_64.json" ];
  # MangoHud (its Vulkan layer, XDG_DATA_DIRS) without its 32-bit half: the game is 64-bit.
  # xdg-utils (Perl, ~55 MB) only opens links: the host's xdg-open instead.
  # libdecor without its GTK 3 plugin (as in shell.nix) for MangoHud's glfw: GTK 3, tinysparql, ICU.
  libdecor = pkgs.libdecor.overrideAttrs (old: {
    buildInputs = lib.remove pkgs.gtk3 old.buildInputs;
    mesonFlags = old.mesonFlags ++ [ "-Dgtk=disabled" ];
  });
  mangohud = pkgs.mangohud.override {
    lowerBitnessSupport = false;
    glfw = pkgs.glfw.override { inherit libdecor; };
    xdg-utils = pkgs.writeShellScriptBin "xdg-open" ''
      host=$(command -v xdg-open) && exec "$host" "$@"
      exit 1
    '';
  };
  # The launcher's GTK without GStreamer (video playback: flite, x265, ... ~200 MB), printing and
  # the search indexer, which it never uses; libadwaita on top of it.
  gtk4 = (pkgs.gtk4.override { cupsSupport = false; trackerSupport = false; }).overrideAttrs (old: {
    buildInputs = lib.filter (p: !(lib.hasPrefix "gst-" (p.pname or ""))) old.buildInputs;
    mesonFlags = old.mesonFlags ++ [ "-Dmedia-gstreamer=disabled" ];
  });
  libadwaita = pkgs.libadwaita.override { inherit gtk4; };
  # Fonts: bundled DejaVu and Adwaita plus the host's usual font directories, but not the host's
  # /etc/fonts: on NixOS it names fonts in the host's /nix/store, which the AppImage hides behind
  # its own store in some environments (Steam's FHS sandbox), and the launcher showed boxes.
  fontsConf = pkgs.writeText "bbport-fonts.conf" ''
    <?xml version="1.0"?>
    <!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">
    <fontconfig>
      <dir>${pkgs.dejavu_fonts}/share/fonts</dir>
      <dir>${pkgs.adwaita-fonts}/share/fonts</dir>
      <dir>/usr/share/fonts</dir>
      <dir>/usr/local/share/fonts</dir>
      <dir prefix="xdg">fonts</dir>
      <cachedir prefix="xdg">bbport/fontconfig</cachedir>
      <include ignore_missing="yes">${pkgs.fontconfig.out}/etc/fonts/conf.d</include>
    </fontconfig>
  '';
  # Environment the closure needs on any host: icon themes, SVG icon loader, fonts (above) and
  # a UTF-8 locale built into glibc.
  common = ''
      --prefix XDG_DATA_DIRS : ${mangohud}/share:${pkgs.adwaita-icon-theme}/share:${pkgs.hicolor-icon-theme}/share:${gtk4}/share/gsettings-schemas/${gtk4.name} \
      --set-default GDK_PIXBUF_MODULE_FILE ${pkgs.librsvg}/${pkgs.gdk-pixbuf.moduleDir}.cache \
      --set-default FONTCONFIG_FILE ${fontsConf} \
      --set-default LC_ALL C.UTF-8 \
      --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath [ pkgs.libglvnd pkgs.libx11 pkgs.libxext ]} \
      --set BB_VULKANINFO ${pkgs.vulkan-tools}/bin/vulkaninfo \
      --run '[ -r "''${SSL_CERT_FILE:-}" ] || export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt; [ -d "''${SSL_CERT_DIR:-}" ] || unset SSL_CERT_DIR' \
  '';
in
pkgs.stdenv.mkDerivation {
  pname = "bbport";
  version = "0.1";
  inherit src;
  # The hook adds its GTK to the wrapper (typelibs, schemas): ours, not nixpkgs' with GStreamer.
  nativeBuildInputs = [ pkgs.makeShellWrapper (pkgs.wrapGAppsHook3.override { gtk3 = gtk4; }) pkgs.gobject-introspection ];
  buildInputs = [ gtk4 libadwaita pkgs.adwaita-icon-theme pkgs.librsvg ]
    ++ map builtins.storePath runtimePaths;
  dontBuild = true;
  dontConfigure = true;
  # The binaries live under share/ (next to the scripts run.sh expects): strip them too, which
  # also drops the compiler and header paths their debug info would keep in the closure.
  stripDebugList = [ "share/bbport/bin/gpu" ];
  # patchelf (RPATH shrinking) corrupts the non-PIE game binary's symbol versions; it finds
  # its library through $ORIGIN/gpu and its other libraries through the build's RUNPATH.
  dontPatchELF = true;
  dontWrapGApps = true; # wrapped once below, together with the launcher's own variables
  installPhase = ''
    runHook preInstall
    d=$out/share/bbport
    mkdir -p $d/bin $out/bin
    cp run.sh $d/
    cp -r scripts patches fsr4_shaders launcher $d/
    mkdir -p $d/tools
    cp -r tools/fsr4cap $d/tools/
    rm -rf $d/tools/fsr4cap/__pycache__
    mkdir -p $d/tools/fsr4cap/bin
    ln -s ${dxilSpirv}/bin/dxil-spirv $d/tools/fsr4cap/bin/dxil-spirv
    install -m644 ${fsr4capExe} $d/tools/fsr4cap/bin/fsr4cap.exe
    # FSR 4.1.1 models only with BB_PACKAGE_FSR411=1 (see above); otherwise run.sh finds them in
    # the data directory (~/.local/share/bbport/fsr4_411).
    if [ -d fsr4_411 ]; then
      ${pkgs.python3}/bin/python3 - <<'PY'
    import sys
    from pathlib import Path
    sys.path.insert(0, 'launcher')
    from bbport_assets import fsr411_problem
    for output in ('1920x1080', '3840x2160'):
        for preset in (0, 4):
            problem = fsr411_problem(Path('fsr4_411'), output, preset)
            if problem:
                raise SystemExit(problem)
    PY
      cp -r fsr4_411 $d/
    fi
    install -m755 out/bb-probe $d/bin/bb-probe
    install -m755 out/bb-gpu-capabilities $d/bin/bb-gpu-capabilities
    install -Dm755 out/gpu/libbbgpu.so $d/bin/gpu/libbbgpu.so
    # Their debug info names the build environment's headers (the full GCC, Vulkan headers): Nix
    # took those for dependencies, ~300 MB. Symbols stay (crash reports).
    ${pkgs.binutils-unwrapped}/bin/strip --strip-debug $d/bin/bb-probe $d/bin/bb-gpu-capabilities
    # The online module (gpu/bbnet), when the build has it.
    if [ -f out/gpu/libbbnet.so ]; then
      install -Dm755 out/gpu/libbbnet.so $d/bin/gpu/libbbnet.so
    fi
    if [ -f out/libbbport_dlss.so ]; then
      install -m755 out/libbbport_dlss.so $d/bin/libbbport_dlss.so
      for ngx in out/libnvidia-ngx-dlss.so.*; do
        [ -f "$ngx" ] && install -m755 "$ngx" $d/bin/
      done
      [ -f out/NVIDIA-DLSS-LICENSE.txt ] && install -m644 out/NVIDIA-DLSS-LICENSE.txt $d/NVIDIA-DLSS-LICENSE.txt
    fi
    runHook postInstall
  '';
  postFixup = ''
    # bin/bbport is a static program (bbport-entry.c) that clears the host's LD_PRELOAD (Steam's
    # overlay) and similar before the wrapper's own dynamic programs start.
    mkdir -p $out/libexec
    gcc -O2 -Wall -Werror -static -L${pkgs.glibc.static}/lib -DTARGET="\"$out/libexec/bbport\"" \
      ${./bbport-entry.c} -o $out/bin/bbport
    makeShellWrapper ${python}/bin/python3 $out/libexec/bbport \
      "''${gappsWrapperArgs[@]}" \
      ${common}      --add-flags "$out/share/bbport/launcher/bbport_vulkan.py ${python}/bin/python3 $out/share/bbport/launcher/bbport_launcher.py" \
      --set BB_PREBUILT 1 \
      --set PYTHON ${pkgs.python3}/bin/python3 \
      --set BB_BUNDLED_VK_DRIVER_FILES ${icds} \
      --prefix PATH : ${lib.makeBinPath [ pkgs.bash pkgs.coreutils pkgs.util-linux pkgs.procps
                                          pkgs.gawk pkgs.gnused pkgs.gnugrep pkgs.spirv-tools
                                          pkgs.systemdMinimal ]} \
      --run 'export BB_DATA_DIR=''${BB_DATA_DIR:-''${XDG_DATA_HOME:-$HOME/.local/share}/bbport}; mkdir -p "$BB_DATA_DIR"'
    # The game alone, without the launcher (settings from the data directory's bbport.ini).
    makeShellWrapper ${pkgs.python3}/bin/python3 $out/bin/bbport-game \
      ${common}      --add-flags "$out/share/bbport/launcher/bbport_vulkan.py ${pkgs.bash}/bin/bash $out/share/bbport/run.sh" \
      --set BB_PREBUILT 1 \
      --set PYTHON ${pkgs.python3}/bin/python3 \
      --set BB_BUNDLED_VK_DRIVER_FILES ${icds} \
      --prefix PATH : ${lib.makeBinPath [ pkgs.bash pkgs.coreutils pkgs.procps ]} \
      --run 'export BB_DATA_DIR=''${BB_DATA_DIR:-''${XDG_DATA_HOME:-$HOME/.local/share}/bbport}; mkdir -p "$BB_DATA_DIR"'
  '';
  meta.mainProgram = "bbport";
}
