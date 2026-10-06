# Sourced by capture_all.sh and verify.sh: the Proton and vkd3d-proton settings of a capture.
#
# PROTONPATH, else the newest GE-Proton of Steam's compatibilitytools.d (native, ~/.steam or
# Flatpak). Version order: `ls | tail -1` took GE-Proton9-27 over GE-Proton10-x, and older
# vkd3d-proton loads only the DLL's FSR 2/3 providers (issue #4).
if [[ -z ${PROTONPATH:-} ]]; then
    PROTONPATH=$(for d in "$HOME/.local/share/Steam/compatibilitytools.d" \
                          "$HOME/.steam/root/compatibilitytools.d" \
                          "$HOME/.var/app/com.valvesoftware.Steam/data/Steam/compatibilitytools.d"; do
                     for p in "$d"/GE-Proton*; do [[ -d $p ]] && printf '%s\t%s\n' "${p##*/}" "$p"; done
                 done | sort -V -k1,1 | tail -1 | cut -f2)
    if [[ -z $PROTONPATH ]]; then
        echo "No GE-Proton found in Steam's compatibilitytools.d: install GE-Proton 10 or newer" \
             "(ProtonUp-Qt), or set PROTONPATH to a Proton folder." >&2
        exit 1
    fi
fi
export PROTONPATH
echo "Proton: $PROTONPATH"
# The DLL picks its shader variant from what vkd3d-proton offers: with FP8 cooperative matrices
# (RDNA4) it dispatches another variant than the one vk_fsr411.cpp replays (issue #12). Hidden
# here; on an RX 7800 XT the capture is the same byte for byte either way.
export VKD3D_DISABLE_EXTENSIONS=${VKD3D_DISABLE_EXTENSIONS-VK_KHR_cooperative_matrix,VK_NV_cooperative_matrix2,VK_EXT_shader_float8}
