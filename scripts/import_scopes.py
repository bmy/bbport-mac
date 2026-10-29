#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Import names in one spelling for every game.

A PS4 executable names an import NID#L#M, where L and M are its own numbers for the library and
the module (from its dynamic table): libkernel is #p#J in Bloodborne's eboot.bin and #u#c in
another game's. The runtime's tables were written against Bloodborne 1.09, so imports are spelled
with its numbers for the libraries and modules it imports, and NID#library for the others.
"""

# Bloodborne 1.09 (CUSA03173 and the other retail releases): library and module numbers.
LIBRARIES = {'libSceAjm': 'N', 'libSceAppContent': 'c', 'libSceAudioIn': 'Y', 'libSceAudioOut': 'G',
             'libSceAvPlayer': 'm', 'libSceCommonDialog': 'Z', 'libSceFios2': 'F',
             'libSceGameLiveStreaming': 'e', 'libSceGnmDriver': 'B', 'libSceHttp': 'K',
             'libSceImeDialog': 'b', 'libSceMouse': 'U', 'libSceMsgDialog': 'j', 'libSceNet': 'E',
             'libSceNetCtl': 'J', 'libSceNpAuth': 'X', 'libSceNpCommerce': 'k',
             'libSceNpCommon': 'Q', 'libSceNpManager': 'R', 'libSceNpMatching2': 'P',
             'libSceNpProfileDialog': 'h', 'libSceNpScore': 'o', 'libSceNpSignaling': 'T',
             'libSceNpTrophy': 'l', 'libSceNpUtility': 'S', 'libSceNpWebApi': 'W', 'libScePad': 'A',
             'libScePlayGo': 'n', 'libScePosix': 'I', 'libSceRtc': 'H', 'libSceSaveData': 'O',
             'libSceSaveDataDialog': 'd', 'libSceScreenShot': 'g', 'libSceSsl': 'L',
             'libSceSysmodule': 'M', 'libSceSystemService': 'V', 'libSceUserService': 'D',
             'libSceVideoOut': 'C', 'libSceVideoRecording': 'f', 'libSceVoice': 'i',
             'libSceWebBrowserDialog': 'a', 'libc': 'q', 'libkernel': 'p'}
MODULES = {'libSceAjm': 'O', 'libSceAppContentUtil': 'd', 'libSceAudioIn': 'Z', 'libSceAudioOut': 'H',
           'libSceAvPlayer': 'n', 'libSceCommonDialog': 'a', 'libSceFios2': 'G',
           'libSceGameLiveStreaming': 'f', 'libSceGnmDriver': 'C', 'libSceHttp': 'L',
           'libSceImeDialog': 'c', 'libSceMouse': 'V', 'libSceMsgDialog': 'k', 'libSceNet': 'F',
           'libSceNetCtl': 'K', 'libSceNpAuth': 'Y', 'libSceNpCommerce': 'l', 'libSceNpCommon': 'R',
           'libSceNpManager': 'S', 'libSceNpMatching2': 'Q', 'libSceNpProfileDialog': 'i',
           'libSceNpScore': 'p', 'libSceNpSignaling': 'U', 'libSceNpTrophy': 'm',
           'libSceNpUtility': 'T', 'libSceNpWebApi': 'X', 'libScePad': 'B', 'libScePlayGo': 'o',
           'libSceRtc': 'I', 'libSceSaveData': 'P', 'libSceSaveDataDialog': 'e',
           'libSceScreenShot': 'h', 'libSceSsl': 'M', 'libSceSysmodule': 'N',
           'libSceSystemService': 'W', 'libSceUserService': 'E', 'libSceVideoOut': 'D',
           'libSceVideoRecording': 'g', 'libSceVoice': 'j', 'libSceWebBrowserDialog': 'b',
           'libc': 'q', 'libkernel': 'J'}


def canonical(name, libraries, modules):
    """`name` (NID#L#M in the executable's numbers; `libraries` and `modules` map them to
    (name, version)) in the runtime's spelling. Other names are kept."""
    parts = name.split('#')
    if len(parts) != 3 or parts[1] not in libraries or parts[2] not in modules:
        return name
    nid, library, module = parts[0], libraries[parts[1]][0], modules[parts[2]][0]
    if library in LIBRARIES and module in MODULES:
        return f'{nid}#{LIBRARIES[library]}#{MODULES[module]}'
    return f'{nid}#{library}'
