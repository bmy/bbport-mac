# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compile selected shadPS4/GoldHEN XML patches into out/patches.bin for the loader.

Patch addresses are PS4 virtual addresses (eboot base 0x400000); the loader's
image places eboot vaddr 0 at image offset 0. Only literal writes are supported
(bytes, bytes16/32/64, float32/64, utf8, utf16); pattern ("mask") patches are rejected.
"""
import argparse
import json
import os
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

EBOOT_BASE=0x400000
# BB_FPS presets: patch names from patches/Bloodborne.xml (app version 01.09).
# Above 60 FPS the sprint fix always goes with the frame rate patch: without it sprinting drops to
# half speed (the game's wall detector measured distance per frame, a 30 FPS rule).
FPS_PRESETS={'30':[],'60':['60 FPS++'],'90':['90 FPS++','Sprint Fix (High FPS)'],
             'uncap':['Uncap FPS++','Sprint Fix (High FPS)']}
# Above 30 FPS the frame rate is done by hooks in the game's code (gpu/shim/bbport_frame_rate.cpp:
# 60 and 90 are the same as uncap, at most that many frames a second): no patches. The patches
# only with BB_FRAME_HOOKS=0 (comparisons).


def fps_patches(fps, env=os.environ):
    if fps!='30' and env.get('BB_FRAME_HOOKS','1')!='0':
        return []
    return FPS_PRESETS[fps]
# Upscaler presets (bbport.ini "preset", the in-game menu): output / render size ratio. The game
# then renders at 1920x1080 / ratio and the port's temporal upscaler restores the output size.
OUTPUT_SIZE=(1920,1080)
PRESET_SCALES=[1.0,1.5,1.7,2.0,3.0,1.25]  # 5: Ultra Quality (bbport-mac)
# The community patch changes two independent consumers: the game render/window setup
# and the UI movie viewport. Keep the latter at native size so glyph rasterisation and
# vector tessellation do not inherit the scene's FSR resolution.
RESOLUTION_TEMPLATE='Resolution Patch 1280x720 (16:9)'
# Effect switches (bbport.ini, in-game menu, launcher): key -> (patch when the key is 0, patch
# when it is 1). The game reads these at start; a change applies after a restart.
EFFECTS={
    'effect_chromatic_aberration':('Disable Chromatic Aberration',None),
    'effect_dof':('Disable DoF',None),
    'effect_motion_blur':('Disable Motion Blur (perf increase)',None),
    'effect_ssao':('Disable SSAO',None),
    'effect_game_aa':('Disable AA',None),
    'effect_dynamic_shadows':('Disable Dynamic Light Shadows (perf increase)',None),
    'effect_ssr':(None,'Enable Screen Space Reflections (READ NOTE)'),
    'effect_bloom':('Disable Bloom',None),
    'skip_intro':(None,'Skip Intro'),
    'debug_camera':(None,'Restore Debug Camera'),
    'debug_menu':(None,'Restore Debug Menu (READ NOTES)'),
}
# Intel CPUs: the game's tone mapping turns black (DLC areas most; reported as darker than on the
# PS4): the game code runs natively, and Intel's approximate float instructions differ from the
# PS4's AMD CPU, so a block the game gates on a computed flag never runs. The community fix runs
# it always; on by default on Intel. BB_INTEL_TONEMAP_FIX=0/1 forces it off or on.
INTEL_TONEMAP='Intel Black Tonemap Fix'
# The game's AA step forced to FXAA2_GS (anti_aliasing=fxaa); the engine writes it live unless BB_HOT_EFFECTS=0.
FORCE_FXAA='FXAA2_GS'


def intel_cpu(cpuinfo='/proc/cpuinfo'):
    try:
        with open(cpuinfo) as f:
            return any(line.startswith('vendor_id') and 'GenuineIntel' in line for line in f)
    except OSError:
        # Windows: "Intel64 Family 6 Model 151 Stepping 2, GenuineIntel".
        return 'GenuineIntel' in os.environ.get('PROCESSOR_IDENTIFIER', '')


# The title's PLAY ONLINE / PLAY OFFLINE dialog: the port has no PSN, the game goes straight to the
# main menu offline. On by default (the launcher's switch); BB_SKIP_NETWORK_CHOICE=0 shows it.
SKIP_NETWORK_CHOICE='Skip Online/Offline Choice'


def skip_network_choice(env=os.environ):
    return env.get('BB_SKIP_NETWORK_CHOICE','1')!='0'


# Two per-frame GX command heaps: the game writes through NULL when a 256 KiB heap fills (#162).
GX_COMMAND_HEAP='Larger GX Per-Frame Command Heap'


def gx_command_heap(env=os.environ):
    return env.get('BB_GX_COMMAND_HEAP','1')!='0'


# The community patch that grows the game's two graphics heaps (2287 and 886 MiB) to 3952 and
# 2560 MiB (the fallback of graphics_heap_writes below).
GRAPHICS_HEAP_PATCH='Increased Graphics Heap Sizes'


def graphics_heaps(size, large_heaps):
    """Whether larger graphics heaps go in: a scene above 1080p (sized for it), or a texture pack
    (--large-heaps from run.sh / run.py: game_check.texture_pack, #240; the patch's full sizes:
    upscaled map textures fill the retail heap whatever the scene)."""
    return large_heaps or bool(size and size[0]*size[1]>OUTPUT_SIZE[0]*OUTPUT_SIZE[1])


def intel_tonemap_fix(env=os.environ, cpuinfo='/proc/cpuinfo'):
    forced=env.get('BB_INTEL_TONEMAP_FIX')
    return forced=='1' if forced in ('0','1') else intel_cpu(cpuinfo)

# model_lod: -2 highest, 0 the game's, 1 lower, 2 lowest.
MODEL_LOD={'-2':'Model LOD -2 (Highest)','1':'Model LOD 1 (Lower)','2':'Model LOD 2 (Lowest)'}


DEBUG_MENU='Restore Debug Menu (READ NOTES)'
# What the patched game reads for the debug menu (BB_FILE_TRACE=1), under dvdroot_ps4. The game asks
# for lower-case names; the PS4's file system and the port's runtime ignore case.
DEBUG_MENU_FONTS=('adhoc/font/DbgFont14h.ccm','adhoc/font/DbgFont14h.tpf')
DEBUG_MENU_SHADERS=('adhoc/FontShader/debugFont_vs.vpo','adhoc/FontShader/debugFont_ps.ppo')


def find_ignoring_case(root, relative):
    """`root`/`relative`, each component matched ignoring case; None when absent."""
    path=Path(root)
    for part in relative.split('/'):
        exact=path/part
        if exact.exists():
            path=exact
            continue
        try:
            matches=[e for e in path.iterdir() if e.name.casefold()==part.casefold()]
        except OSError:
            return None
        if not matches:
            return None
        path=matches[0]
    return path


def debug_menu_problem(game):
    """Why the debug menu patch cannot work with this game folder, or None."""
    dvdroot=Path(game)/'dvdroot_ps4'
    missing=[]
    for relative in DEBUG_MENU_FONTS:
        found=find_ignoring_case(dvdroot,relative)
        if not found or not found.is_file() or found.stat().st_size==0:
            missing.append(relative)
    if not missing:
        return None
    hint=''
    if all(find_ignoring_case(dvdroot,'font/'+r.rsplit('/',1)[1]) for r in missing):
        hint=' (they are in dvdroot_ps4/font: the game reads them from dvdroot_ps4/adhoc/font)'
    return ('the debug menu needs the font files from https://www.nexusmods.com/bloodborne/mods/253 '
            f'in dvdroot_ps4 of the game folder or of a mod: missing {", ".join(missing)}{hint}')


def validate_patch_requirements(names, game):
    """The patches to apply: raises for conflicts; drops the debug menu (with a message) when its
    font files are missing, as the game crashes opening the menu without them."""
    if 'Restore Debug Camera' in names and 'Enemy Control' in names:
        raise ValueError('Restore Debug Camera conflicts with Enemy Control; enable only one')
    if DEBUG_MENU in names:
        problem=debug_menu_problem(game)
        if problem:
            print(f'Patches: debug menu off: {problem}',file=sys.stderr)
            return [n for n in names if n!=DEBUG_MENU]
        absent=[r for r in DEBUG_MENU_SHADERS if not find_ignoring_case(Path(game)/'dvdroot_ps4',r)]
        if absent:
            print(f'Patches: debug menu: {", ".join(absent)} not found (the mod\'s font shaders); '
                  'the menu may not draw',file=sys.stderr)
    return names


# Effects the engine switches live (gpu/games/bloodborne/hooks_effects.cpp, "game effects switched live"):
# they get no startup patch, the engine writes the same bytes from the settings (and again on
# every change in the menu). BB_HOT_EFFECTS=0: the patches at start, as before.
HOT_EFFECTS={'effect_chromatic_aberration','effect_dof','effect_motion_blur','effect_ssao',
             'effect_game_aa','effect_dynamic_shadows','effect_bloom'}
# SSR also changes the size of a resource allocated by the renderer constructor
# (0x22c428d). Applying only its shader/pass switches corrupts the scene; apply
# the complete community patch before construction, and require a restart.


def anti_aliasing(settings):
    """(the game's own AA on, its AA step forced to FXAA2_GS) of bbport.ini anti_aliasing
    (BbSettings::SyncAntiAliasing): auto = the game's own without an upscaler; game; fxaa (FXAA2_GS,
    the AA step on); off; smaa (the port's SMAA, the game's AA off). Without the key
    (older settings) effect_game_aa as it is."""
    mode=settings.get('anti_aliasing')
    if mode is None: return settings.get('effect_game_aa','1')=='1',False
    upscaling=settings.get('upscaler','fsr3')!='off'
    return mode in ('game','fxaa') or (mode=='auto' and not upscaling),mode=='fxaa'


def effect_patches(settings, env=os.environ):
    names=[]
    hot=env.get('BB_HOT_EFFECTS','1')!='0'
    game_aa,fxaa=anti_aliasing(settings)
    settings=dict(settings,effect_game_aa='1' if game_aa else '0')
    if fxaa and not hot: names.append(FORCE_FXAA)
    for key,(off,on) in EFFECTS.items():
        if key not in settings: continue
        if hot and key in HOT_EFFECTS: continue
        name=on if settings[key]=='1' else off
        if name: names.append(name)
    lod=MODEL_LOD.get(settings.get('model_lod','0'))
    if lod: names.append(lod)
    return names
SCENE_WIDTH=0x02196A6B-EBOOT_BASE
SCENE_HEIGHT=0x02196A7A-EBOOT_BASE
UI_WIDTH=0x02358554-EBOOT_BASE
UI_HEIGHT=0x0235855D-EBOOT_BASE


# Issue #269: the game's text and voice language whatever the copy's sell region allows (the EU
# release shows English for Chinese and Korean and plays English voices for Japanese, though the
# files are there; a save keeps the language it was made with). CSLocalizeLang (+0x1fb6cf0) and
# the save's language (+0x14ef0..) map the wanted language through the sell region's tables (one
# per region: game language -> shown language); the patches rewrite those tables: every language
# as itself (voices: those with voice files), or every one as the chosen language.
# Game language numbers (the order of the game's codes jajp, engb, ... zhcn).
GAME_LANGUAGES=['jajp','engb','frfr','itit','dede','eses','esar','ptbr','enus','nlnl','plpl',
                'ruru','dadk','nono','fifi','svse','trtr','ptpt','zhtw','kokr','zhcn']
VOICE_COUNT=8  # jajp .. ptbr have voice files (sound/*_jaj|eng|frf|iti|ded|ese|esa|ptb.fsb)
ENGLISH=1
# The six regions' tables (21 int32 each), text and voice; region 2's text table as a check.
TEXT_TABLES=(0x47366b0,0x4736720,0x47367c0,0x4736830,0x4736890,0x4736950)
VOICE_TABLES=(0x47369b0,0x4736a20,0x4736a80,0x4736ae0,0x4736b40,0x4736ba0)
EU_TEXT_TABLE=(0x47367c0,[1,1,2,3,4,5,5,17,1,9,10,11,12,13,14,15,16,17,1,1,1])


def language_writes(settings, elf):
    """The sell regions' language tables for bbport.ini game_text_language and
    game_voice_language (a code of GAME_LANGUAGES; auto or none: the game's own choice, without
    the region's limits)."""
    site,expected=EU_TEXT_TABLE
    if len(elf)<0x4000+site+84 or list(struct.unpack_from('<21i',elf,0x4000+site))!=expected:
        print('Patches: game languages: this eboot\'s tables differ; left as they are')
        return []
    text=settings.get('game_text_language','auto')
    voice=settings.get('game_voice_language','auto')
    if text!='auto' and text not in GAME_LANGUAGES:
        print(f'Patches: game_text_language={text} is not one of the game\'s languages')
        text='auto'
    if voice!='auto' and (voice not in GAME_LANGUAGES or GAME_LANGUAGES.index(voice)>=VOICE_COUNT):
        print(f'Patches: game_voice_language={voice} has no voice files')
        voice='auto'
    text_map=[GAME_LANGUAGES.index(text)]*21 if text!='auto' else list(range(21))
    voice_map=([GAME_LANGUAGES.index(voice)]*21 if voice!='auto'
               else [i if i<VOICE_COUNT else ENGLISH for i in range(21)])
    writes=[(site,struct.pack('<21i',*text_map)) for site in TEXT_TABLES]
    writes+=[(site,struct.pack('<21i',*voice_map)) for site in VOICE_TABLES]
    print(f'Patches: game languages: text {text}, voice {voice} (any sell region)')
    return writes


def read_settings(path):
    settings={}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            key,sep,value=line.partition('=')
            if sep and not line.startswith('#'): settings[key.strip()]=value.strip()
    return settings


def render_size(settings,override=''):
    """Render resolution for the upscaler preset, or None for native."""
    if override:
        w,h=(int(v) for v in override.lower().split('x'))
        return (w,h)
    if settings.get('upscaler','fsr3')=='off': return None
    preset=int(settings.get('preset','0') or 0)
    scale=PRESET_SCALES[max(0,min(preset,len(PRESET_SCALES)-1))]
    if scale==1.0: return None
    # Even sizes (the game has half-resolution buffers).
    return tuple(max(2,round(v/scale/2)*2) for v in OUTPUT_SIZE)


def output_size(settings):
    """Output (UI) size from bbport.ini output_res, e.g. 3840x2160. auto (and no key: the
    default): the monitor's (BB_MONITOR_SIZE, found by run.sh), else 1920x1080."""
    value=settings.get('output_res','auto').lower()
    if value=='auto': value=os.environ.get('BB_MONITOR_SIZE','')
    try:
        w,h=(int(v) for v in value.split('x'))
        if w>0 and h>0: return (min(w,7680),min(h,4320))
    except ValueError:
        pass
    return OUTPUT_SIZE


def scaled_sizes(settings,align=2):
    """(render, output) for an output other than 1080p (above it, or 720p for the Steam Deck):
    the game renders at output / preset scale (or at the output size without upscaler) and the
    upscaler fills the output. None at 1080p and for TAA (native, live host targets only)."""
    out=output_size(settings)
    # 1080p: the game's own size; only live presets over the startup patch (align 8) patch it,
    # and only for a scene smaller than it.
    if settings.get('upscaler')=='taa' or (out==OUTPUT_SIZE and align!=8): return None
    scale=1.0
    if settings.get('upscaler','fsr3')!='off':
        preset=int(settings.get('preset','0') or 0)
        scale=PRESET_SCALES[max(0,min(preset,len(PRESET_SCALES)-1))]
    # align 8 (live presets over the startup patch): the game rounds its targets up to 8 pixels;
    # a scene of exactly their size keeps the proxies' scale the targets' own (854x534 in 856x536
    # targets drew live presets misplaced and blocky on the Steam Deck).
    render=tuple(max(align,round(v/scale/align)*align) for v in out)
    # A scene of exactly 1920x1080 (4K Performance) is indistinguishable from the game's UI
    # coordinate space, which the port's UI composition recognizes by that size.
    if render==OUTPUT_SIZE:
        # Frame generation runs on the scaled path only (TemporalUpscaler::RunScaled): with it on, a
        # native 1080p session is scaled too, from the size the 4K case uses (a GTX 1060 user's
        # FSR 3.1 frame generation never started at 1080p Native AA, 2026-10-21).
        frame_generation=settings.get('frame_generation','off') not in ('off','0','')
        if out==OUTPUT_SIZE and not frame_generation: return None
        render=(1912,1072) if align==8 else (1916,1078)
    return render,out


def resolution_writes(xml,size,app_version,segments,ui=OUTPUT_SIZE):
    writes=compile_patches(xml,[RESOLUTION_TEMPLATE],app_version,segments)
    replacements={SCENE_WIDTH:(0xB8,0x500,size[0]),
                  SCENE_HEIGHT:(0xB8,0x2D0,size[1]),
                  UI_WIDTH:(0xB8,0x500,ui[0]),
                  UI_HEIGHT:(0xB9,0x2D0,ui[1])}
    out=[]
    seen=set()
    for offset,data in writes:
        if offset in replacements:
            opcode,old,value=replacements[offset]
            if data!=bytes([opcode])+old.to_bytes(3,'little') or offset in seen:
                raise ValueError(f'unexpected resolution patch at {offset+EBOOT_BASE:#x}')
            data=bytes([opcode])+value.to_bytes(3,'little')
            seen.add(offset)
        out.append((offset,data))
    if seen!=replacements.keys():
        raise ValueError('resolution patch is missing scene/UI viewport instructions')
    return out


# The game's direct memory heap sizes (u64 bytes, a table in its data): the two graphics heaps,
# 2287 and 886 MiB. The community patch "Increased Graphics Heap Sizes" sets 3952 and 2560 MiB,
# sized for 4K. With the GPU using the game's memory in place, all of a heap is resident in RAM:
# at 2560x1080 those sizes left 3.7 GiB more RAM in use, while the game used about what it uses
# at 1080p (1.1 and 0.6 GiB, measured in a heavy area). The heaps grow with the scene instead:
# by (pixels / 1080p - 1) / 3 of the patch's increase, all of it at 4K.
GRAPHICS_HEAPS=((0x04b36dd8,0x8ef00000,0xf7000000),(0x04b36de0,0x37600000,0xa0000000))


def elf_read(elf, vaddr, size):
    """Bytes at a virtual address of the ELF image (its PT_LOAD file contents), or None."""
    phoff,=struct.unpack_from('<Q',elf,0x20)
    phentsize,phnum=struct.unpack_from('<HH',elf,0x36)
    for i in range(phnum):
        kind,_,offset,start,_,filesz,_,_=struct.unpack_from('<IIQQQQQQ',elf,phoff+i*phentsize)
        if kind==1 and start<=vaddr and vaddr+size<=start+filesz:
            return elf[offset+vaddr-start:offset+vaddr-start+size]
    return None


def graphics_heap_writes(elf, size, full=False):
    """The graphics heaps sized for the scene `size` (`full`: the patch's sizes, a texture pack),
    or None when the eboot's table differs."""
    scale=1.0 if full else min(1.0,max(0.0,(size[0]*size[1]/(OUTPUT_SIZE[0]*OUTPUT_SIZE[1])-1)/3))
    writes=[]
    for address,original,full in GRAPHICS_HEAPS:
        offset=address-EBOOT_BASE
        current=elf_read(elf,offset,8)
        if current is None or struct.unpack('<Q',current)[0]!=original:
            return None
        granule=16<<20
        grown=original+-(-int((full-original)*scale)//granule)*granule
        writes.append((offset,struct.pack('<Q',min(grown,full))))
    return writes


def eboot_segments(elf):
    phoff,=struct.unpack_from('<Q',elf,0x20)
    phentsize,phnum=struct.unpack_from('<HH',elf,0x36)
    segments=[]
    for i in range(phnum):
        kind,_,_,vaddr,_,_,memsz,_=struct.unpack_from('<IIQQQQQQ',elf,phoff+i*phentsize)
        if kind==1: segments.append((vaddr,vaddr+memsz))
    return segments


def encode(line):
    kind,value=line.get('Type'),line.get('Value')
    if kind=='bytes': return bytes.fromhex(value.replace(' ',''))
    if kind in ('bytes16','bytes32','bytes64'):
        return int(value,0).to_bytes(int(kind[5:])//8,'little')
    if kind=='float32': return struct.pack('<f',float(value))
    if kind=='float64': return struct.pack('<d',float(value))
    if kind=='utf8': return value.encode()+b'\0'
    if kind=='utf16': return value.encode('utf-16-le')+b'\0\0'
    raise ValueError(f'unsupported patch type {kind!r}')


def compile_patches(xml, names, app_version, segments):
    found={}
    for meta in ET.parse(xml).getroot().iter('Metadata'):
        if meta.get('Name') in names and meta.get('AppVer')==app_version and meta.get('AppElf','eboot.bin')=='eboot.bin':
            found[meta.get('Name')]=meta
    missing=[n for n in names if n not in found]
    if missing: raise ValueError(f'patches not found for app version {app_version}: {missing}')
    writes=[]
    for name in names:
        for line in found[name].iter('Line'):
            offset=int(line.get('Address'),0)-EBOOT_BASE
            data=encode(line)
            if not any(start<=offset and offset+len(data)<=end for start,end in segments):
                raise ValueError(f'{name}: address {line.get("Address")} is outside the eboot')
            writes.append((offset,data))
    return writes


# Third-party patch files (shadPS4/GoldHEN XML) in the data directory's patches/ folder.
# The retail releases (scripts/game_check.py SUPPORTED_TITLES): one 1.09 executable.
BLOODBORNE_IDS={'CUSA00900','CUSA00207','CUSA00208','CUSA00299','CUSA01363',
                'CUSA03179','CUSA03173','CUSA03014','CUSA03023'}


def external_patches(directory, app_version='01.09', exclude=Path(__file__).resolve().parent.parent/'patches/Bloodborne.xml'):
    """[(key, file, metadata)] of eboot patches for this version in directory/*.xml.
    key is "<file name>/<patch name>" (the launcher's selection, patches.json)."""
    found=[]
    directory=Path(directory)
    for path in sorted(directory.glob('*.xml')) if directory.is_dir() else []:
        if path.resolve()==Path(exclude).resolve(): continue  # the built-in file (patches/ in a checkout)
        try:
            root=ET.parse(path).getroot()
        except ET.ParseError as error:
            print(f'Patches: {path.name}: {error}',file=sys.stderr)
            continue
        ids={e.text.strip() for e in root.iter('ID') if e.text}
        if ids and not ids&BLOODBORNE_IDS: continue
        for meta in root.iter('Metadata'):
            if meta.get('AppVer')==app_version and meta.get('AppElf','eboot.bin')=='eboot.bin':
                found.append((f'{path.name}/{meta.get("Name")}',path,meta))
    return found


def external_selection(found, config):
    """Selected external patches: patches.json {"enabled": [...], "disabled": [...]} overrides
    each file's isEnabled."""
    settings={}
    if config and Path(config).is_file():
        settings=json.loads(Path(config).read_text(encoding='utf-8'))
    enabled,disabled=set(settings.get('enabled',[])),set(settings.get('disabled',[]))
    return [(key,path,meta) for key,path,meta in found
            if key in enabled or (key not in disabled and meta.get('isEnabled','false').lower()=='true')]


def compile_external(selected, segments):
    """Writes of the selected external patches; a patch with unsupported lines is skipped whole."""
    writes=[]
    for key,_,meta in selected:
        try:
            ours=[]
            for line in meta.iter('Line'):
                offset=int(line.get('Address') or '',0)-EBOOT_BASE
                data=encode(line)
                if not any(start<=offset and offset+len(data)<=end for start,end in segments):
                    raise ValueError(f'address {line.get("Address")} is outside the eboot')
                ours.append((offset,data))
        except ValueError as error:
            print(f'Patches: skipped {key}: {error}',file=sys.stderr)
            continue
        writes+=ours
        print(f'Patches: external {key} ({len(ours)} writes)')
    return writes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patches-dir',type=Path,help='third-party patch XML files (shadPS4 format)')
    p.add_argument('--patches-config',type=Path,help='patches.json: enabled/disabled external patches')
    p.add_argument('--xml',type=Path,default=Path(__file__).resolve().parent.parent/'patches/Bloodborne.xml')
    p.add_argument('--fps',choices=sorted(FPS_PRESETS),default='uncap')
    p.add_argument('--extra',default='',help='additional patch names, separated by ";"')
    p.add_argument('--app-version',default='01.09')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent.parent/'out')
    p.add_argument('--settings',type=Path,default=Path(__file__).resolve().parent.parent/'bbport.ini')
    p.add_argument('--game-dir',type=Path,default=Path(os.environ.get('BB_GAME_DIR','../CUSA03173')))
    p.add_argument('--render-res',default='',help='render resolution WxH (overrides the preset)')
    p.add_argument('--print-preset-size',action='store_true',help='print the selected preset size, if reduced')
    p.add_argument('--output-res',default='',help='output resolution WxH (the upscaler\'s; the UI stays 1920x1080)')
    p.add_argument('--large-heaps',choices=('0','1'),default='0',
                   help='1: the full graphics heaps whatever the resolution (a texture pack, #240)')
    p.add_argument('--print-scaled',action='store_true',
                   help='print "RENDER OUTPUT" (WxH) when bbport.ini selects an output other than 1080p')
    p.add_argument('--align',type=int,default=2,help='--print-scaled: the scene size\'s multiple (8: live presets)')
    a=p.parse_args()
    if a.print_scaled:
        sizes=scaled_sizes(read_settings(a.settings),a.align)
        if sizes: print(f'{sizes[0][0]}x{sizes[0][1]} {sizes[1][0]}x{sizes[1][1]}')
        return
    if a.print_preset_size:
        settings=read_settings(a.settings)
        if 'BB_UPSCALER' in os.environ:
            settings['upscaler']='fsr3' if os.environ['BB_UPSCALER']=='fsr3' else 'off'
        if 'BB_UPSCALE_PRESET' in os.environ:
            settings['preset']=os.environ['BB_UPSCALE_PRESET']
        size=render_size(settings)
        if size: print(f'{size[0]}x{size[1]}')
        return
    names=fps_patches(a.fps)+[n.strip() for n in a.extra.split(';') if n.strip()]
    names+=[n for n in effect_patches(read_settings(a.settings)) if n not in names]
    if intel_tonemap_fix() and INTEL_TONEMAP not in names:
        names.append(INTEL_TONEMAP)
    if skip_network_choice() and SKIP_NETWORK_CHOICE not in names:
        names.append(SKIP_NETWORK_CHOICE)
    if gx_command_heap() and GX_COMMAND_HEAP not in names:
        names.append(GX_COMMAND_HEAP)
    names=validate_patch_requirements(names,a.game_dir)
    elf=(a.out/'eboot.elf').read_bytes()
    segments=eboot_segments(elf)
    writes=compile_patches(a.xml,names,a.app_version,segments)
    writes+=language_writes(read_settings(a.settings),elf)
    size=render_size(read_settings(a.settings),a.render_res) if a.render_res else None
    # The UI keeps the game's 1920x1080 coordinates even for a larger output: the port draws
    # it into the output-size image with a viewport scaled by output / 1920
    # (UiComposition::NativeViewport), so it is rasterized at the output resolution.
    ui=OUTPUT_SIZE
    if size:
        writes+=resolution_writes(a.xml,size,a.app_version,segments,ui)
    large=a.large_heaps=='1'
    # BB_PATCHES may name the heap patch already: its writes go in once.
    if graphics_heaps(size,large) and GRAPHICS_HEAP_PATCH not in names:
        heaps=graphics_heap_writes(elf,size,full=large)
        if heaps:
            writes+=heaps
            print('Patches: graphics heaps '+', '.join(
                f'{struct.unpack("<Q",data)[0]>>20} MiB' for _,data in heaps)+
                (' (texture pack)' if large else ' for the scene'))
        else:
            writes+=compile_patches(a.xml,[GRAPHICS_HEAP_PATCH],a.app_version,segments)
            names.append(GRAPHICS_HEAP_PATCH)
    if size:
        print(f'Patches: scene {size[0]}x{size[1]}; UI {ui[0]}x{ui[1]}')
    if a.patches_dir:
        # After the built-in ones: an external patch of the same bytes wins.
        writes+=compile_external(external_selection(external_patches(a.patches_dir,a.app_version,a.xml),
                                                    a.patches_config),segments)
    # BBPATCH2: the patch base, so the loader can rebase pointers the patches write into
    # relocated slots (60/90 FPS++ replace function pointers).
    blob=struct.pack('<8sQQ',b'BBPATCH2',EBOOT_BASE,len(writes))
    for offset,data in writes: blob+=struct.pack('<QQ',offset,len(data))+data
    (a.out/'patches.bin').write_bytes(blob)
    print(f'Patches: FPS preset {a.fps}; {len(writes)} writes from {names or "none"}')


if __name__=='__main__':
    main()
