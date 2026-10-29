#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""FSR 4.1.1 assets on Windows, without Proton: build_assets.sh's Windows counterpart.

    build_assets_win.py <amd_fidelityfx_upscaler_dx12.dll 4.1.x> [loader DLL 2.x]
    build_assets_win.py <amdxcffx64.dll> [upscaler DLL 4.1.x]

The launcher runs it (bbport-run.exe --build-fsr411 DLL [DLL]). fsr4cap.exe records AMD's
upscaler DLL on vkd3d-proton (d3d12.dll, d3d12core.dll: D3D12 over Vulkan, the same on every GPU)
and DXVK's dxgi.dll, with amdxc64.dll (tools/fsr4cap/amdxc64.c) answering the DLL's driver
queries, all next to it in the work folder; dxil-spirv.exe and extract.py translate the passes.
These tools come with the package in tools/fsr4cap/bin (packaging/windows/package.sh). Nothing of
AMD's is shipped: the DLLs are the player's own.

The second form adds FSR 4.1.2's models from AMD's driver (Adrenalin 26.10.1 and later: its
amdxcffx64.dll, in C:\\Windows\\System32\\DriverStore\\FileRepository\\u0*\\): INT8 RDNA2 (navi2/)
and INT8 Lite (lite/), through the upscaler DLL's driver query (FSR4_UPGRADE=1; amdxc64.dll then
hands it to the copy of the driver DLL driver_model.py made next to fsr4cap.exe).

Environment: BB_FSR411_OUT (default <data>/fsr4_411), BB_FSR4CAP_WORK (default
<data>/fsr4cap-work), BB_FSR4CAP_FP8 (auto/1/0, as capture_all.sh), BB_FSR4CAP_SPOOF=0,
BB_FSR4_DRIVER_MODELS ("navi2 lite"), BB_FSR4CAP_CLEAN=1 (work folder removed after a build).
Exit status as build_assets.sh: 2 the DLL is not supported, 4 tools missing, 5 the upscaler did
not run, 6 the captures break the replay rules, 7 the DLL needs a loader and none was found.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = getattr(sys, 'frozen', False)
# bbport-run.exe (PyInstaller) runs the Python scripts itself: bbport-run.exe --run-script X.
PYTHON = [sys.executable, '--run-script'] if FROZEN else [sys.executable]
TOOLS = Path(os.environ.get('BB_FSR4CAP_TOOLS', HERE / 'bin'))
# Next to fsr4cap.exe in the work folder: the recording's D3D12, DXGI and AMD extension modules.
RECORDER_FILES = ['fsr4cap.exe', 'amdxc64.dll', 'd3d12.dll', 'd3d12core.dll', 'dxgi.dll']
SPOOF = 'dxgi.customVendorId = 1002; dxgi.customDeviceId = 747e'  # an RX 7800 XT (issue #186)
SPOOF_RDNA4 = 'dxgi.customVendorId = 1002; dxgi.customDeviceId = 7550'  # an RX 9070 XT
# INT8 recordings: vkd3d-proton's cooperative matrices hidden (proton.sh), as every GPU runs.
INT8_ENV = {'VKD3D_DISABLE_EXTENSIONS':
            'VK_KHR_cooperative_matrix,VK_NV_cooperative_matrix2,VK_EXT_shader_float8'}


def say(text):
    print(text, flush=True)


def data_dir():
    return Path(os.environ.get('BB_DATA_DIR', HERE.parent.parent))


def run_script(name, *args, env=None):
    return subprocess.run([*PYTHON, str(HERE / name), *map(str, args)], env=env).returncode


def copy_in(source, target):
    target = Path(target)
    if target.exists():
        target.chmod(0o666)
        target.unlink()
    shutil.copyfile(source, target)


def sizes():
    """The output sizes and render sizes capture_all.sh records (20)."""
    out = []
    for ow, oh in ((1920, 1080), (2560, 1440), (3840, 2160)):
        for ratio in (1.0, 1.5, 1.7, 2.0, 3.0):
            out.append((f'{int(ow / ratio + 0.5)}x{int(oh / ratio + 0.5)}', f'{ow}x{oh}'))
    out += [('853x533', '1280x800'), ('640x400', '1280x800'), ('1067x600', '1600x900'),
            ('1707x720', '2560x1080'), ('2293x960', '3440x1440')]
    return out


def rdna4_present():
    """An RX 9000 among this PC's GPUs (bb-gpu-capabilities --gpus: index, vendor, kind, name)."""
    tool = HERE.parent.parent / 'bin' / 'bb-gpu-capabilities.exe'
    try:
        listing = subprocess.run([str(tool), '--gpus'], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(word in listing for word in ('RX 9070', 'RX 9060', 'Radeon AI PRO R9700'))


class Recorder:
    def __init__(self, work, version):
        self.work = Path(work)
        self.version = version
        self.extra = {}

    def last_line(self):
        try:
            lines = [l for l in (self.work / 'fsr4cap.log').read_text(errors='replace').splitlines() if l.strip()]
        except OSError:
            return ''
        return lines[-1] if lines else ''

    def capture(self, render, output, variant='int8', fp8_config=''):
        """True when the DLL ran its frames for render -> output."""
        sub = self.work / ('fp8' if variant == 'fp8' else '') / f'capture_{render}_{output}'
        shutil.rmtree(sub, ignore_errors=True)
        (self.work / 'fsr4cap.log').unlink(missing_ok=True)
        env = dict(os.environ, **self.extra)
        env.pop('FSR4CAP_SUBDIR', None)
        if variant == 'fp8':
            env['VKD3D_DISABLE_EXTENSIONS'] = ''
            env['FSR4CAP_SUBDIR'] = 'fp8'
            if fp8_config:
                env['DXIL_SPIRV_CONFIG'] = fp8_config
        else:
            env.update(INT8_ENV)
        with open(self.work / 'run.log', 'w', encoding='utf-8', errors='replace') as log:
            try:
                subprocess.run([str(self.work / 'fsr4cap.exe'), self.version, render, output, '3'],
                               cwd=self.work, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=600)
            except (OSError, subprocess.SubprocessError) as e:
                log.write(f'{e}\n')
        return 'frames done' in self.last_line()


def record_all(work, version, fp8_mode):
    """capture_all.sh on Windows: the INT8 set and, on RDNA4 (or asked for), the FP8 one."""
    rec = Recorder(work, version)
    todo = sizes()
    first = todo[0]
    say('Recording the DLL\'s passes (20 sizes)...')
    if not rec.capture(*first):
        if os.environ.get('BB_FSR4CAP_SPOOF', '1') != '0' and rec.last_line().startswith('no version matching'):
            say('  The DLL did not offer FSR 4.1 to this GPU\'s id; again under an RX 7800 XT\'s id')
            rec.extra['DXVK_CONFIG'] = SPOOF
            if not rec.capture(*first):
                rec.extra.pop('DXVK_CONFIG')
        if 'frames done' not in rec.last_line():
            say(f'FSR 4.1 did not start: {rec.last_line() or "no output"} (log: {work}\\run.log)')
            return 5
    done = 1
    say(f'[{done:2d}/20] {first[1]:<9} <- {first[0]:<9} {rec.last_line()}')
    fp8 = False
    shutil.rmtree(Path(work) / 'fp8', ignore_errors=True)
    if fp8_mode != '0':
        saved = dict(rec.extra)
        if rec.extra.get('DXVK_CONFIG') == SPOOF and rdna4_present():
            rec.extra['DXVK_CONFIG'] = SPOOF_RDNA4
        config = 'wmma_rdna3_workaround' if fp8_mode == '1' else ''

        def shaders(d):
            return sorted(p.name for p in Path(d).glob('cs_*.dxil'))
        if rec.capture(*first, variant='fp8', fp8_config=config) and \
                shaders(Path(work) / 'fp8' / f'capture_{first[0]}_{first[1]}') != \
                shaders(Path(work) / f'capture_{first[0]}_{first[1]}'):
            fp8 = True
            say('FP8 variant (RDNA4): recorded too')
        else:
            shutil.rmtree(Path(work) / 'fp8', ignore_errors=True)
            say('FP8 variant: not offered on this GPU (RDNA4 only), INT8 only')
        rec.extra = saved
    total = 40 if fp8 else 20
    for render, output in todo[1:]:
        if not rec.capture(render, output):
            say(f'The FSR 4.1.1 upscaler stopped running: {rec.last_line()} (log: {work}\\run.log)')
            return 5
        done += 1
        say(f'[{done:2d}/{total}] {output:<9} <- {render:<9} {rec.last_line()}')
    if fp8:
        for render, output in todo[1:]:
            if not rec.capture(render, output, variant='fp8', fp8_config=config):
                say('The FP8 variant stopped running: INT8 only.')
                shutil.rmtree(Path(work) / 'fp8', ignore_errors=True)
                break
            done += 1
            say(f'[{done:2d}/{total}] {output:<9} <- {render:<9} FP8: {rec.last_line()}')
    return 0


def replace_set(new, out):
    """The new set in place of the old one; FSR 4.1.2's driver models (navi2/, lite/) stay."""
    old = Path(str(out) + '.old')
    shutil.rmtree(old, ignore_errors=True)
    if out.exists():
        out.rename(old)
    Path(new).rename(out)
    for model in ('navi2', 'lite'):
        if (old / model).is_dir():
            shutil.move(str(old / model), str(out / model))
    shutil.rmtree(old, ignore_errors=True)


def main(argv):
    if not 1 <= len(argv) <= 2:
        say(__doc__.split('\n\n')[1])
        return 1
    first = Path(argv[0]).resolve()
    second = Path(argv[1]).resolve() if len(argv) == 2 else None
    work = Path(os.environ.get('BB_FSR4CAP_WORK', data_dir() / 'fsr4cap-work'))
    out = Path(os.environ.get('BB_FSR411_OUT', data_dir() / 'fsr4_411'))
    missing = [name for name in RECORDER_FILES + ['dxil-spirv.exe', 'spirv-as.exe', 'spirv-dis.exe']
               if not (TOOLS / name).is_file()]
    if missing:
        say(f'Tools missing in {TOOLS}: {", ".join(missing)} (the package brings them)')
        return 4
    driver = None
    if first.name.lower() in ('amdxcffx64.dll', 'amdvkffx64.dll'):
        driver = first.with_name('amdxcffx64.dll')
        if not driver.is_file():
            say('amdvkffx64.dll is the driver\'s Vulkan module; the recording needs its D3D12 one, '
                'amdxcffx64.dll, from the same driver folder.')
            return 2
        upscaler = second or work / 'amd_fidelityfx_upscaler_dx12.dll'
        loader = None
        if not upscaler.is_file():
            say('FSR 4.1.2 from the driver also needs AMD\'s FSR 4.1.x upscaler DLL '
                '(amd_fidelityfx_upscaler_dx12.dll): give it as the second file, or build FSR 4.1.1 '
                'from it first.')
            return 2
    else:
        upscaler, loader = first, second
    # The DLL first: is it one bbport replays, and does it need a loader?
    sys.path.insert(0, str(HERE))
    import dll_info
    status, problem, details = dll_info.inspect(str(upscaler), str(loader) if loader else None)
    say(f'Upscaler DLL: version {details.get("version", "?")}, models: '
        f'{", ".join(details.get("models", [])) or "none"}')
    if problem:
        say(f'Not usable: {problem}')
        return status
    loader = details.get('loader') or loader

    work.mkdir(parents=True, exist_ok=True)
    for name in RECORDER_FILES:
        copy_in(TOOLS / name, work / name)
    target = work / 'amd_fidelityfx_upscaler_dx12.dll'
    if not (target.exists() and os.path.samefile(upscaler, target)):  # the driver form may name it
        copy_in(upscaler, target)
    (work / 'amd_fidelityfx_loader_dx12.dll').unlink(missing_ok=True)
    if loader:
        copy_in(loader, work / 'amd_fidelityfx_loader_dx12.dll')
    # extract.py runs spirv-dis / spirv-as by name.
    env = dict(os.environ, PATH=str(TOOLS) + os.pathsep + os.environ.get('PATH', ''))
    dxil_spirv = TOOLS / 'dxil-spirv.exe'

    if driver:
        models = os.environ.get('BB_FSR4_DRIVER_MODELS', 'navi2 lite').split()
        if not (out / 't1080_m0' / 'initializer.bin').is_file():
            models.insert(0, 'int8')
        if run_script('driver_model.py', 'info', driver) != 0:
            say(f'{driver} does not hold FSR 4.1.2\'s models (Adrenalin 26.10.1 or later is needed)')
            return 2
        d = work / 'driver'
        d.mkdir(exist_ok=True)
        for name in RECORDER_FILES + ['amd_fidelityfx_upscaler_dx12.dll']:
            copy_in(work / name, d / name)
        driver_out = work / 'driver-out'
        shutil.rmtree(driver_out, ignore_errors=True)
        os.environ['FSR4_UPGRADE'] = '1'
        try:
            for model in models:
                say(f'FSR 4.1.2 {model}: recording the driver\'s passes (20 sizes)...')
                for old in list(d.glob('capture_*')) + [d / 'fp8']:
                    shutil.rmtree(old, ignore_errors=True)
                if run_script('driver_model.py', 'force', driver, d / 'amdxcffx64.dll', model) != 0:
                    return 2
                status = record_all(d, '4.1.2', '0')
                if status:
                    return status
                say(f'FSR 4.1.2 {model}: translating the passes to SPIR-V...')
                status = run_script('extract.py', dxil_spirv, d, driver_out,
                                    env=dict(env, FSR4CAP_MODEL=model))
                if status:
                    return 6
        finally:
            os.environ.pop('FSR4_UPGRADE', None)
        for model in models:
            source = driver_out if model == 'int8' else driver_out / model
            target = out if model == 'int8' else out / model
            target.mkdir(parents=True, exist_ok=True)
            for sets in source.glob('t*_m*'):
                shutil.rmtree(target / sets.name, ignore_errors=True)
                shutil.move(str(sets), str(target / sets.name))
        shutil.rmtree(driver_out, ignore_errors=True)
        say(f'FSR 4.1.2 models ({" ".join(models)}) in {out} (the menu\'s FSR 4 model)')
        return 0

    status = record_all(work, os.environ.get('FSR4CAP_VERSION', '4.1.'),
                        os.environ.get('BB_FSR4CAP_FP8', 'auto'))
    if status:
        return status
    say('Translating the passes to SPIR-V...')
    new = Path(str(out) + '.new')
    shutil.rmtree(new, ignore_errors=True)
    if run_script('extract.py', dxil_spirv, work, new, env=env) != 0:
        return 6
    replace_set(new, out)
    if os.environ.get('BB_FSR4CAP_CLEAN') == '1':
        shutil.rmtree(work, ignore_errors=True)
    say(f'FSR 4.1.1 assets in {out} (upscaler FSR 4.1.1)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
