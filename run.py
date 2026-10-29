#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Starts the game on Windows (run.sh is the Linux one; this follows it step for step).

The game's files are prepared (scripts/prepare.py, link_libc.py, link_modules.py,
content_profile.py, patches.py) into out/, then bin/bb-probe.exe runs the original eboot. Everything
the port writes lives next to it (portable): bbport.ini, user/ (saves), out/, logs/.

Environment: BB_GAME_DIR (the game's folder, with eboot.bin), BB_DATA_DIR (writable folder for the
port's files; default: this folder), BB_FPS (uncap, 90, 60, 30), BB_SAVE_LOG=1 (frame statistics
in logs/ too), and the BB_* variables of run.sh. The launcher (launcher/bbport_launcher_win.py)
calls run() in its own process when frozen into Bloodborne.exe, so no Python is installed.
"""
import contextlib
import io
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

PORT_DIR = Path(getattr(sys, '_MEIPASS', '')) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
# A frozen launcher keeps the scripts next to Bloodborne.exe (the package's folder).
PACKAGE_DIR = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else PORT_DIR
SCRIPTS = PACKAGE_DIR / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def script(name, *args):
    """Runs scripts/<name> with args; its standard output (some print a result) is returned."""
    path = SCRIPTS / name
    if getattr(sys, 'frozen', False):
        out = io.StringIO()
        argv = sys.argv
        sys.argv = [str(path), *map(str, args)]
        try:
            with contextlib.redirect_stdout(out):
                runpy.run_path(str(path), run_name='__main__')
        except SystemExit as stop:
            if stop.code not in (None, 0):
                raise RuntimeError(f'{name} failed ({stop.code}): {out.getvalue().strip()}') from None
        finally:
            sys.argv = argv
        text = out.getvalue()
    else:
        done = subprocess.run([sys.executable, str(path), *map(str, args)], stdout=subprocess.PIPE,
                              text=True, encoding='utf-8')
        text = done.stdout
        if done.returncode != 0:
            raise RuntimeError(f'{name} failed ({done.returncode}): {text.strip()}')
    for line in text.splitlines():
        if line.strip():
            log(f'  {line}')
    return text


_log_file = None


def log(line):
    if sys.stdout:  # a windowed Bloodborne.exe has none
        print(line, flush=True)
    if _log_file:
        _log_file.write(line + '\n')
        _log_file.flush()


def setdefault(env, name, value):
    if not env.get(name):
        env[name] = value


def online(env, user, bindir):
    """Online play (BB_ONLINE=1, the launcher's Online section), as run.sh sets it up: the online
    module online/bbnet20.dll (BB_NET_MODULE overrides it) and the addresses of the pool: Seamless
    Hunt (the shadNet server and its WebAPI, the game server in host_overrides.json) or The
    Hunter's Dream (BB_NET_BACKEND=thd: the account from the launcher's Discord sign-in, its
    overrides). BBSC's local game server fallback is Linux-only for now."""
    if env.get('BB_ONLINE') != '1':
        return
    module = Path(env.get('BB_NET_MODULE') or PACKAGE_DIR / 'online' / 'bbnet20.dll')
    if not module.is_file():
        log(f'Online: no {module}: offline play')
        env.pop('BB_ONLINE', None)
        return
    env['BB_NET_MODULE'] = str(module)
    if env.get('BB_NET_BACKEND') == 'thd':
        env.pop('BB_BBSC', None)
        if not env.get('BB_THD_TOKEN') or not env.get('BB_THD_ONLINE_ID'):
            raise RuntimeError("The Hunter's Dream: not signed in (launcher: Online -> Sign in with Discord)")
        os.environ.update(env)
        lines = script('online.py', 'thd-overrides', '--user', user).strip().splitlines()
        if len(lines) < 2:
            raise RuntimeError("The Hunter's Dream: no host overrides")
        env['SHADPS4_HTTP_HOST_OVERRIDES_JSON'] = lines[-2].strip()
        base = lines[-1].strip()
        setdefault(env, 'BB_THD_SERVER', base + ':18671')
        setdefault(env, 'BB_THD_AUTH', base)
        for name in ('BB_SHADNET_SERVER', 'BB_SHADNET_WEBAPI', 'BB_SHADNET_NPID', 'BB_SHADNET_PASSWORD',
                     'BB_SHADNET_TOKEN'):
            env.pop(name, None)
            os.environ.pop(name, None)
        log(f"Online: {module.name}, The Hunter's Dream {env['BB_THD_SERVER']} as {env['BB_THD_ONLINE_ID']}")
        return
    setdefault(env, 'BB_SHADNET_SERVER', 'play.seamlesshunt.com:31313')
    setdefault(env, 'BB_SHADNET_WEBAPI', 'http://' + env['BB_SHADNET_SERVER'].rsplit(':', 1)[0] + ':31315')
    os.environ.update(env)
    if env.get('BB_BBSC') == '1':
        bbsc_local_server(env, user)
    try:
        path = script('online.py', 'overrides', '--user', user).strip().splitlines()
    except RuntimeError as error:
        log(f'Online: no host overrides ({error})')
        path = []
    if path:
        env['SHADPS4_HTTP_HOST_OVERRIDES_JSON'] = path[-1].strip()
    log(f'Online: {module.name}, shadNet {env["BB_SHADNET_SERVER"]}, WebAPI {env["BB_SHADNET_WEBAPI"]}'
        + (', BBSC' if env.get('BB_BBSC') == '1' else ''))


_bbsc_server = None
_mod_games = []


def bbsc_local_server(env, user):
    """BBSC: its game server on this PC when the server's own does not answer (online.py
    bbsc-local; run.sh does the same): scripts/bbsc_gameserver run by this interpreter (the frozen
    bbport-run.exe --run-script, or Python), for as long as the game runs."""
    global _bbsc_server
    server_script = SCRIPTS / 'bbsc_gameserver' / 'bbsc_gameserver.py'
    if not server_script.is_file():
        return
    try:
        shadnet = script('online.py', 'bbsc-local').strip()
    except RuntimeError:
        shadnet = ''
    if not shadnet:
        return
    import socket
    try:
        socket.create_connection(('127.0.0.1', 20443), timeout=1).close()
        log('Co-op: a game server answers on 127.0.0.1:20443 already: used')
    except OSError:
        folder = Path(user) / 'bbsc'
        (folder / 'logs').mkdir(parents=True, exist_ok=True)
        if getattr(sys, 'frozen', False):
            command = [sys.executable, '--run-script', str(server_script)]
        else:
            command = [sys.executable, str(server_script)]
        command += ['--shadnet', shadnet, '--host', '127.0.0.1', '--port', '20443', '--relay-port', '0',
                    '--no-bbhost', '--no-playlog', '--data', str(folder / 'online-data.json'),
                    '--log-dir', str(folder / 'logs')]
        console = open(folder / 'logs' / 'gameserver-console.log', 'w', encoding='utf-8', errors='replace')
        try:
            _bbsc_server = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT,
                                            cwd=str(server_script.parent), creationflags=NO_WINDOW)
        except OSError as error:
            log(f'Co-op: the local BBSC game server did not start ({error})')
            return
        finally:
            console.close()
        log(f'Co-op: BBSC game server on 127.0.0.1:20443 (summons -> {shadnet}:31315)')
    env['BB_COMMUNITY_SERVER'] = 'http://127.0.0.1'
    os.environ['BB_COMMUNITY_SERVER'] = env['BB_COMMUNITY_SERVER']


def stop_bbsc_local_server():
    global _bbsc_server
    if _bbsc_server is not None:
        with contextlib.suppress(OSError):
            _bbsc_server.terminate()
        with contextlib.suppress(Exception):
            _bbsc_server.wait(timeout=5)
        _bbsc_server = None


def large_scene_memory(env, scene):
    """More direct memory only for the game's larger heaps (a scene above 1920x1080, see run.sh)."""
    width, _, height = scene.partition('x')
    if width.isdigit() and height.isdigit() and int(width) * int(height) > 1920 * 1080:
        setdefault(env, 'BB_DMEM_MB', '9152')


def run(extra_args=(), env=None, on_line=None):
    """Prepares and runs the game; the exit code. on_line(text) gets every line of output."""
    global _log_file
    env = dict(os.environ if env is None else env)
    # A test build's diagnostics: test-env.txt next to the package, KEY=VALUE lines (# comments),
    # under what the player set.
    test_env = PACKAGE_DIR / 'test-env.txt'
    test_keys = []
    if test_env.is_file():
        for line in test_env.read_text(encoding='utf-8', errors='replace').splitlines():
            key, sep, value = line.strip().partition('=')
            if sep and not key.startswith('#'):
                test_keys.append(key.strip())
                setdefault(env, key.strip(), value.strip())
    data = Path(env.get('BB_DATA_DIR') or PACKAGE_DIR)
    out = data / 'out'
    out.mkdir(parents=True, exist_ok=True)
    setdefault(env, 'BB_CONFIG', str(data / 'bbport.ini'))
    user = Path(env.get('BB_USER_DIR') or data / 'user')
    setdefault(env, 'BB_GPU_USER_DIR', str(user))
    # Every run is logged on Windows (an issue report needs it): logs/<time>.log.
    logs = data / 'logs'
    logs.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    _log_file = open(logs / f'{stamp}.log', 'w', encoding='utf-8', errors='replace')
    if env.get('BB_SAVE_LOG') == '1':
        env['BB_FRAME_STATS'] = '1'
        setdefault(env, 'BB_FRAME_LOG', str(logs / f'{stamp}.frames.csv'))
    elif sys.platform == 'win32':
        # Windows test builds: the 10 s summaries, a per-frame CSV and slow image creations
        # (20 ms and more) are always logged; the hot paths only read the clock every 1024
        # draws. BB_FRAME_STATS=off (or BB_SAVE_LOG=0) leaves them out.
        if env.get('BB_FRAME_STATS') != 'off' and env.get('BB_SAVE_LOG') != '0':
            setdefault(env, 'BB_FRAME_STATS', '1')
            setdefault(env, 'BB_FRAME_LOG', str(logs / f'{stamp}.frames.csv'))
            setdefault(env, 'BB_SLOW_BIND_LOG', '20000')
        elif env.get('BB_FRAME_STATS') == 'off':
            del env['BB_FRAME_STATS']
    if sys.platform == 'win32':
        # The FSR 4 v07 set the package brings, and the FSR 4.1.1 set in the data folder (the
        # launcher copies a built one there), as run.sh looks them up: the game runs in the data
        # folder, which BB_DATA_DIR may move away from the package.
        if (PACKAGE_DIR / 'fsr4_shaders').is_dir():
            setdefault(env, 'BB_FSR4_DIR', str(PACKAGE_DIR / 'fsr4_shaders'))
        if (data / 'fsr4_411').is_dir():
            setdefault(env, 'BB_FSR411_DIR', str(data / 'fsr4_411'))
    log(f'Log: {logs / f"{stamp}.log"}')
    if (PACKAGE_DIR / 'VERSION').is_file():  # issue #307
        log(f"bbport {(PACKAGE_DIR / 'VERSION').read_text().strip()}")
    if test_keys:
        log(f'Test settings ({test_env.name}): ' + ' '.join(f'{key}={env[key]}' for key in test_keys))
    try:
        game = Path(env.get('BB_GAME_DIR') or PACKAGE_DIR.parent / 'CUSA03173')
        if not (game / 'eboot.bin').is_file():
            log(f'No eboot.bin in {game} (choose the game folder in the launcher).')
            return 1
        original_game = game  # the game's own files (shader recipes read program code there)
        os.environ.update(env)  # the scripts read BB_* from the environment
        game = Path(script('mods.py', game, '--out', out, '--mods-dir', env.get('BB_MODS_DIR', data / 'mods'),
                           '--config', env.get('BB_MODS_CONFIG', data / 'mods.json'),
                           '--enabled', env.get('BB_MODS_ENABLED', '1'),
                           '--saves', user / 'savedata', '--state', data / 'mods.last').strip().splitlines()[-1])
        if game != original_game and game.name.startswith('mod-game-'):
            _mod_games.append(game)  # the game with the mods' files linked in: removed at the end
        script('prepare.py', game, '--out', out)
        script('link_libc.py', game, '--out', out)
        script('link_modules.py', game, '--out', out)
        script('content_profile.py', game, '--out', out, '--sku', env.get('BB_CONTENT_SKU', 'full'))
        if env.get('BB_AUTO_RENDER_RES') == '1':
            for name in ('BB_RENDER_RES', 'BB_OUTPUT_RES', 'BB_AUTO_RENDER_RES'):
                env.pop(name, None)
        fps = env.get('BB_FPS', 'uncap')
        bindir = PACKAGE_DIR / 'bin'
        caps = bindir / 'bb-gpu-capabilities.exe'
        scaled_render = scaled_output = ''
        ini_lines = []
        if Path(env['BB_CONFIG']).is_file():
            ini_lines = Path(env['BB_CONFIG']).read_text(encoding='utf-8', errors='replace').splitlines()
        ini = dict(line.split('=', 1) for line in ini_lines if '=' in line and not line.startswith('#'))
        # output_res=auto (and no key, the default): the monitor's resolution (BB_DISPLAY: its name or
        # number, else the primary one).
        if ini.get('output_res', 'auto') == 'auto' and not env.get('BB_MONITOR_SIZE'):
            try:
                rows = subprocess.run([str(caps), '--displays'], stdout=subprocess.PIPE, text=True, env=env,
                                      creationflags=NO_WINDOW).stdout.splitlines()
            except OSError:
                rows = []
            wanted = env.get('BB_DISPLAY', '')
            for number, row in enumerate(rows, 1):
                parts = row.split('\t')
                if len(parts) == 3 and ((wanted and wanted in (parts[0], str(number))) or
                                        (not wanted and parts[2] == '1')):
                    env['BB_MONITOR_SIZE'] = parts[1]
            if env.get('BB_MONITOR_SIZE'):
                log(f'Output: the monitor\'s resolution, {env["BB_MONITOR_SIZE"]} (output_res=auto)')
        # patches.py reads BB_MONITOR_SIZE from the environment: the output stayed 1920x1080 on any
        # other monitor, stretched over an ultrawide one (3440x1440, 2026-10-27).
        os.environ.update(env)
        if not env.get('BB_RENDER_RES'):
            printed = script('patches.py', '--print-scaled', '--settings', env['BB_CONFIG']).split()
            if len(printed) == 2:
                scaled_render, scaled_output = printed
        live = '0'
        # 1080p output: the live path always ran there; over the startup patch (BB_LIVE_HYBRID) a
        # preset below Native AA gets the game's targets at its size.
        if not scaled_output and not env.get('BB_RENDER_RES') and env.get('BB_LIVE_HYBRID', '1') == '1':
            printed = script('patches.py', '--print-scaled', '--align', '8', '--settings', env['BB_CONFIG']).split()
            if len(printed) == 2:
                scaled_render, scaled_output = printed
                env['BB_LIVE_RES'] = '1'
        if scaled_output:
            # Always on (free over the startup patch); BB_LIVE_RES=0 (tests): the patch alone.
            live = '1' if env.get('BB_LIVE_RES', '1') == '1' else '0'
        hybrid = live == '1' and env.get('BB_LIVE_HYBRID', '1') == '1'
        if hybrid:
            # Live preset changes over the startup patch (see run.sh), the scene a multiple of 8.
            scaled_render, scaled_output = script('patches.py', '--print-scaled', '--align', '8', '--settings',
                                                  env['BB_CONFIG']).split()
            env.update(BB_RENDER_RES=scaled_render, BB_OUTPUT_RES=scaled_output, BB_AUTO_RENDER_RES='1',
                       BB_LIVE_HYBRID='1')
            large_scene_memory(env, scaled_render)
            log(f'Output {scaled_output}: scene {scaled_render}, live preset changes over it')
        elif live == '1':
            log(f'Output {scaled_output}: live resolution changes (BB_LIVE_HYBRID=0: proxies only)')
        elif scaled_output:
            env.update(BB_RENDER_RES=scaled_render, BB_OUTPUT_RES=scaled_output, BB_AUTO_RENDER_RES='1')
            large_scene_memory(env, scaled_render)
            log(f'Output {scaled_output}: scene {scaled_render}, direct memory {env.get("BB_DMEM_MB") or 5056} MiB')
        # Texture packs fill the game's graphics heaps (#240, see run.sh): the full larger heaps
        # and their direct memory. BB_LARGE_HEAPS=1/0 forces it.
        large_heaps = env.get('BB_LARGE_HEAPS', '')
        pack = ''
        if large_heaps not in ('0', '1'):
            try:
                found = script('game_check.py', '--texture-pack', game, out / 'texture-pack.json').split(maxsplit=1)
            except RuntimeError:
                found = []
            large_heaps = found[0] if found and found[0] == '1' else '0'
            pack = found[1].strip() if len(found) > 1 else ''
        if large_heaps == '1':
            setdefault(env, 'BB_DMEM_MB', '9152')
            log(f'Graphics heaps: larger ({pack or "BB_LARGE_HEAPS=1"}), direct memory {env["BB_DMEM_MB"]} MiB '
                '(BB_LARGE_HEAPS=0: the game\'s own sizes)')
        os.environ.update(env)
        script('patches.py', '--out', out, '--fps', fps, '--extra', env.get('BB_PATCHES', ''),
               '--settings', env['BB_CONFIG'], '--game-dir', game, '--render-res', env.get('BB_RENDER_RES', ''),
               '--output-res', env.get('BB_OUTPUT_RES', ''), '--patches-dir', env.get('BB_PATCHES_DIR', data / 'patches'),
               '--patches-config', env.get('BB_PATCHES_CONFIG', data / 'patches.json'), '--large-heaps', large_heaps)
        setdefault(env, 'BB_COPY_GPU_BUFFERS', '1')
        setdefault(env, 'BB_GPU_WRITE_TWINS', '1')
        setdefault(env, 'BB_GPU_WRITE_TWINS_MAX', '65536')
        setdefault(env, 'BB_VBLANK_HZ', {'uncap': '1000', '90': '90'}.get(fps, '60'))
        setdefault(env, 'BB_FPS', fps)
        # Shader recipes (gpu/render/shader_rebuild.h): the shaders of the recorded game, translated
        # for this GPU by a background process at start, the code read from the game's own files.
        if 'BB_SHADER_RECIPES' not in env and (PACKAGE_DIR / 'shader_recipes.bin').is_file():
            env['BB_SHADER_RECIPES'] = str(PACKAGE_DIR / 'shader_recipes.bin')
        # Image shapes (gpu/textures/image_spares.h): spare textures made ahead from the first session on.
        if 'BB_IMAGE_SHAPES' not in env and (PACKAGE_DIR / 'image_shapes.txt').is_file():
            env['BB_IMAGE_SHAPES'] = str(PACKAGE_DIR / 'image_shapes.txt')
        setdefault(env, 'BB_SHADER_GAME_DIR', str(original_game))
        online(env, user, bindir)
        probe = Path(env.get('BB_PROBE') or bindir / 'bb-probe.exe')
        args = [str(probe), str(out / 'boot-linked.bin'), '--content-profile', str(out / 'content.bin'),
                '--patches', str(out / 'patches.bin'), '--app0', str(game), '--user', str(user),
                '--timeout', env.get('BB_TIMEOUT', '0'), *extra_args]
        log('Starting ' + ' '.join(args))
        with subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd=str(data), creationflags=NO_WINDOW) as game_process:
            for raw in game_process.stdout:
                line = raw.decode('utf-8', 'replace').rstrip('\r\n')
                log(line)
                if on_line:
                    on_line(line)
            code = game_process.wait()
        log(f'— the game exited (code {code}) —')
        return code
    except (OSError, RuntimeError) as error:
        log(f'STOP: {error}')
        return 1
    finally:
        stop_bbsc_local_server()
        # As run.sh: the linked game folder goes (links and junctions only, never what they point to).
        while _mod_games:
            shutil.rmtree(_mod_games.pop(), ignore_errors=True)
        _log_file.close()
        _log_file = None


NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

if __name__ == '__main__':
    sys.exit(run(sys.argv[1:]))
