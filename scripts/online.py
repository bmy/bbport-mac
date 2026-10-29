#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Co-op addresses (BB_ONLINE=1): the game's own server, the shadNet server and its WebAPI.

From run.py and the launcher of the Windows port of bbport. Messages,
bloodstains and ghosts come from the community game server; bells and summons go through a shadNet
server. By default both are Seamless Hunt's (play.seamlesshunt.com, BBSC's server): shadPS4's public
shadNet (srv.shadps4.net) accepts only shadPS4's own builds since 2026-10-12.

`online.py bbsc-local` (BB_BBSC=1): prints the shadNet host when BBSC's game server should run on
this PC (scripts/bbsc_gameserver, started by run.sh): BB_BBSC_LOCAL_SERVER=1, or =auto (default)
when the server's own game server does not answer (its ss.info on port 20443 timed out for players
while the rest worked; the local one serves ss.info and sends summons to the shadNet server, as
BBSC's players do). Nothing printed: the server's game server is used.

`online.py overrides --user DIR` writes DIR/host_overrides.json for this launch (the game's
server -> the community server; summon signs -> the shadNet WebAPI when it has the Bloodborne
summon service) and prints its path for SHADPS4_HTTP_HOST_OVERRIDES_JSON. A player who needs
other addresses puts them in DIR/host_overrides.custom.json, which is then used as it is.
"""
import argparse
import json
from pathlib import Path
import os
import socket
import sys
import time
import urllib.error
import urllib.request

COMMUNITY_SERVER = 'http://play.seamlesshunt.com'
SHADNET_SERVER = 'play.seamlesshunt.com:31313'
WEBAPI_SERVER = 'http://play.seamlesshunt.com:31315'
# The defaults before 2026-10-15 (the launcher replaces them in saved settings).
OLD_DEFAULTS = {
    'online_server': ('srv.shadps4.net:31313',),
    'online_community': ('https://thehuntersdream.com', ''),
    'online_webapi': ('',),
}
# Bloodborne's own game server, answered by the community server on the same port.
GAME_SERVER = 'https://ss4.scej-network.jp:20443'
# The server's ss.info sends the game to this port for its API (summons among them).
COMMUNITY_API_PORT = 18671


def community_address(value):
    """The game server as scheme://host (the game picks the ports), Seamless Hunt by default.
    Without a scheme: https, but http for the default server's host (it serves plain http)."""
    value = str(value or '').strip() or COMMUNITY_SERVER
    scheme, _sep, rest = value.rpartition('://')
    host = rest.split('/')[0].split(':')[0]
    default_scheme, _sep, default_host = COMMUNITY_SERVER.partition('://')
    return f'{scheme or (default_scheme if host == default_host else "https")}://{host}'


def webapi_address(server):
    """The shadNet WebAPI of a server address host:port (the server's default port 31315)."""
    server = str(server or '').strip() or SHADNET_SERVER
    host = server.rsplit(':', 1)[0] if server.count(':') == 1 else server
    return f'http://{host}:31315'


def webapi_setting(webapi, server, community):
    """The WebAPI field, unless it holds the game server's address (a common mix-up): then the
    shadNet server's own WebAPI."""
    webapi = str(webapi or '').strip()
    community_host = community_address(community).split('://', 1)[1]
    if webapi and community_host in webapi:
        webapi = ''
    return webapi or webapi_address(server)


def summon_service(webapi):
    """True when a shadNet WebAPI answers Bloodborne's summon calls: the route exists there
    (an empty request is refused with 400), where other servers answer 404."""
    if not webapi:
        return False
    request = urllib.request.Request(webapi.rstrip('/') + '/summon_messenger/create', data=b'',
                                     headers={'User-Agent': 'Bloodborne/01.09'}, method='POST')
    try:
        urllib.request.urlopen(request, timeout=4).close()
        return True
    except urllib.error.HTTPError as error:
        return error.code != 404
    except (OSError, ValueError):
        return False


def write_overrides(user_dir, community, webapi, plain=False):
    """The host overrides file for this launch; returns its path."""
    user_dir = Path(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    custom = user_dir / 'host_overrides.custom.json'
    if custom.exists():
        print(f'Co-op: addresses from {custom}', file=sys.stderr)
        return custom
    community = community_address(community)
    host = community.split('://', 1)[1]
    overrides = {GAME_SERVER: community}
    if plain:
        # The shadPS4 0.20 module: the game server only; summons stay on shadNet.
        print(f'Co-op: game server {community}', file=sys.stderr)
    elif summon_service(webapi):
        overrides.update({f'{scheme}://{host}:{COMMUNITY_API_PORT}/summon_messenger': webapi
                          for scheme in ('https', 'http')})
        print(f'Co-op: summon signs through {webapi}', file=sys.stderr)
    else:
        print(f'Co-op: {webapi} has no summon service; summon signs through {community}',
              file=sys.stderr)
    path = user_dir / 'host_overrides.json'
    path.write_text(json.dumps(overrides, indent=2) + '\n', encoding='utf-8')
    return path


def check(server, webapi, community):
    """(name, address, ok, detail) for the game server, the shadNet server and its WebAPI."""
    results = []
    try:
        game_url = community if (':20443' in community) else f"{community.rstrip('/')}:20443"
        req = urllib.request.Request(f'{game_url}/bb-eu/ss.info',
                                     headers={'User-Agent': 'Bloodborne/01.09'})
        with urllib.request.urlopen(req, timeout=6) as reply:
            reply.read(64)
        results.append(('game', community, True, ''))
    except urllib.error.HTTPError:
        results.append(('game', community, True, ''))  # the server answered
    except (OSError, ValueError) as error:
        results.append(('game', community, False, str(error)))
    host, _sep, port = server.rpartition(':')
    try:
        socket.create_connection((host or server, int(port or 31313)), timeout=4).close()
        results.append(('shadnet', server, True, ''))
    except (OSError, ValueError) as error:
        results.append(('shadnet', server, False, str(error)))
    try:
        req = urllib.request.Request(webapi.rstrip('/') + '/status',
                                     headers={'User-Agent': 'Bloodborne/01.09'})
        with urllib.request.urlopen(req, timeout=4) as reply:
            ok = b'"ok":true' in reply.read(512).replace(b' ', b'')
        results.append(('webapi', webapi, ok, '' if ok else 'not a shadNet server'))
    except (OSError, ValueError) as error:
        results.append(('webapi', webapi, False, str(error)))
    return results


# ---- The Hunter's Dream pool (BB_NET_BACKEND=thd, gpu/bbnet20/thd; bbhost's protocol) ------------

HUNTERS_DREAM_SERVER = 'https://thehuntersdream.com'
# The play logs the game sends to FromSoftware's old storage buckets go to the game server's API.
PLAY_LOG_HOSTS = ('bb-playlog-prod.s3.amazonaws.com', 'bb-playlog-test.s3.amazonaws.com')


def thd_base(value):
    """The Hunter's Dream's address as scheme://host (https unless a scheme is given)."""
    value = str(value or '').strip().rstrip('/') or HUNTERS_DREAM_SERVER
    if '://' not in value:
        value = 'https://' + value
    scheme, rest = value.split('://', 1)
    host_part = rest.split('/')[0]
    return f'{scheme}://{host_part}'


def thd_call(url, body):
    """A call to The Hunter's Dream's account service (bbhost's): its reply, or ValueError with the
    server's reason."""
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST',
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            text = response.read()
    except urllib.error.HTTPError as error:
        text = error.read()
    except (OSError, ValueError) as error:
        raise ValueError(f'{url}: {error}') from error
    try:
        reply = json.loads(text)
    except ValueError:
        reply = None
    if not isinstance(reply, dict) or 'ResKind' not in reply:
        raise ValueError(f'{url} has no account service')
    if reply['ResKind'] != 0:
        raise ValueError(reply.get('Message') or 'refused')
    return reply


def thd_sign_in(server, on_code, device='bbport', cancelled=lambda: False):
    """Discord sign-in the way bbhost does it: a code to approve on the website. on_code(page, code)
    shows it. Returns (online id, token); raises ValueError."""
    auth = thd_base(server)
    start = thd_call(auth + '/auth/device/start', {'DeviceName': device})
    code, user_code = start.get('DeviceCode', ''), start.get('UserCode', '')
    if not code or not user_code:
        raise ValueError('no code in the reply')
    interval = max(1, int(start.get('Interval') or 5))
    deadline = time.monotonic() + int(start.get('ExpiresIn') or 600)
    on_code(start.get('VerificationUri') or auth + '/link', user_code)
    while time.monotonic() < deadline:
        for _ in range(interval * 4):  # a cancel is noticed within a quarter of a second
            if cancelled():
                raise ValueError('cancelled')
            time.sleep(0.25)
        poll = thd_call(auth + '/auth/device/poll', {'DeviceCode': code})
        status = poll.get('Status')
        if status == 'approved':
            if not poll.get('OnlineId') or not poll.get('Token'):
                raise ValueError('no token in the reply')
            return poll['OnlineId'], poll['Token']
        if status == 'slow_down':
            interval += 5
        elif status == 'denied':
            raise ValueError('the sign-in was denied')
        elif status == 'expired':
            break
    raise ValueError('the code expired; try again')


def write_thd_overrides(user_dir, community):
    """The host overrides file for this launch (the game's servers and play logs -> The Hunter's
    Dream); returns (path, base address). host_overrides.custom.json is used as it is."""
    user_dir = Path(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    base = thd_base(community)
    custom = user_dir / 'host_overrides.custom.json'
    if custom.exists():
        print(f'Co-op: addresses from {custom}', file=sys.stderr)
        return custom, base
    overrides = {'*.scej-network.jp': base}
    overrides.update({host: f'{base}:{COMMUNITY_API_PORT}' for host in PLAY_LOG_HOSTS})
    path = user_dir / 'host_overrides.json'
    path.write_text(json.dumps(overrides, indent=2) + '\n', encoding='utf-8')
    print(f"Co-op: The Hunter's Dream {base}", file=sys.stderr)
    return path, base


def check_thd(community):
    """(name, address, ok, detail) for The Hunter's Dream's game server and co-op service."""
    base = thd_base(community)
    results = []
    for name, url in (('game', base + ':20443/bb-eu/ss.info'), ('service', base + f':{COMMUNITY_API_PORT}/')):
        try:
            urllib.request.urlopen(url, timeout=6).close()
            results.append((name, url.split('/')[2], True, ''))
        except urllib.error.HTTPError:
            results.append((name, url.split('/')[2], True, ''))  # it answered
        except (OSError, ValueError) as error:
            results.append((name, url.split('/')[2], False, str(error)))
    return results


def bbsc_local_server():
    """The shadNet host for a local BBSC game server, or '' (see the module's description)."""
    mode = os.environ.get('BB_BBSC_LOCAL_SERVER', 'auto').strip().lower()
    server = os.environ.get('BB_SHADNET_SERVER') or SHADNET_SERVER
    host = server.rsplit(':', 1)[0] if server.count(':') == 1 else server
    if mode in ('0', 'off', 'no'):
        return ''
    if mode in ('1', 'on', 'yes'):
        return host
    community = community_address(os.environ.get('BB_COMMUNITY_SERVER'))
    comm_host = community.split('://', 1)[1].split(':')[0]
    if comm_host in ('127.0.0.1', 'localhost'):
        return ''  # pointed at a local one already
    try:
        game_url = community if (':20443' in community) else f"{community.rstrip('/')}:20443"
        req = urllib.request.Request(f'{game_url}/bb-eu/ss.info',
                                     headers={'User-Agent': 'Bloodborne/01.09'})
        with urllib.request.urlopen(req, timeout=5) as reply:
            reply.read(64)
        return ''
    except urllib.error.HTTPError:
        # An HTTP status reply (even 403 or 401) confirms the server is reachable and running.
        return ''
    except (OSError, ValueError) as error:
        try:
            # Fallback: check if port 20443 is accepting TCP connections
            socket.create_connection((comm_host, 20443), timeout=3).close()
            return ''
        except (OSError, ValueError):
            pass
        print(f'Co-op: {community}:20443 does not answer ({error}): BBSC game server on this PC',
              file=sys.stderr)
        return host


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    overrides = sub.add_parser('overrides')
    overrides.add_argument('--user', required=True)
    overrides.add_argument('--plain', action='store_true')
    sub.add_parser('check')
    thd = sub.add_parser('thd-overrides')
    thd.add_argument('--user', required=True)
    sub.add_parser('thd-check')
    sub.add_parser('thd-signin')
    sub.add_parser('bbsc-local')
    a = parser.parse_args()
    server = os.environ.get('BB_SHADNET_SERVER') or SHADNET_SERVER
    community = community_address(os.environ.get('BB_COMMUNITY_SERVER'))
    webapi = webapi_setting(os.environ.get('BB_SHADNET_WEBAPI'), server, community)
    if a.command == 'thd-overrides':
        path, base = write_thd_overrides(a.user, os.environ.get('BB_COMMUNITY_SERVER'))
        print(path)
        print(base)
        return
    if a.command == 'thd-check':
        for name, address, ok, detail in check_thd(os.environ.get('BB_COMMUNITY_SERVER')):
            print(f'{name}: {address}: {"reachable" if ok else "not reachable"} {detail}'.rstrip())
        return
    if a.command == 'thd-signin':
        try:
            name, token = thd_sign_in(os.environ.get('BB_COMMUNITY_SERVER'),
                                      lambda page, code: print(f'Open {page} and enter {code}',
                                                               file=sys.stderr, flush=True))
        except ValueError as error:
            sys.exit(f'The Hunter\'s Dream sign-in: {error}')
        print(name)
        print(token)
        return
    if a.command == 'bbsc-local':
        print(bbsc_local_server())
        return
    if a.command == 'overrides':
        print(write_overrides(a.user, community, webapi, a.plain))
    else:
        for name, address, ok, detail in check(server, webapi, community):
            print(f'{name}: {address}: {"reachable" if ok else "not reachable"} {detail}'.rstrip())


if __name__ == '__main__':
    main()
