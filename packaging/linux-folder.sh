#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
# Builds dist/Bloodborne-bbport-<version>-linux-x86_64.tar.zst: the AppImage's content as a folder
# (run packaging/appimage.sh first), for players who want the files open:
#   Bloodborne.sh          opens the launcher (arguments go through: --play starts the game)
#   online/libbbnet20.so   the online module (shadNet, BBSC), outside the bundle: a newer build of it
#                          (BBSC's) is dropped in here; Bloodborne.sh hands it over in BB_NET_MODULE
#   app/                   the AppImage unpacked (its AppRun mounts the bundled /nix/store)
# Data (saves, settings, the shader cache) stay in ~/.local/share/bbport, as with the AppImage.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
appimage=dist/Bloodborne-bbport-x86_64.AppImage
[[ -f $appimage ]] || { echo "Build the AppImage first: bash packaging/appimage.sh" >&2; exit 1; }
version=$(tr -d '[:space:]' < VERSION)
name=Bloodborne-bbport-$version-linux-x86_64
work=$(mktemp -d)
trap 'chmod -R u+w "$work" 2>/dev/null; rm -rf "$work"' EXIT
(cd "$work" && "$OLDPWD/$appimage" --appimage-extract > /dev/null)
folder=$work/$name
mkdir -p "$folder/online"
mv "$work/squashfs-root" "$folder/app"
# The online module out of the bundle (one copy: the launcher and run.sh take BB_NET_MODULE).
module=$(find "$folder/app/nix/store" -path '*/share/bbport/bin/gpu/libbbnet20.so' -print -quit)
if [[ -n $module ]]; then
    chmod u+w "$(dirname "$module")"
    mv "$module" "$folder/online/libbbnet20.so"
else
    echo "No libbbnet20.so in the AppImage: the folder plays offline only" >&2
fi
cat > "$folder/Bloodborne.sh" <<'SH'
#!/bin/sh
# Bloodborne (bbport): opens the launcher; "Bloodborne.sh --play" starts the game with its settings.
here=$(cd -- "$(dirname -- "$0")" && pwd)
if [ -z "${BB_NET_MODULE:-}" ] && [ -f "$here/online/libbbnet20.so" ]; then
    BB_NET_MODULE=$here/online/libbbnet20.so
    export BB_NET_MODULE
fi
exec "$here/app/AppRun" "$@"
SH
chmod 755 "$folder/Bloodborne.sh"
cat > "$folder/README.txt" <<'TXT'
Bloodborne (bbport) for Linux, as a folder

You need your own dump of Bloodborne (CUSA03173, version 1.09) and a GPU with Vulkan 1.3.

1. Unpack this archive anywhere: tar -xf Bloodborne-bbport-*-linux-x86_64.tar.zst
2. Start Bloodborne.sh (double-click it, or ./Bloodborne.sh in a terminal), choose the game folder
   (the one with eboot.bin), press Play. ./Bloodborne.sh --play starts the game at once.
3. In the game: Insert (or L3+R3) opens the port's settings menu.

Saves, settings and the shader cache are in ~/.local/share/bbport (the same as the AppImage's).

Online play: the launcher's Online page. The online module is online/libbbnet20.so: to try another
build of it (a newer BBSC), replace that file (BB_NET_MODULE=<path> points to one elsewhere).

-----------------------------------------------------------------------------------------------
Bloodborne (bbport) для Linux, папкой

Нужен свой дамп Bloodborne (CUSA03173, версия 1.09) и видеокарта с Vulkan 1.3.

1. Распакуйте архив в любую папку: tar -xf Bloodborne-bbport-*-linux-x86_64.tar.zst
2. Запустите Bloodborne.sh (двойной щелчок или ./Bloodborne.sh в терминале), выберите папку игры
   (где лежит eboot.bin), нажмите «Играть». ./Bloodborne.sh --play сразу запускает игру.
3. В игре: Insert (или L3+R3) открывает меню настроек порта.

Сохранения, настройки и кэш шейдеров — в ~/.local/share/bbport (те же, что у AppImage).

Игра по сети: страница лаунчера «Онлайн». Модуль онлайна — online/libbbnet20.so: чтобы попробовать
другую его сборку (новый BBSC), замените этот файл (BB_NET_MODULE=<путь> укажет файл в другом месте).
TXT
mkdir -p dist
rm -f "dist/$name.tar.zst"
tar -C "$work" -I 'zstd -19 -T0' -cf "dist/$name.tar.zst" "$name"
ls -lh "dist/$name.tar.zst"
