#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
# press.sh <tokens> [hold_seconds]
f=$(dirname -- "$(realpath -- "$0")")/../out/pad
echo "$1" > $f; sleep ${2:-0.3}; : > $f
