#!/usr/bin/env bash
# tools/macos/update.sh: brings this checkout up to date with GitHub and rebuilds what changed.
#   bash tools/macos/update.sh            # the current branch
#   bash tools/macos/update.sh <branch>   # switch to a test branch (macos-port is the main one)
# Local changes to tracked files are discarded (the branch matches GitHub exactly); bbport.ini,
# saves, deps-x86_64 and out/ are untracked and stay. The Mac app's Update button runs this.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
branch=${1:-$(git branch --show-current)}
[[ -n $branch ]] || { echo "STOP: no branch checked out; name one: bash tools/macos/update.sh macos-port" >&2; exit 1; }
old=$(git rev-parse HEAD)

echo "=== Fetching $branch from GitHub"
# Submodules are pinned and fetched by build.sh when missing; recursing here fails on refs that
# the submodules' own remotes no longer serve.
git fetch --no-recurse-submodules origin
git rev-parse --verify --quiet "origin/$branch" >/dev/null || { echo "STOP: no branch '$branch' on GitHub" >&2; exit 1; }
git checkout -q -f "$branch" 2>/dev/null || git checkout -q -f -b "$branch" "origin/$branch"
git reset -q --hard "origin/$branch"
new=$(git rev-parse HEAD)
if [[ $old == "$new" ]]; then
    echo "Already up to date: $(git log -1 --format='%h %s')"
else
    echo "Updated to: $(git log -1 --format='%h %s')"
    git log --format='  %h %s' "$old..$new" 2>/dev/null | head -20 || true
fi
changed() { [[ $old != "$new" ]] && ! git diff --quiet "$old" "$new" -- "$@" 2>/dev/null; }

# Dependencies only when their recipe changed (setup_deps.sh skips what is built, but it still
# calls Homebrew, which can take a while).
if [[ ! -f deps-x86_64/env.sh ]] || changed tools/macos/setup_deps.sh tools/macos/kosmickrisp-patches; then
    echo "=== Dependencies"
    bash tools/macos/setup_deps.sh
fi

echo "=== Build"
bash tools/macos/build.sh

if [[ ! -d out/bbport.app ]] || changed launcher-macos tools/macos/build_launcher.sh; then
    echo "=== Mac app"
    bash tools/macos/build_launcher.sh
    echo "The Mac app was rebuilt: quit and reopen it to use the new version."
fi
echo "=== Done"
