# Security review: every upstream release

bbport is not trusted software: it runs the game's code inside its own process, loads mods and
third-party patches, and builds from sources and dependencies we don't control. So every time an
upstream bbport release is ported, and every time a pinned dependency changes, the code is
reviewed for security **before anything is pushed or released**.

## When

- A new upstream bbport release or pre-release merged into a `macos-*` branch.
- A dependency pin changed in `tools/macos/setup_deps.sh` (a tag and its commit), the
  KosmicKrisp revision (`KK_REV`), xbyak, the pinned Python modules, or a submodule.
- Before any `release/*` push, if the code changed since the last review.

## How

Two reviews, done independently (separate reviewers or separate agent runs that haven't seen
the work being reviewed), so the work isn't grading itself:

1. **Upstream.** The diff from the previously ported upstream commit to the new one
   (`git diff <old>..<new>` in deadinside28/bloodborne_pc), plus any new files whole.
2. **Ours.** Everything the Mac port adds or changes: the merge's conflict resolutions, our
   commits since the last review, and the parts the upstream change touches (the native GPU
   process's channel and memory, `tools/macos/`, the launcher, the CI workflow).

Each review lists findings as Critical / High / Medium / Low / Informational, with the file and
line, what could go wrong, and a suggested fix. Anything that looks deliberately malicious
(obfuscated code, network access that wasn't there, writes outside the data folder, downloads
without a checksum, code run from data files) is reported at once, whatever its size.

## What to look at

**Upstream code**
- New network access, downloads, or anything fetched at run time or build time; whether it's
  pinned and checked (hashes, commits).
- Code loaded or run from outside the build: `dlopen`, scripts, patch files, mod files,
  shader or pipeline caches; how their input is parsed (bounds, sizes, offsets).
- Files written: paths built from game data or settings, temporary files (`O_EXCL`,
  `O_NOFOLLOW`), symlinks, permissions, anything outside the data and saves folders.
- Process launching: shells, `system`, environment variables passed on, arguments built from
  settings.
- Memory: new mappings with execute permission, signal handlers, fault handlers, shared memory.
- Build scripts and CMake: new `FetchContent`, `ExternalProject`, downloads, submodules.

**Our code**
- The native GPU process: everything read from the other process (channel headers, payloads,
  reply sizes, addresses and sizes in map/unmap/protect/trap messages) is checked before use;
  descriptors passed to `bb-gpu`; no mapping over the control block.
- `tools/macos/*.sh`: quoting, `rm -rf` targets, paths in logs (`~`, not the home folder),
  dependency pins and their verification.
- The launcher: what it runs, its environment, quarantine handling, paths it writes.
- `.github/workflows/release.yml`: the build job stays read-only and keeps no token; only the
  publish job writes; versions are checked before use.

## Known gaps

What the port doesn't pin or check yet, to keep in mind in every review:
- Homebrew's build tools (cmake, meson, LLVM, glslang and others in `setup_deps.sh`) are
  whatever Homebrew has that day; the Python modules for Mesa are pinned by version, not by
  hash.
- The release zip's SHA-256 is computed by the same build job that ran the third-party build
  scripts: it shows the file arrived intact, not that the build was clean.
- The native GPU process checks the channel's messages, but not every guest address in the GPU
  commands it is sent: it isn't a security boundary against a compromised game process (both
  run as the same user, unsandboxed).
- Paths printed by upstream's `run.sh` and the engine can still show the home folder in logs.

## Afterwards

- Fix our findings on every branch they apply to (`macos-0.4`, `macos-0.5`, `macos-native`,
  `macos-native-0.5` and later ones), and run CI (a `ci-*` branch) before pushing.
- Upstream findings: note them in the release notes when they matter to users, and report them
  to deadinside28 privately when they're serious.
- Keep a short summary of both reviews (findings and what was done) with the port's notes for
  that release (`docs/release/<version>.md` or the pull request).
