# Shader recipes

The first session of a new install, a new GPU or a cleared cache used to translate and compile
every shader while the game ran: about 450 compilations when entering the first level, 0.7 s of
hitches on the Steam Deck. The port now ships `shader_recipes.bin`: the inputs of every shader
compilation of a recorded playthrough, without the game's code.

## What the pack holds

For each recorded program (by the hash in its "OrbShdr" block) and each of its permutations:

- the input record (`.in`, render/shader_capture.h) with the program code removed: user data,
  runtime info, bindings, flattened user data and the descriptors the compiler read;
- the metadata (`.meta`: specialization and info);
- references to programs the compiler read through memory (fetch or copy shaders), by hash.

And the pipeline keys (`.key`) whose stages all come from those programs. No game code: the
import reads it from the player's own dump (`dvdroot_ps4/shader/*.dcx` and `eboot.bin`).

## At start

1. The pipeline cache preloads as before and writes this GPU's profile.
2. `bb-probe --import-shaders` runs as a separate, niced process: it translates the pack's
   programs the cache does not have yet with that profile (into `<cache>/import`). In its own
   process the recorded guest memory can be mapped at its addresses. A compiler crash ends only a
   child; that program is remembered in `import.failed` and skipped.
3. The game merges the folder a few programs a frame (2 ms budget). Graphics pipelines are created
   on the compiler's threads; when the game needs one still queued, it builds that one at once.
   Programs the game translated meanwhile keep the game's own permutation numbers; the pack's
   permutations of them join after those (renumbered, their pipeline keys rewritten to match;
   before 2026-10-27 they were dropped and compiled again in the level).
4. A cache of another build is not lost (Linux): `BlobStore::Clear` keeps its `*.meta` and `*.key`
   in `previous/`, and the next import first merges them into the pack (MergeMetas, as for another
   player's metadata) and imports the result once.

The SPIR-V is byte-identical to what the game compiles. The next start finds everything in the
cache, as with a cache of one's own.

## Measured (fresh cache)

| | without recipes | with recipes |
|---|---|---|
| PC, entering the level | 435 compiles, 306 ms | 77 compiles, 44 ms |
| Steam Deck, entering the level | 453 compiles, 664 ms | 55 compiles, 25 ms |
| Import time (PC / Deck) | | 1.5 s / 2.9 s |

## Making the pack

From a cache that has played through the game (the more areas, the more it covers):

    out/bb-probe --export-shader-recipes user/cache/CUSA03173 ../game_files/CUSA03173 shader_recipes.bin

The import runs on Linux and Windows (a child process; on Windows a program the translator
crashed on is remembered in `import.failed`). `BB_SHADER_RECIPES=` (empty) turns it off.

## Adding other players' caches

The tools below run on Linux (they fork a process per program: a translator failure ends only it).
Each takes the device profile of a cache made by this build (`CACHE_DIR/profile.bin`), the
game's files and a work folder, and checks every permutation it adds by translating it.

**A whole cache folder** (`*.in`, `*.meta`, `*.key`: what a player's `user/cache/CUSA03173`
holds). Export it, then merge the two packs:

    out/bb-probe --export-shader-recipes THEIR_CACHE ../game_files/CUSA03173 their.bin
    out/bb-probe --merge-shader-packs CACHE_DIR shader_recipes.bin their.bin ../game_files/CUSA03173 merged.bin WORK 12

A permutation of theirs with the specialization of one in the pack is the same one; the others
are appended to their programs, and their pipelines are renamed to the merged permutations
(a pipeline with a stage that did not make it is dropped). Pipelines are what saves the most:
the driver's compile (about 70 ms each on a GTX 1050 laptop) costs far more than translating a
shader (about 1 ms).

**Metadata only** (`*.meta`, e.g. an older build's cache, versions 7 and 8): each permutation's input is
rebuilt from the metadata itself (BuildMetaInput: its flat user data, runtime info and bindings,
made-up vertex V# tables and a fetch shader generated from the specialization: the game makes
its fetch shaders at run time, none is in its files) and kept when it translates to exactly that
specialization (96.9 % of the pack's own permutations do); else the input of one in the pack
that gives it (the program's own permutations first, then programs near it), or it is left out.
Their `*.key` files (in META_DIR or beside it) are added with their stages renamed:

    out/bb-probe --merge-shader-metas CACHE_DIR shader_recipes.bin ../game_files/CUSA03173 META_DIR merged.bin WORK 12

What to ask players for: the folder `user/cache/CUSA03173` (Linux AppImage:
`~/.local/share/bbport/user/cache/CUSA03173`) as an archive, sent privately: its `*.in` records
hold the game's shader code, which the pack leaves out and must not be posted publicly.

2026-10-24: one player's metadata (1183 permutations) added 22; this PC's test cache added 74
permutations and 49 pipelines: 490 programs, 944 permutations, 660 pipelines. A fresh cache now
compiles nothing at the title once the import is merged (37 compiles, 5 ms, during the logos
before it).

## The game's own programs (`tools/shader_catalog.py`)

Idea from the SrShadowy fork's static catalog. `build GAME_DIR catalog.json` lists every program
in `dvdroot_ps4/shader/*.shaderbnd.dcx` and in `eboot.bin` by its hash (the port's `pgm_hash`), with
its stage and BND4 entry names (`GXFlver_<material>_<pass>[_Skin][_Ins]`). `coverage catalog.json
shader_recipes.bin [cache folders...]` shows how many of them a pack or a cache holds, by bundle,
stage and pass.

Measured 2026-10-25 (CUSA03173 1.09): 8851 programs (3382 in the bundles, 5469 small ones in the
executable); the recipe pack holds 626, all of them from the game's files: gxflvershader 327 of
2813, eboot 141 of 5469, gxrenderershader 53 of 111. The flver passes it has least of: FwdSSdw
0 of 221, DptVACube 0 of 64. The names are what a prediction from materials and passes would start
from.

2026-10-27: the player's own caches (Linux, Windows, and an older build's 561 metadata files) added
6 programs, 10 permutations and **219 pipelines** (930 → 1149): the programs were nearly all there,
their pipelines were not. Fresh cache, standing in the level: 68 → 48 compiles (6 after the import
instead of 25); the next start 0.
