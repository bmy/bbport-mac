# Trophies

The game's trophy calls (`sceNpTrophy*`, `src/runtime_services.c`) are kept and announced by
`gpu/shim/bbport_trophies.cpp`, after shadPS4's `np_trophy.cpp` and `trophy_ui.cpp`:

- Every unlock is kept in `user/trophy/<NP communication ID>/unlocked.txt` (Bloodborne:
  `NPWR05818_00`). Only the first unlock of a trophy is announced; the platinum follows once
  every trophy counting towards it is in.
- The announcement: a panel at the top centre for 6 s (the game's HUD keeps the corners) with the
  trophy's icon and a dot in the grade's colour (platinum, gold, silver, bronze), "Trophy earned"
  and the name. The icons are decoded with SDL 3.4's PNG loader and handed to ImGui as user
  textures; with an older SDL, or no icon, a badge in the grade's colour stands in.
- Names and grades come from the game's `sce_sys/trophy/trophy00.trp`. Its XML is encrypted;
  the port decrypts it once into the same folder (with the trophy icons) when a trophy key is
  given: `BB_TROPHY_KEY=<32 hex digits>` in the environment or in a `.env` file in the port's
  folder. The key is the user's own (shadPS4 asks for the same one); it is never part of the
  repository. Without a key, or a build without OpenSSL's libcrypto, trophies are still kept and
  announced by number.
- The language is the game's (`BB_LANGUAGE`, the PS4 numbers: 1 English, 8 Russian).
- `BB_TROPHY_TEST=<id>`: shows that trophy once, about 40 s after the start, nothing kept (to
  see the look).

Not yet: a sound.
