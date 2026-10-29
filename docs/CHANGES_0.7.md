**English** · [Русский](#русский)

# 0.7

bbport is the counterpart of Wine + DXVK for a single game: Bloodborne (v1.09), now on **Linux
and Windows**. Changes since [0.6](https://github.com/deadinside28/bloodborne_pc/releases/tag/0.6).

Questions about this project: Discord https://discord.gg/KYZRKk9CB — not the shadPS4 server.

### Windows

- **A native Windows build** (Windows 10/11, x86-64, Vulkan 1.3), on the new memory model only:
  `bbport-windows.zip` with `Bloodborne.exe`, a launcher in the Windows 11 style (WPF; Russian,
  English, Brazilian Portuguese, Simplified Chinese, French), `bbport-run.exe` for `--play`, the
  shader recipes and `README-Windows.txt`. Unpack it anywhere, start `Bloodborne.exe`, choose the
  game folder. Saves, settings and the shader cache stay inside the folder.
- **Online play on Windows**: the online module is `online\bbnet20.dll` (shadPS4 0.20 network with
  BBSC, the same code as `libbbnet20.so`), set in the launcher's new *Online* section as on
  Linux: both pools (Seamless Hunt with account and BBSC, The Hunter's Dream with Discord
  sign-in), UPnP, connection check, and BBSC's local game server when the server's own does not
  answer. Mods and third-party XML patches from the launcher too (the *Mods and patches* section;
  without Developer Mode the modded game is put together with junctions and hard links).
- **FSR 4 from AMD's own FSR SDK 2.4 DLLs** (`fsr_amd`, in-game menu): put AMD's Vulkan DLLs
  (`amd_fidelityfx_loader_vk.dll`, `amd_fidelityfx_upscaler_vk.dll`) into `user\fsr` or set
  `BB_FFX_DIR`; the driver's newest model (4.1.2 on Radeon) is used. Windows only.
- Built with MSYS2 CLANG64 (`packaging/windows/`), or cross-built from Linux with llvm-mingw.
- Upscalers on Windows: FSR 3.1, FSR 4, **FSR 4.1.1** (assets built from your own DLL on the same
  PC, no Proton needed), **FSR 4.1.2 driver models** on Radeon (INT8 on RDNA2/3, the *Lite* model;
  taken from the driver's `amdxcffx64.dll`), **DLSS** and **DLSS Frame Generation** through the
  bridge (needs NVIDIA's DLSS SDK at build time).
- Fixed on Windows during testing: stalls in `VirtualQuery` while the game's heaps grew,
  BC6H textures that could not be storage images, the opening movie crash, the spinning character
  in the creator, the HUD atlas decoding, DLSS Frame Generation black frames, a lost shader
  cache after an update.
- NVIDIA on Windows: the port puts a *prefer maximum performance* profile for `bb-probe.exe` into
  the driver (NVAPI; your own Control Panel choice is kept; `BB_NV_POWER=0` off). GPUs stuck at
  low clocks halved the FPS in heavy places. A tester's RTX 5060 now runs as expected.
- Windows + NVIDIA uses FIFO instead of Mailbox (present bursts on Optimus laptops;
  `BB_NVIDIA_MAILBOX=1` keeps Mailbox).
- Tested on: Ryzen 7 5700X3D + RX 7800 XT, i5-7300HQ + GTX 1050 2 GB, i5-8250U + GeForce 930MX.

### One memory model

- **The old memory model of 0.3 is gone.** The new memory and translation model is the only
  one, on every GPU; *Memory model* is no longer in the launcher. A driver that cannot use the
  game's memory in place stops with a `STOP: the GPU driver cannot use the game's memory in place`
  line instead of falling back.
- **The game's graphics heaps grow on demand**: about 1.5 GB of the PS4 heaps were allocated
  but never used; now memory is added as the game takes it and given back when freed (`BB_HEAP_GROW=0`: whole from the start).
- Texture packs with upscaled map textures get larger graphics heaps by themselves (PR #360 by
  wheat32; `BB_LARGE_HEAPS=1/0`).
- *Texture quality* (launcher, game menu): full, half, quarter, or *auto* (half on cards with up to
  3 GB for the game). Big mipmapped textures then skip their top levels.
- Cards with up to 3 GB: smaller stream and constant rings.

### Faster

- **Occlusion queries counted only while the game's queries are open**: before, the results
  were collected ~5 times a frame, each time waiting for the GPU (~0.8 ms of GPU time per frame).
  RX 7800 XT, Old Yharnam, 1080p FSR 4 Native AA: **164 → 189 FPS** in the heaviest view
  (`BB_OCCLUSION_INTERVALS=0`: the old way).
- **PGO was silently off** in the releases since the source folder moved; retrained:
  **+15–20 %** on the PC, Steam Deck FSR 4.1.1 48.8 → 51.3 FPS.
- Steam Deck: draws prepared on the threads that use them on CPUs with fewer than 12 hardware
  threads (−CPU power, +GPU clock); register state reset only where touched; fetch-shader decodes
  cached; texture sets reused without draw preparation. Standing in one spot: 68.6 → 72 FPS.
- Stage A no longer waits for the GPU before reading descriptor tables when the frame's tasks only
  read the CPU's data (Deck 63.5 → 65 FPS, laptop pacing).
- Windows: one system call per draw removed (`std::jthread::joinable` → flag): +3 %.
- Frame generation FSR 3.1: the port's own optical-flow and interpolation shaders (bit-exact SAD,
  hardware bilinear) and an early frame retire: Steam Deck 34.9 → 38.4 FPS with FG.
- **Less traffic over PCIe**: constants already in video memory are not sent again (ring dedup,
  ~300 MB/s less), and of the game's copy shader calls only the copies that change something run
  (checked on a separate thread): +1–4 % in heavy views, more with a busy bus
  (`BB_RING_DEDUP=0`, `BB_COPY_SKIP_SAME=0`: off).
- Live resolution changes no longer cost FPS: the game starts at the chosen preset's size and
  smaller **and larger** presets switch live; the output resolution and aspect ratio switch live
  too. *Auto* output = the monitor's resolution. *Live resolution* is on by default.
- **Shaders before you play:** a shader recipe pack (655 programs, 1389 permutations, 1149
  pipelines from players' caches) is compiled in the background at start; the cache is kept across
  updates (rebuilt from the saved inputs). 0 compiles in a repeat session.

### New

- **Anti-aliasing** menu (launcher, game menu, overlay): *auto* (the game's AA only without an
  upscaler), the game's, **FXAA** (Kyo's *Force FXAA2*), **SMAA 1x** (the port's own, before the
  UI), off. Sharpening (RCAS) works without an upscaler too.
- **FSR 1** (spatial) for weak GPUs.
- **GT-VBAO** in place of the game's SSAO (no dark halos around thin objects, ~7 % FPS; port by
  redditor69).
- **FSR 3.1 frame generation** is shown on every GPU (experimental); DLSS FG only on NVIDIA.
- **AMD Anti-Lag 2** (`VK_AMD_anti_lag`) beside NVIDIA Reflex (Reflex and DLSS FG: PR #260 by
  adds39939).
- **Trophies**: earned trophies go to `user/trophies.jsonl` (port of guutol's work).
- Game menu: *Exit to desktop* in *System* (title screen too). A slow-walk key (Left Alt). Up to
  4 keys per action, desktop shortcut button (launcher). *Invert mouse X*. Skipping videos (Enter,
  Esc, Options).
- The game's mouse auto-rotation is off by default (`BB_MOUSE_AUTO_ROTATE=1`: the game's; from
  ArenaMMORPG's #352).
- Engine fixes from SrShadowy's fork: play time counted correctly above 30 FPS, the hit
  collector's cap raised to 65536, a character-id guard; BBSC 0.0.2 with the relay client and P2P
  relay fallback (Teefpyre's BBSC Linux fixes included).
- BBSC: the hunter's kit and Kin Bell signs and summons in every area (PR #342 by whp199, the
  signs and summons part); vanilla bells back when BBSC is off (PR #346 by DominickCobb-rs).
- Launcher: French (PR #347 by Sarxzer), Simplified Chinese completed (PR #157 by Qiming-Liu);
  settings grouped in sections with descriptions; monitor and controller choice.
- Steam Deck: the Steam overlay is no longer taken out of the game (#217, #77); hold L3+R3 for the port's menu; Steam Input
  only (no doubled keyboard/mouse events).
- Every `BB_*` variable is listed with what it does in [ENVIRONMENT.md](ENVIRONMENT.md); the log's
  `Environment:` line names unknown ones (typos).

### Fixes

- **Cloth fluttering above 30 FPS** (#58): Havok's cloth solver ran fps/30 times a second with
  the stiffness of one 30 FPS step; the stiffness factor now follows the solves per second (the
  old 0.5 clamp was only a range check and did nothing).
- **Wedges in the shadows of lamps and other point lights**: each face of the point-light
  shadow cube is cleared on its own.
- **Grid of item icons over the left half of the screen**: the HUD's stencil masks went into the
  UI depth while the masked icons tested the scene's stencil.
- Vertex explosions (stretched railings in the Hunter's Dream) where a buffer crossed a boundary
  between two pieces of a grown heap.
- FPS drop to 16 after quitting to the title and loading another save (GTX 1060 and other
  NVIDIA): marks from the previous save's loading stayed on memory the game reused for GPU data.
- GTX 10xx: changing the resolution live dropped FPS (no depth proxy without D32S8 blits or
  stencil export): such GPUs now apply it at the next start.
- FG ghosts (bushes where there are none) while moving: instanced foliage got another object's
  previous positions.
- Black effects after a live preset change.
- Crash at `0x2169edc` (GX occlusion query without a handle, #336; PR #343 by whp199).
- Online module: QueryProtection of the game's low memory (#282; "connection appeared" confirmed
  by the user).
- FSR 3.1 never in protected memory (ROG Ally and others, #85; PR #293 by iannitram).
- FIFO Relaxed present mode accepted (PR #225 by AndiMajore).
- Exit 139 with FG on and Mesa's anti-lag layer; Mesa's layer bug with `vkQueueSubmit`
  (the port submits with `vkQueueSubmit2`).
- Frame limiter: no catch-up slots (48/18 ms saw-tooth at 30 FPS).
- Sewers: no AA and an FPS drop (the upscaler was recreated twice a frame by the reactive mask).
- Wayland HiDPI scaling (#328, #378, #234); AppImage `.desktop` and AppStream data (#362, #287).
- Heap shrink race (black blotches on the ground); single crashes at load under raised thread
  priorities (priorities stay normal).
- Steam Deck lazy-memory imports counted twice in GTT (device lost after ~80 min).

### Removed

- The old memory model (see above) and the former online module on shadp2p (`libbbnet.so`):
  `libbbnet20.so` is the only one.
- Texture preload at load time (did not find its images).
- What other games needed and Bloodborne never calls: the LEGO Dimensions toy pad (USB), NGS2
  sound, event flags, time zones, dialogs and PSN services beyond Bloodborne's set (checked
  against the eboot's and libc's 734 imports).

**Known limits:** two-player co-op is tested by players, not by the author; NVIDIA on Windows
(FPS halving in heavy places on RTX 3080/5060) improved, more reports welcome; RDNA4 FP8 runs untested on
hardware; Polaris (RX 400/500) unverified. Bug #32 (rare guest fault at `0x263b8e7`) is still
open.

**Downloads:** `Bloodborne-bbport-0.7-windows-x64.zip` (Windows 10/11: unpack, start
`Bloodborne.exe`), `Bloodborne-bbport-0.7-x86_64.AppImage` (Linux and Steam Deck, one file) and
`Bloodborne-bbport-0.7-linux-x86_64.tar.zst` (Linux as a folder: `Bloodborne.sh` opens the
launcher; the online module `online/libbbnet20.so` lies open, so a BBSC build can replace it).

The packages contain no game files and no AMD FSR 4 model assets. `SHA256SUMS` accompanies the
download.

**Related projects:** [Bloodborne-Android](https://github.com/ducvd89/Bloodborne-Android) (a
fork of bbport for Android) and [bloodborne-mac](https://github.com/bloodborne-native/bloodborne-mac)
(native on Apple Silicon, built on bbport and shadPS4).

**Thanks** to everyone whose work is in this release: wheat32 (#360), Sarxzer (#347), whp199
(#342, #343), DominickCobb-rs (#346), maiconjsv (#339), iannitram (#293), adds39939 (#260),
AndiMajore (#225), Qiming-Liu (#157), ArenaMMORPG (#352/#353), guutol (trophies), redditor69
(GT-VBAO), SrShadowy and Teefpyre (engine and BBSC fixes in SrShadowy's fork), Supermedo and Awaken (the
online module's network stack), IFreemz (DLSS
bridge), the players who sent their shader caches, and everyone who reported bugs with logs.
The authors of the pull requests taken into 0.4–0.6 are credited as co-authors of this release
too ([docs/CONTRIBUTORS_0.7.md](CONTRIBUTORS_0.7.md)).

## Русский

# 0.7

bbport — аналог Wine + DXVK для одной игры: Bloodborne (v1.09), теперь на **Linux и Windows**.
Изменения после [0.6](https://github.com/deadinside28/bloodborne_pc/releases/tag/0.6).

Вопросы по проекту: Discord https://discord.gg/KYZRKk9CB — не сервер shadPS4.

### Windows

- **Нативная сборка для Windows** (Windows 10/11, x86-64, Vulkan 1.3), только на новой модели
  памяти: `bbport-windows.zip` с `Bloodborne.exe`, лаунчером в стиле Windows 11 (WPF; русский,
  английский, бразильский португальский, упрощённый китайский, французский), `bbport-run.exe` для
  `--play`, рецептами шейдеров и `README-Windows.txt`. Распакуйте куда угодно, запустите
  `Bloodborne.exe`, выберите папку игры. Сохранения, настройки и кэш шейдеров — внутри папки.
- **Онлайн на Windows**: модуль онлайна — `online\bbnet20.dll` (сеть shadPS4 0.20 с BBSC, тот же
  код, что `libbbnet20.so`), настраивается в новом разделе лаунчера *Онлайн*, как в Linux: оба
  пула (Seamless Hunt с учётной записью и BBSC, The Hunter's Dream со входом через Discord), UPnP,
  проверка соединения и локальный игровой сервер BBSC, когда сервер не отвечает. Моды и сторонние
  XML-патчи из лаунчера — тоже (раздел *Моды и патчи*; без режима разработчика игра с модами
  собирается через junction и жёсткие ссылки).
- **FSR 4 из DLL самой AMD (FSR SDK 2.4)** (`fsr_amd`, меню в игре): положите Vulkan-DLL AMD
  (`amd_fidelityfx_loader_vk.dll`, `amd_fidelityfx_upscaler_vk.dll`) в `user\fsr` или укажите
  `BB_FFX_DIR`; берётся новейшая модель драйвера (4.1.2 на Radeon). Только Windows.
- Сборка в MSYS2 CLANG64 (`packaging/windows/`) или кросс-сборкой из Linux через llvm-mingw.
- Апскейлеры на Windows: FSR 3.1, FSR 4, **FSR 4.1.1** (ассеты из своей DLL на том же ПК, без
  Proton), **модели FSR 4.1.2 из драйвера** на Radeon (INT8 на RDNA2/3, модель *Lite*; из
  драйверного `amdxcffx64.dll`), **DLSS** и **DLSS Frame Generation** через мост (нужен DLSS SDK
  от NVIDIA при сборке).
- Исправлено на Windows при тестах: остановки в `VirtualQuery` при росте куч игры, текстуры BC6H,
  которые не могли быть storage-изображениями, вылет на вступительном ролике, вращающийся
  персонаж в редакторе, декодирование атласа HUD, чёрные кадры DLSS Frame Generation, потеря кэша
  шейдеров после обновления.
- NVIDIA на Windows: порт ставит в драйвер профиль *максимальная производительность* для
  `bb-probe.exe` (NVAPI; ваш выбор в панели NVIDIA сохраняется; `BB_NV_POWER=0` — выключить).
  Видеокарты на низких частотах вдвое теряли FPS в тяжёлых местах. У тестера на RTX 5060 теперь
  всё как надо.
- Windows + NVIDIA — FIFO вместо Mailbox (пачки кадров на ноутбуках с Optimus;
  `BB_NVIDIA_MAILBOX=1` оставляет Mailbox).
- Проверено на: Ryzen 7 5700X3D + RX 7800 XT, i5-7300HQ + GTX 1050 2 ГБ, i5-8250U + GeForce 930MX.

### Одна модель памяти

- **Старой модели памяти из 0.3 больше нет.** Новая модель памяти и трансляции — единственная, на
  любой видеокарте; пункта *Модель памяти* в лаунчере нет. Драйвер, который не может использовать
  память игры на месте, останавливает игру строкой `STOP: the GPU driver cannot use the game's
  memory in place`, без запасного пути.
- **Графические кучи игры растут по требованию**: около 1,5 ГБ куч PS4 выделялись, но не
  использовались; теперь память добавляется по мере того, как игра её занимает, и отдаётся при
  освобождении (`BB_HEAP_GROW=0` — вся сразу).
- Текстур-паки с увеличенными текстурами карт сами получают большие графические кучи (PR #360 от
  wheat32; `BB_LARGE_HEAPS=1/0`).
- *Качество текстур* (лаунчер, меню игры): полное, половина, четверть или *авто* (половина на
  картах до 3 ГБ для игры). Большие текстуры с мип-уровнями тогда пропускают верхние уровни.
- Карты до 3 ГБ: меньшие кольцевые буферы потока и констант.

### Быстрее

- **Occlusion-запросы считаются только пока открыты запросы игры**: раньше итоги собирались
  ~5 раз за кадр, каждый раз с ожиданием видеочипа (~0,8 мс видеочипа на кадр). RX 7800 XT, Старый
  Ярнам, 1080p FSR 4 Native AA: **164 → 189 FPS** в самом тяжёлом виде
  (`BB_OCCLUSION_INTERVALS=0` — по-старому).
- **PGO молча не работал** в релизах после переноса папки исходников; переобучен: **+15–20 %** на ПК,
  Steam Deck FSR 4.1.1 48,8 → 51,3 FPS.
- Steam Deck: на процессорах меньше чем с 12 аппаратными потоками отрисовки готовятся там же, где
  используются (меньше мощности процессору, выше частота видеочипа); сброс регистров только там,
  где они менялись; кэш разбора fetch-шейдеров; наборы текстур переиспользуются без подготовки
  отрисовок. На одном месте: 68,6 → 72 FPS.
- Стадия A больше не ждёт видеочип перед чтением таблиц дескрипторов, когда задачи кадра читают
  только данные процессора (Deck 63,5 → 65 FPS, плавность на ноутбуке).
- Windows: убран один системный вызов на каждую отрисовку (`std::jthread::joinable` → флаг): +3 %.
- Генерация кадров FSR 3.1: свои шейдеры оптического потока и интерполяции (побитово тот же SAD,
  аппаратная билинейная выборка) и ранний выпуск кадра: Steam Deck 34,9 → 38,4 FPS с FG.
- **Меньше данных по PCIe**: константы, которые уже лежат в видеопамяти, не отправляются заново
  (ring dedup, ~300 МБ/с меньше), а из вызовов копирующего шейдера игры выполняются только копии,
  которые что-то меняют (проверка в отдельном потоке): +1–4 % в тяжёлых местах, больше при
  занятой шине (`BB_RING_DEDUP=0`, `BB_COPY_SKIP_SAME=0` — выключить).
- Смена разрешения на лету больше не стоит FPS: игра стартует в размере выбранного пресета, а
  меньшие **и большие** пресеты переключаются на лету; разрешение вывода и соотношение сторон —
  тоже. *Авто* вывод — разрешение монитора. *Смена разрешения на лету* включена по умолчанию.
- **Шейдеры до игры:** пакет рецептов шейдеров (655 программ, 1389 вариантов, 1149 конвейеров
  из кэшей игроков) компилируется в фоне при запуске; кэш переживает обновления (пересобирается из
  сохранённых входов). Повторная сессия — 0 компиляций.

### Новое

- Меню **сглаживания** (лаунчер, меню игры, оверлей): *авто* (сглаживание игры только без
  апскейлера), игры, **FXAA** (*Force FXAA2* от Kyo), **SMAA 1x** (своё, до UI), выкл. Резкость
  (RCAS) работает и без апскейлера.
- **FSR 1** (пространственный) для слабых видеокарт.
- **GT-VBAO** вместо SSAO игры (без тёмных ореолов вокруг тонких предметов, ~7 % FPS; порт от
  redditor69).
- **Генерация кадров FSR 3.1** показана на любой видеокарте (экспериментально); DLSS FG — только
  на NVIDIA.
- **AMD Anti-Lag 2** (`VK_AMD_anti_lag`) рядом с NVIDIA Reflex (Reflex и DLSS FG — PR #260 от
  adds39939).
- **Трофеи**: полученные трофеи пишутся в `user/trophies.jsonl` (перенос работы guutol).
- Меню игры: *Выход на рабочий стол* в *Системе* (и на титульном экране). Клавиша медленной ходьбы
  (Left Alt). До 4 клавиш на действие, кнопка ярлыка на рабочем столе (лаунчер). *Инверсия мыши
  по X*. Пропуск роликов (Enter, Esc, Options).
- Автоповорот камеры игры от мыши по умолчанию выключен (`BB_MOUSE_AUTO_ROTATE=1` — как в игре; из
  #352 от ArenaMMORPG).
- Исправления движка из форка SrShadowy: время игры считается верно выше 30 FPS, предел сборщика
  попаданий поднят до 65536, защита id персонажа; BBSC 0.0.2 с relay-клиентом и запасным P2P relay
  (с исправлениями BBSC для Linux от Teefpyre).
- BBSC: набор охотника, знаки и призывы Колокола родства в любой зоне (PR #342 от whp199, часть со
  знаками и призывами); ванильные колокола при выключенном BBSC (PR #346 от DominickCobb-rs).
- Лаунчер: французский (PR #347 от Sarxzer), китайский дополнен (PR #157 от Qiming-Liu); настройки
  по разделам с описаниями; выбор монитора и контроллера.
- Steam Deck: оверлей Steam больше не убирается из игры (#217, #77); меню порта — удержание L3+R3; только Steam Input (без
  двойных событий клавиатуры и мыши).
- Все переменные `BB_*` с описанием — в [ENVIRONMENT.md](ENVIRONMENT.md); строка `Environment:` в
  журнале называет неизвестные (опечатки).

### Исправления

- **Трепет ткани выше 30 FPS** (#58): решатель ткани Havok работал fps/30 раз в секунду с
  жёсткостью одного шага 30 FPS; теперь множитель жёсткости следует числу шагов в секунду (старое
  ограничение 0,5 было лишь проверкой диапазона и ничего не делало).
- **Клинья в тенях ламп и других точечных источников**: каждая грань куба теней точечного
  источника очищается отдельно.
- **Сетка из иконок предметов на левой половине экрана**: трафаретные маски HUD уходили в
  глубину UI, а иконки под маской проверяли трафарет сцены.
- «Взрывы» вершин (растянутые перила во Сне охотника), где буфер пересекал границу двух кусков
  выросшей кучи.
- Падение FPS до 16 после выхода в меню и загрузки другого сохранения (GTX 1060 и другие NVIDIA):
  метки от загрузки прошлого сохранения оставались на памяти, которую игра заняла под данные
  видеочипа.
- GTX 10xx: смена разрешения на лету роняла FPS (нет прокси глубины без blit D32S8 или stencil
  export): на таких видеокартах смена теперь со следующего запуска.
- Призраки FG (кусты там, где их нет) при движении: инстансная растительность получала прошлые
  позиции другого объекта.
- Чёрные эффекты после смены пресета на лету.
- Вылет на `0x2169edc` (GX occlusion query без дескриптора, #336; PR #343 от whp199).
- Онлайн-модуль: QueryProtection нижней памяти игры (#282; «соединение появилось» —
  подтверждено пользователем).
- FSR 3.1 никогда не в protected memory (ROG Ally и другие, #85; PR #293 от iannitram).
- Режим вывода FIFO Relaxed принимается (PR #225 от AndiMajore).
- Выход 139 с FG и слоем anti-lag из Mesa; ошибка этого слоя в `vkQueueSubmit` (порт отправляет
  через `vkQueueSubmit2`).
- Ограничитель кадров: без догоняющих слотов («пила» 48/18 мс на 30 FPS).
- Канализация: без сглаживания и с падением FPS (реактивная маска пересоздавала апскейлер дважды
  за кадр).
- Масштабирование HiDPI под Wayland (#328, #378, #234); `.desktop` и данные AppStream в AppImage
  (#362, #287).
- Гонка при сжатии куч (чёрные пятна на земле); редкие вылеты при загрузке с повышенными
  приоритетами потоков (приоритеты остаются обычными).
- Steam Deck: импорты ленивой памяти считались в GTT дважды (device lost примерно через 80 минут).

### Убрано

- Старая модель памяти (см. выше) и прежний онлайн-модуль на shadp2p (`libbbnet.so`): остался
  только `libbbnet20.so`.
- Предзагрузка текстур при загрузке (не находила свои изображения).
- То, что было нужно другим играм и что Bloodborne не вызывает: USB-портал LEGO Dimensions, звук
  NGS2, флаги событий, часовые пояса, диалоги и сервисы PSN сверх набора Bloodborne (сверено с
  734 импортами eboot и libc).

**Известные ограничения:** кооператив вдвоём проверяют игроки, а не автор; NVIDIA на Windows
(FPS вдвое ниже в тяжёлых местах на RTX 3080/5060) улучшено, ждём ещё отчётов; FP8 на RDNA4 не проверен на
железе; Polaris (RX 400/500) не проверены. Ошибка #32 (редкий guest fault на `0x263b8e7`) ещё
открыта.

**Загрузки:** `Bloodborne-bbport-0.7-windows-x64.zip` (Windows 10/11: распакуйте, запустите
`Bloodborne.exe`), `Bloodborne-bbport-0.7-x86_64.AppImage` (Linux и Steam Deck, один файл) и
`Bloodborne-bbport-0.7-linux-x86_64.tar.zst` (Linux папкой: `Bloodborne.sh` открывает лаунчер;
модуль онлайна `online/libbbnet20.so` лежит открыто, сборку BBSC можно подложить вместо него).

Пакеты не содержат файлов игры и ассетов моделей FSR 4 от AMD. К загрузке приложен `SHA256SUMS`.

**Связанные проекты:** [Bloodborne-Android](https://github.com/ducvd89/Bloodborne-Android) (форк
bbport для Android) и [bloodborne-mac](https://github.com/bloodborne-native/bloodborne-mac)
(нативно на Apple Silicon, на основе bbport и shadPS4).

**Спасибо** всем, чья работа вошла в этот выпуск: wheat32 (#360), Sarxzer (#347), whp199 (#342,
#343), DominickCobb-rs (#346), maiconjsv (#339), iannitram (#293), adds39939 (#260), AndiMajore
(#225), Qiming-Liu (#157), ArenaMMORPG (#352/#353), guutol (трофеи), redditor69 (GT-VBAO),
SrShadowy и Teefpyre (исправления движка и BBSC в форке SrShadowy), Supermedo и Awaken (сетевой
стек модуля онлайна), IFreemz (мост DLSS), игрокам,
приславшим свои кэши шейдеров, и всем, кто сообщал об ошибках с журналами.
Авторы pull request'ов, вошедших в 0.4–0.6, тоже указаны соавторами этого выпуска
([docs/CONTRIBUTORS_0.7.md](CONTRIBUTORS_0.7.md)).
