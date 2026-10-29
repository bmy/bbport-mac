# Сессия 27 октября 2026: Windows на основном ПК

Основной ПК (Ryzen 7 5700X3D, RX 7800 XT, 32 ГБ) теперь на Windows 11 LTSC. Всё лежит в
`E:\bloodborne`. Сборка — нативная, MSYS2 CLANG64 (`C:\msys64`), скрипт
`out/win-native/build.sh`. Ничего не опубликовано: всё локально, в git не отправлялось.

## Исправлено (проверено в игре)

| Что | Причина | Где |
|---|---|---|
| Белый мир (только HUD) на AMD под Windows | текстуры BC6H создавались с флагом Storage, который драйвер AMD не поддерживает; теперь без него, если драйвер отказывает | `gpu/textures/image.cpp` |
| Настройки меню не сохранялись (`Settings: cannot write`) | `std::rename` в Windows не заменяет существующий файл → `std::filesystem::rename` | `gpu/shim/bbport_settings.cpp` |
| **Статтеры 50–850 мс** при подгрузке областей (и у друга на RTX 5060) | обёртка `mprotect` для Windows звала `VirtualQuery`, а тот проходит все страницы до конца однородного региона: 5–16 мс на ГиБ, куча игры — один вид на 2,3 ГиБ. Теперь регионы берутся из таблицы `pieces` арены | `gpu/platform/win32/posix_memory.cpp` |
| Статтеры ~200 мс при росте кучи игры | импорт куска 320 МиБ в видеокарту (120–170 мс) шёл в потоке игры под её замком. Теперь куски импортируются заранее в фоне (`runtime_memory_commit_ahead`, `HeapGrowAhead`, `BB_HEAP_GROW_AHEAD=N`, по умолчанию 2) | `src/runtime_memory.c`, `gpu/games/bloodborne/hooks_heap.cpp` |
| HUD без значков пузырьков/пуль, рамок | копия атласа HUD для иконок кнопок делалась blit-ом из BC7 — на AMD/Windows пустая. Теперь compute-проход `atlas_decode.comp` (заодно ушла двойная гамма sRGB) | `gpu/render/vk_button_prompts.*`, `gpu/render/shaders/atlas_decode.comp`, `gpu/CMakeLists.txt` |
| Модель в редакторе персонажа крутится в 7 раз быстрее при 200 FPS | `0x1db2780` прибавляет к углу `стик × [0x4928dfc]` (1,5) за кадр; добавлено в перенаправляемые значения (49 из 49) | `gpu/shim/bbport_frame_rate.cpp` |
| Вылет на вступительном ролике «Новой игры» (DL_PANIC «Mutex is not initialized», и у друга с RTX 4060) | игра обрабатывает одно событие плеера за кадр и при 200+ FPS успевала попытаться запустить ролик раньше нашего асинхронного StateReady (~10 мс) → разбирала ролик, а поток звука стартовал на мёртвом объекте. Теперь дорожки ищутся и StateReady отправляется синхронно в `sceAvPlayerAddSource` (`BB_AVPLAYER_ASYNC_READY=1` — по-старому) | `gpu/platform/avplayer/player.cpp` |

Замер стоя (одна точка, 100 с): потеряно в кадрах > 50 мс **9,6 с → 1,2 с**, медианный кадр
**21,2 → 10,8 мс**.

## Сделано, но не проверено пользователем

- Мышь в меню порта: координаты из окна пересчитываются в размер кадра (`ToDisplay`), в
  оконном режиме промах был 25 % по ширине. Логика меню проверена сценарным щелчком.
- Метки апскейлера в профайлере GPU по выбранному апскейлеру (FSR 4, FSR 3.1, …).

## Осталось

- Портрет слота «bes» в списке сохранений — цветной шум. Версия: портрет снимается при
  сохранении с GPU-рендера, и процессор прочитал старую память вместо VRAM-копии. Не проверено.
- Плеер шлёт игре StateStop сразу после Start, хотя ролик идёт (на показ не влияет).
- Статтеры входа в мир (~470 мс, сотни текстур) и кадры 100–145 мс при загрузке текстур.
- Тяжёлый проход игры 1920x800 (ps `6c2323c2`) — 5,2 мс GPU; возможно SSR из патча порта.
- Вызовы `VirtualQuery` ещё в `Carve`/`SaveProtections` (munmap части вида).

## Диагностика (по умолчанию выключено)

- `BB_OVERLAY_TRACE=1` — мышь в меню порта и переключение галочек; `BB_MENU_KEYS_FILE` понимает
  `click X Y`.
- `BB_AVPLAYER_TRACE=1` — вызовы видеоплеера и события, которые он шлёт игре.
- Строки статтеров: `write traps (all threads)` (ожидание замков / защита / МиБ), `Windows: N
  VirtualProtect calls`; `heap grown … (import / lock wait / locked / GPU side told)`.
- Точки наблюдения на Windows (отладочные регистры): `BB_RE_WATCH_ADDR`,
  `BB_RE_WATCH_VIRTUAL`+`BB_RE_WATCH_INDEX`, `BB_RE_WATCH_FOLLOW`, `BB_RE_WATCH_FILE` (адрес из
  файла, на лету). `BB_RE_ANALOG` считает оба стика.
- `scePthreadMutexInit` пишет причину отказа в лог.
- `out/win-native/bench/`: `standrun.ps1` (прогон стоя на копии сохранений), `stalls.ps1`,
  `gpuprof.ps1`, `watch.ps1` (загрузка CPU/GPU/памяти со стороны), `memscan.exe` (поиск значения
  в памяти игры: changing/still/find/peek), микротесты `vp_bench*`, `vq_bench`, `import_*`.

## Изменённые файлы

`gpu/CMakeLists.txt`, `gpu/games/bloodborne/hooks_heap.cpp`, `gpu/games/bloodborne/hooks_probes.cpp`,
`gpu/platform/avplayer/avplayer.cpp`, `gpu/platform/avplayer/player.cpp`,
`gpu/platform/videoout/frame_stats.cpp`, `gpu/platform/win32/posix_memory.cpp`,
`gpu/render/shaders/atlas_decode.comp`, `gpu/render/vk_button_prompts.cpp`,
`gpu/render/vk_button_prompts.h`, `gpu/shim/bbport_frame_rate.cpp`,
`gpu/shim/bbport_guest_memory.cpp`, `gpu/shim/bbport_overlay.cpp`, `gpu/shim/bbport_settings.cpp`,
`gpu/textures/image.cpp`, `gpu/upscale/vk_temporal_upscaler.cpp`, `src/runtime_memory.c`,
`src/runtime_mutex.c`, `src/runtime_rewatch.c`. 2026-10-27 перенесены в Linux-дерево: сборка под Linux и `build.sh --test` проходят, в игре под Linux не проверялось.
