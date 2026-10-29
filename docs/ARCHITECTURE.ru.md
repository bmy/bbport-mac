# Карта кода bbport (2026-10-26)

Короткий путеводитель: что где лежит и куда смотреть, когда что-то не так. Подробности по
отдельным темам — в доках, на которые ссылаются заголовки файлов (`docs/parallel_gpu.md`,
`docs/MEMORY_MODULE_PLAN.ru.md`, `docs/upscaler.md`, ...). Все настройки `BB_*` с описаниями —
`docs/ENVIRONMENT.md` (из `gpu/shim/bb_env_vars.def`).

## Как устроен запуск

1. `scripts/` (Python, офлайн): eboot игры проверяется (`game_check.py`), превращается в плоский
   образ, в него вкомпоновываются PS4 libc и libSceFios2 (`prepare.py`, `link_libc.py`,
   `link_modules.py`), применяются патчи (`patches.py`, `patches/Bloodborne.xml`).
2. `run.sh` (Linux) / `run.py` (Windows) собирают окружение из `bbport.ini` и лаунчера и
   запускают `bb-probe`.
3. `src/probe.c` — загрузчик: отображает образ и прыгает в код игры. Код игры исполняется на CPU
   как есть.
4. `src/runtime_*.c` — рантайм на C: то, что игра вызывает из системных библиотек PS4 (память,
   потоки, синхронизация, файлы, звук с ATRAC9, геймпад, сохранения, AppContent, сеть-заглушки).
   Два самых больших файла разложены по темам в подкаталоги: `src/runtime_memory/` (пул memfd и
   его второй вид, таблица VMA, функции `sceKernel*` памяти, нижняя область, рост кучи, ловушки
   записи) и `src/runtime_pad/` (калибровка стиков, тачпад, выбор геймпада, раскладка клавиатуры и
   мыши, подсказки кнопок, движение без датчиков, выборка). Части подключаются через `#include` в
   `runtime_memory.c` / `runtime_pad.c` по порядку: это одна единица трансляции, static-переменные
   общие, отдельно части не компилируются.
5. Графические вызовы и команды GPU идут в библиотеку `gpu/` (`libbbgpu`).

## Путь кадра на GPU

```
игра пишет PM4 (через статически вкомпонованный libGnm)
  -> gpu/cp/              командный процессор: разбор PM4, регистры, DMA, метки, fence
  -> стадия A (поток GpuComm, gpu/render/raster_pipe.cpp):
       выбор программ и пайплайна (render/pipeline_*.cpp),
       константы в кольцо (CollectRingBindings), упаковка draw
  -> стадия B (поток DrawRec): привязка буферов/текстур (render/raster_bind.cpp,
       raster_textures.cpp), цели рендеринга (raster_targets.cpp), состояние (raster_state.cpp)
  -> поток записи Vulkan (VkRecorder, render/scheduler.cpp) -> vkQueueSubmit
  -> презентер (render/presenter.cpp): апскейлер, frame generation, UI, swapchain
```

Шейдеры GCN транслируются в SPIR-V в `gpu/shader/` (frontend → IR → проходы → backend). Готовые
пайплайны кэшируются на диске (`render/pipeline_store.cpp`, `render/blob_store.cpp`); рецепты
шейдеров (`shader_recipes.bin`) позволяют компилировать почти всё заранее в фоне
(`render/pipeline_import.cpp`, `render/shader_rebuild.cpp`).

## Каталоги `gpu/`

| Каталог | Что внутри | С чего начать |
|---|---|---|
| `cp/` | командный процессор PM4: графика, compute-очереди, задачи «в порядке потока» | `command_processor.h`, `cp_graphics.cpp` |
| `render/` | Rasterizer (draw/dispatch → Vulkan), кэш пайплайнов, планировщик, презентер | `rasterizer.h`, `raster_pipe.cpp`, `pipeline_cache.h` |
| `shader/` | транслятор GCN → SPIR-V (родом из shadPS4) | `recompiler.h`, `specialization.h`, `resource.h` |
| `gcn/` | описание железа GCN: ISA, регистры, sharps (V#/T#/S#) | `isa.h`, `regs_targets.h`, `sharps.h` |
| `buffers/` | кэш буферов: память игры для GPU, резидентность в VRAM, readback | `buffer_cache.h` (+ `buffer_*.cpp`) |
| `layer/` | модуль памяти прослойки: память игры на месте + VRAM-зеркала | `bblayer_gpu_memory.h`, `buffer_cache_layer.cpp` |
| `memory/` | отслеживание грязных страниц, защиты, источники фолтов | `dirty_tracker.h`, `page_watch.h` |
| `textures/` | кэш текстур: поиск, загрузка, детайлинг, сборщик, запасные изображения | `texture_cache.h`, `image.h`, `image_spares.h` |
| `upscale/` | FSR 3.1/4/4.1.1, DLSS, frame generation, motion vectors, TAA/SMAA, VBAO | `vk_temporal_upscaler.h` (+ `upscaler_*.cpp`) |
| `vulkan/` | инстанс, устройство, swapchain, пулы, timeline | `instance.h` |
| `platform/` | системные библиотеки на стороне GPU-библиотеки (videoout, gnm, kernel) | `videoout/`, `gnm/` |
| `shim/` | связка с рантаймом, оверлей-меню, настройки, окно, диагностика, хуки «любой игры»; память игры для Vulkan (`bbport_guest_memory*.cpp`: выбор модели и поиск кусков, куски целиком, ленивый импорт memfd, проверки драйвера при старте) | `bbgpu.cpp`, `bbport_settings.h`, `bbport_overlay.h`, `bbport_guest_memory_internal.h` |
| `games/` | всё, что знает про конкретную игру | `game_profile.h`, `bloodborne/` |
| `bbnet20/` | онлайн-модуль (`libbbnet20.so`, только Linux): сеть/NP shadPS4 0.20, BBSC, THD | `bbnet.h`, `SHADNET-PORT.md` |
| `third_party/` | сторонние библиотеки (sirit, imgui, fsr-vulkan, ...) | — |

## Граница «общий слой / игра»

- Ядро (cp, render, shader, buffers, layer, textures) не знает про Bloodborne: игра без профиля
  идёт по общим путям (`BB_GAME_PROFILE=none`).
- Всё про Bloodborne — `gpu/games/`:
  - `bloodborne.cpp` — профиль: сериалы, хуки, известные писатели покадровых данных;
  - `bloodborne/hooks*.cpp` — хуки в коде игры: память (`hooks.cpp`), эффекты, job-фиксы,
    рост кучи, камера/HUD под любое соотношение сторон, защиты GX, RE-пробы;
  - `bloodborne_menu_pointer.cpp`, `bloodborne_kit_restore.cpp`.
- Ещё специфичны для игры (пока не в `games/`): `shim/bbport_frame_rate.cpp` (разблокировка FPS),
  `shim/bbport_game_menu.cpp` (страницы в меню игры), `shim/bbport_engine_fixes.cpp`,
  разметка кадра для апскейлера (`upscale/vk_camera_motion.cpp`: камера из констант G-buffer
  прохода).

## Соглашения

- **Один класс — несколько файлов.** Большие классы разбиты по смыслу: `Rasterizer` →
  `rasterizer.cpp`, `raster_*.cpp`; `BufferCache` → `buffer_*.cpp` и `layer/layer_*.cpp`;
  `PipelineCache` → `pipeline_*.cpp`; `TemporalUpscaler` → `upscaler_*.cpp`. Шапка каждого файла
  говорит, что в нём.
- **`*_internal.h`** — то, что делят части одного модуля (только они их подключают). Общее
  изменяемое состояние там помечено `inline`: один экземпляр на всю программу. Локальное
  состояние файла — в его анонимном namespace.
- **Настройки** — переменные `BB_*` (все с описаниями: `docs/ENVIRONMENT.md`); для игрока — `bbport.ini`,
  меню в игре (`shim/bbport_settings.h`) и лаунчер. Новое поведение обычно вводится с
  переключателем, чтобы его можно было выключить без пересборки.
- **Диагностика** — переменные `BB_*` с пометкой «diagnostics»; в горячих циклах — за одной
  проверкой флага, сама работа в отдельной холодной функции.

## Сборка

- Linux: `build.sh` (nix-shell, `shell.nix`), пакеты — `packaging/` (AppImage, nix).
- Windows: кросс-сборка `packaging/windows/package.sh` (llvm-mingw + MSYS2 sysroot) или нативно
  в MSYS2 CLANG64: `cmake -S gpu -B out/win-native -G Ninja -DBB_NET=OFF` (xbyak — заголовки
  отдельно, см. `CMAKE_CXX_FLAGS`).
- Тесты: `build.sh --test` (C-рантайм), цели `*-test` в `gpu/CMakeLists.txt` (EXCLUDE_FROM_ALL).
