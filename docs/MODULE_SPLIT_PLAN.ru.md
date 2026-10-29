# План: разбиение кода на модули (2026-10-26)

Цель: код должно быть удобно читать человеку. Файлы на 2–4 тысячи строк, функции на 200–700
строк и настройки через `getenv`, разбросанные по всему дереву, — основное, на что жалуются в
отзывах. Поведение порта не меняется: каждый шаг — перенос и переименование, а не новая логика.

## Статус на вечер 2026-10-26

Сделано (сборка под Windows без ошибок; построчная проверка переносов; `llvm-nm`: ни одна
локальная переменная не попала в два объектника). Оригиналы и список новых файлов:
`Desktop\dev\bbport-archive\before-2026-10-26\MANIFEST.md`.

- Этап 1: `buffer_cache.cpp`, `buffer_cache_layer.cpp`, хуки Bloodborne (→ `games/bloodborne/`),
  `vk_temporal_upscaler.cpp`, `shader_rebuild.cpp` (Linux-часть → `shader_predict.cpp`, здесь не
  компилируется), `pipeline_cache.cpp`, `rasterizer.cpp`.
- Этап 2: `Menu` оверлея по разделам; диагностика из горячего цикла `CollectRingBindings`.
- Этап 3: опись `docs/KNOBS.ru.md` с кандидатами на удаление (удаление — по твоему списку).
- Этап 4: удалён старый онлайн-модуль `gpu/bbnet/`.
- Этап 5: корень разобран, карта `docs/ARCHITECTURE.ru.md`.

Доделано 2026-10-27 на Linux (сборка Linux и кросс-сборка Windows без ошибок, `build.sh --test`
проходит; в игре не проверялось). Оригиналы: `bbport-archive/before-2026-10-27-linux-split/`.
- `shim/bbport_guest_memory.cpp` → `bbport_guest_memory.cpp` (выбор модели, `Install`, поиск
  кусков), `_chunks.cpp` (куски целиком: экспорт dma-buf, импорт host, private-регионы),
  `_lazy.cpp` (ленивый импорт memfd: udmabuf / host import), `_probes.cpp` (проверки драйвера при
  старте) + `bbport_guest_memory_internal.h` (общее состояние — `inline`-переменные в
  `BbGuestMemory::detail`). Функции перенесены без правок.
- `src/runtime_memory.c` → 7 частей в `src/runtime_memory/`, `src/runtime_pad.c` → 7 частей в
  `src/runtime_pad/`. Код C с десятками общих static: части подключаются `#include` по порядку
  (одна единица трансляции, раскрытие `#include` даёт исходный файл строка в строку), так что
  ни переименований, ни новых глобальных символов (`bb-probe` собирается с `-rdynamic`).
- `launcher/bbport_launcher.py` (GTK, Linux) → окно и запуск игры остались в нём;
  `bbport_launcher_common.py` (настройки, списки выбора, окружение игры, функции без окна),
  страницы окна — mixin-классы `LauncherWindow`: `_content.py` (моды, патчи), `_online.py`,
  `_trophies.py`, `_log.py`, `_upscalers.py` (сборка FSR 4.1.1, модели 4.1.2, библиотеки DLSS).
  Сверено по AST: все функции и методы те же; одна правка — `create_shortcuts` указывает на
  `bbport_launcher.py`, а не на модуль, где теперь лежит. Окно строится с настоящим GTK; в тесте
  `test_launcher_gpu.py` базы `Adw` — пустые классы (mock-база рядом с mixin — конфликт метаклассов).
- Длинные горячие функции (`LayerBind`, `BindTextures`, `PostDraw`, `CommandProcessor::Graphics`,
  `RunDrawPacket`) — только с замером стоимости draw в игре.
- Остальные холодные длинные функции (`CreateDevice`, `OnFlipStats`, `RunUiOnly`, `probe.c main`).

## Что было (замер до разбиения) (замер по `gpu/` и `src/`, без third_party и вендорного кода)

**Самые большие файлы**

| Строк | Файл | Что внутри |
|---:|---|---|
| 4002 | `gpu/buffers/buffer_cache.cpp` | привязка буферов, readback, резидентность VRAM, write ticks, twins, VRAM-ловушки (отладка), синхронизация, конфиг из env |
| 3129 | `gpu/layer/buffer_cache_layer.cpp` | тот же класс `BufferCache`: привязка слоя, promote/merge/demote, пул mirror'ов, maintain, paged-записи, ловушки и проверки |
| 3039 | `gpu/shim/bbport_guest_hooks.cpp` | все хуки в коде игры: детуры, рост кучи, эффекты, камера/аспект/HUD, job-фиксы, Gx-guards, пробы ввода |
| 2828 | `gpu/upscale/vk_temporal_upscaler.cpp` | дампы кадров, ресурсы и пайплайны, TAA/EASU/резкость/reactive/SMAA, редиректы целей, UI-проход, FSR4/DLSS |
| 2294 | `gpu/render/shader_rebuild.cpp` | пакеты рецептов, экспорт/импорт, пересборка кэшей, предсказание, слияние метаданных |
| 2013 | `gpu/render/pipeline_cache.cpp` | ключи пайплайнов, программы, фоновая компиляция, импорт шейдеров |
| 1849 | `src/runtime_memory.c` | |
| 1714 | `gpu/shim/bbport_overlay.cpp` | одна функция `Menu` — 721 строка |
| 1665 | `gpu/shim/bbport_guest_memory.cpp` | |
| 1598 | `gpu/render/rasterizer.cpp` | |
| 1546 | `src/runtime_pad.c` | |
| 2465 | `launcher/bbport_launcher.py` | |

**Функции длиннее 150 строк — 39 штук.** Самые длинные:

| Строк | Функция | Горячий путь (per-draw)? |
|---:|---|:---:|
| 721 | `shim/bbport_overlay.cpp` `Menu` | нет |
| 546 | `shader/frontend/translate/vector_alu.cpp` `Translator::EmitVectorAlu` | нет (switch по опкодам, стиль shadPS4) |
| 419 | `vulkan/instance.cpp` `Instance::CreateDevice` | нет |
| 410 | `platform/videoout/frame_stats.cpp` `OnFlipStats` | раз в кадр |
| 408 | `render/raster_textures.cpp` `Rasterizer::BindTextures` | **да** |
| 381 | `layer/buffer_cache_layer.cpp` `BufferCache::LayerBind` | **да** |
| 362 | `upscale/vk_temporal_upscaler.cpp` `RunUiOnly` | раз в кадр |
| 359 | `cp/cp_graphics.cpp` `CommandProcessor::Graphics` | **да** (switch по пакетам PM4) |
| 316 | `src/probe.c` `main` | нет |
| 283 | `render/shader_rebuild.cpp` `MergeMetas` | нет |
| 252 | `render/raster_pipe.cpp` `Rasterizer::PostDraw` | **да** |
| 249 | `layer/buffer_cache_layer.cpp` `LayerMaintain` | раз в кадр |
| 239 | `render/raster_pipe.cpp` `CollectRingBindings` | **да** |

**Настройки:** 568 разных переменных `BB_*` и 634 вызова `getenv`, разбросанных по файлам. Многие
из них — A/B-ручки экспериментов, которые уже закончились (например, `BB_MIRROR_VRAM_ONLY`,
`BB_MIRROR_BURST`, `BB_SPEC_FAST_SIG`, `BB_IDLE_FLUSH_DRAWS`). Читателю непонятно, какие из них
настоящие настройки, а какие отладка.

**Классы-«боги»:** `BufferCache` (`buffer_cache.h`, 802 строки, методы в двух `.cpp` на 7 тыс.
строк) и `Rasterizer` (`rasterizer.h`, 924 строки).

## Этапы

### Этап 1. Разрезать большие файлы по смыслу (механический перенос)

Функции переезжают целиком, без правок кода. Классы пока остаются теми же: несколько `.cpp` на
один класс, как уже сделано у `Rasterizer` (`raster_pipe.cpp`, `raster_bind.cpp`,
`raster_textures.cpp`).

1. **`buffers/buffer_cache.cpp`** →
   - `buffer_cache.cpp` — ядро: `ObtainBufferImpl`, `ObtainBufferForImage`, `BindInPlace`, `CommandWrite*`
   - `buffer_readback.cpp` — `ReadMemory`, `DownloadMemory`, `Readbacks`, `SharedReadback`, `LayerCopyBackForCpu`
   - `buffer_residency.cpp` — `EnsureResident`, `AllocateResidency`, `Evict*`, `ProcessIdleBlocks`, `ProcessDemotions`, `QueuePromotions`
   - `buffer_write_ticks.cpp` — `WriteTicks`, `PreciseTicks`, `Note*Write*`, `ProcessLateWrites`, shadows
   - `buffer_twins.cpp` — twins
   - `buffer_sync.cpp` — `SynchronizeMemory*`, `UploadCopies`, `SubmitPendingArenaBinds`, мёртвые образы и unmap
   - `buffer_debug.cpp` — VRAM-ловушки, `Report*`, статистика
   - `buffer_config.cpp` — чтение env (`ChunkUploads`, `LayerMode`, `StreamBufferSize`, …)
2. **`layer/buffer_cache_layer.cpp`** →
   - `layer_bind.cpp` — `LayerBind` и bind memo
   - `layer_promote.cpp` — promote, merge, demote, growth
   - `layer_mirror_pool.cpp` — создание, освобождение, обрезка и уплотнение mirror'ов
   - `layer_maintain.cpp` — `LayerMaintain`, `LayerProcessIdle`, `LayerYieldToImages`
   - `layer_paged.cpp` — записи страниц, paged record/verify
   - `layer_traps.cpp` — ловушки, пробы, read traps, verify
3. **`shim/bbport_guest_hooks.cpp`** → новая папка `shim/hooks/`:
   - `detour.cpp` — `InstallDetour`, `InstallCallDetour`, стабы, прыжки
   - `heap_grow.cpp` — рост и сжатие кучи, трассировка, `HeapFind`
   - `effects.cpp` — горячие эффекты, заглушка FXAA, `FxInitClear`
   - `camera.cpp` — аспект, свободная камера, обзор мышью, viewport, HUD, маркеры
   - `jobs.cpp` — job-слоты, проба аллокаций, кэш ресурсов
   - `gx_guards.cpp` — `GxNode`, `GxQuery`
   - `input_probes.cpp` — analog, stick, entry
   - `install.cpp` — `Install`, `OnTrap`
4. **`upscale/vk_temporal_upscaler.cpp`** →
   - `upscaler_dump.cpp`
   - `upscaler_resources.cpp` — `EnsureResources`, `CreatePipelines`
   - `upscaler_passes.cpp` — TAA, EASU, резкость, reactive, SMAA
   - `upscaler_redirect.cpp` — `Mirror`, `Redirect*`, `OnDraw`, `OnColorTarget`
   - `upscaler_ui.cpp` — `EnsureUiResources`, `PrepareUiDepth`, `RunUiOnly`
   - `upscaler_backends.cpp` — FSR4, DLSS
   - основной `Run` и `RunScaled` остаются в исходном файле
5. **`render/shader_rebuild.cpp`** →
   - `shader_recipes_io.cpp` — пакеты, экспорт и импорт
   - `shader_rebuild.cpp` — пересборка кэшей
   - `shader_predict.cpp` — `PredictTest`, `Spec*`
   - `shader_meta_merge.cpp` — `MergeMetas`, `MetaReconstructTest`
6. **`render/pipeline_cache.cpp`** →
   - `pipeline_key.cpp` — `RefreshGraphicsKey`, `RefreshGraphicsStages`, `BuildRuntimeInfo`, `MakeProfile`
   - `pipeline_programs.cpp` — `GetProgram`, `LookUpProgram`, fetch-код, `SpecSignature`
   - `pipeline_compile.cpp` — фоновая компиляция, очередь предзагрузки
   - `pipeline_import.cpp` — `StartShaderImport`, `MergeShaderImport`
7. Второй заход: `src/runtime_memory.c`, `src/runtime_pad.c`, `shim/bbport_guest_memory.cpp`,
   `render/rasterizer.cpp`, `launcher/bbport_launcher.py` (вкладки и страницы в отдельные модули).

Каждому новому файлу — 2–3 строки в начале: что в нём и с какими файлами он связан.

### Этап 2. Длинные функции → именованные шаги

Длинная функция становится коротким «оглавлением» из вызовов `Step1()`, `Step2()`, … в том же
порядке. Сначала холодные функции: `Menu` (по разделам меню), `CreateDevice` (цепочки фич),
`OnFlipStats`, `RunUiOnly`, `MergeMetas`, `LayerMaintain`, `probe.c main`.

**Горячие функции** (`BindTextures`, `LayerBind`, `PostDraw`, `CollectRingBindings`,
`CommandProcessor::Graphics`, `RunDrawPacket`) — только с `static inline` или
`[[gnu::always_inline]]`. После каждой нужен замер стоимости draw на слабом железе: вызов функции
на Deck/ноутбуке — это заметные проценты. `EmitVectorAlu` (switch по опкодам) не трогаем: так
читается нормально.

### Этап 3. Настройки (`BB_*`) в одном месте

- `common/knobs.h` и `knobs.cpp`: одна таблица (имя, тип, значение по умолчанию, раздел, одна
  строка описания). Значение читается один раз и кэшируется. Вместо `getenv` в коде —
  `Knobs::Get(Knob::MirrorFloorPercent)`.
- `docs/KNOBS.ru.md` генерируется из этой таблицы: полный список настроек с описаниями.
- Разделы: настройки игрока / диагностика / эксперименты.
- **Удаление ручек закончившихся экспериментов** — отдельным списком на твоё одобрение, по
  группам. Ничего не удаляю молча.

### Этап 4. Мёртвый код (только с твоего решения)

- `gpu/bbnet/` (старый онлайн-модуль, 147 файлов) и `gpu/bbnet20/`: BBSC 0.0.2 уже перенесён в
  bbnet20. Нужен ли ещё старый модуль?
- Остатки старой модели памяти, которые оставлены намеренно (readbacks, `memory_tracker`,
  sparse arena при `BB_LAYER_MEMORY=0`): оставить или убрать.

### Этап 5. Порядок в корне проекта и карта модулей

- В корне лежат логи, `message (6/7/8).txt`, 4 бэкапа `shader_recipes.*.bin`,
  `cusa-cache (1).tar.gz`, `Shaders.zip` и т.п. Они переедут в `../bbport-archive`, как
  20.10 (не удаляю).
- `docs/ARCHITECTURE.ru.md`: карта модулей — путь кадра (PM4 → стадия A → стадия B →
  Vulkan-рекордер → презентер), что за что отвечает и где искать.

## Как проверяю

- **Сборка:** каждый шаг собирается нативно под Windows (MSYS2 CLANG64), без предупреждений там,
  где их не было.
- **Тесты:** проходят `tests/`.
- **Linux-код:** код под `#ifdef __linux__` здесь не компилируется. Его проверю сборкой в WSL
  (нужно поставить nix в WSL) или ты соберёшь на Linux-машине.
- **Ранний запуск:** до появления игры работает только `bb-gpu-capabilities`. Когда перенесёшь
  дамп, будет один заход в игру после этапа 1 и после этапа 2, плюс замер стоимости draw после
  горячих функций.
- **PGO:** профиль частично устареет (статические функции переехали). Переобучу после этапа 2.
- **Git:** ничего не публикую до твоей проверки.

## Порядок относительно плана B

Этап 1 → (по желанию этап 3) → план B (создание и привязка изображений в отдельном потоке) →
этап 2 и дальше. План B затрагивает `textures/` (`image.cpp`, `texture_cache*`). Этих файлов нет
среди самых больших, поэтому этап 1 ему не мешает.

## Что нужно решить тебе

1. Одобряешь этапы 1–2 в таком виде (имена файлов и границы)?
2. Этап 3 (настройки): делаем? Удаление старых ручек — списком на одобрение?
3. Этап 4: старый `gpu/bbnet/` — удалить или оставить?
4. Этап 5: переносить мусор из корня в `../bbport-archive`?
