# GitHub: разбор 26 октября 2026

Снимок через GitHub REST API (только чтение) на 26.10 ~17:30: **136 открытых issues, 24 открытых
PR**, с 17.10 закрыто 65 (в основном при релизе 0.6), 185 новых комментариев. Сырые данные — в
`scratchpad` сессии (issues/PR/комментарии в JSON). Статус «локально» — по журналам сессий и
памяти: изменения после 0.6, ещё не опубликованы.

Ничего не закрыто и не прокомментировано — это список к релизу.

## Pull requests (24)

### Уже влиты локально — закрыть при релизе с благодарностью и авторством (Co-authored-by)

| PR | Автор | Что | Состояние |
|---|---|---|---|
| [#343](https://github.com/deadinside28/bloodborne_pc/pull/343) | whp199 | защита GX occlusion query (краш 0x2169edc, issue #336) | влит 22.10 (теперь `games/bloodborne/hooks_gx.cpp`) |
| [#342](https://github.com/deadinside28/bloodborne_pc/pull/342) | whp199 | BBSC: hunter's kit, Kin Bell, призывы | влит 22.10 **частично**: без его memory.h/glue (у нас своё решение #282), патчи зон знаков/призывов взяты |
| [#346](https://github.com/deadinside28/bloodborne_pc/pull/346) | DominickCobb-rs | BBSC выкл. → вернуть ванильные колокола | влит 22.10 (+ правка printf под Windows) |
| [#339](https://github.com/deadinside28/bloodborne_pc/pull/339) | maiconjsv | test_online с путём модуля | влит 22.10; после удаления старого модуля тест проверяет только libbbnet20 |
| [#293](https://github.com/deadinside28/bloodborne_pc/pull/293) | iannitram | FSR 3.1 не в protected memory (issue #85, ROG Ally) | влит 19.10 ночью |
| [#260](https://github.com/deadinside28/bloodborne_pc/pull/260) | adds39939 | DLSS Frame Generation + Reflex | влит 19.10 ночью (+ AMD Anti-Lag 2 сверху) |
| [#225](https://github.com/deadinside28/bloodborne_pc/pull/225) | AndiMajore | present mode FifoRelaxed | влит 19.10 ночью |
| [#157](https://github.com/deadinside28/bloodborne_pc/pull/157) | Qiming-Liu | китайский перевод лаунчера | влит 19.10 ночью |

### Устарели (код 0.51 / `gpu/shadps4`, которого больше нет) — закрыть с объяснением

| PR | Что | Почему |
|---|---|---|
| [#226](https://github.com/deadinside28/bloodborne_pc/pull/226) | утечка кадров корутин liverpool | автор сам пишет: только 0.51, в 0.6 командный процессор другой |
| [#223](https://github.com/deadinside28/bloodborne_pc/pull/223) | RTAO (лучевой AO) | на `gpu/shadps4`; автор (AndiMajore) портировал на 0.6, но PR ещё не открыл — ждать новый PR |
| [#167](https://github.com/deadinside28/bloodborne_pc/pull/167) | замена шейдера по хэшу (shader patches) | на старом pipeline cache; идею можно перенести, если нужна |
| [#138](https://github.com/deadinside28/bloodborne_pc/pull/138) | Windows-порт на 0.51 (Ifjy) | своя Windows-сборка на новой модели памяти |
| [#127](https://github.com/deadinside28/bloodborne_pc/pull/127) | «Trophies» (огромная Windows-ветка) | трофеи уже взяты из коммита guutol; сам PR — старая ветка на 301 файл |
| [#24](https://github.com/deadinside28/bloodborne_pc/pull/24) | мышь и раскладка (draft) | нативный обзор мышью и раскладка уже есть |
| [#6](https://github.com/deadinside28/bloodborne_pc/pull/6) | Windows-порт (yumlevi) | см. #138 |
| [#50](https://github.com/deadinside28/bloodborne_pc/pull/50) | «Thhee» | без описания, копия старого дерева |

### Нужно решение

| PR | Автор | Что | Комментарий |
|---|---|---|---|
| [#364](https://github.com/deadinside28/bloodborne_pc/pull/364) | AndiMajore | трофеи: хранение по пользователю, уведомления в игре с названиями и иконками (нужен ключ трофеев игрока) | **пересекается** с нашим портом трофеев guutol (`runtime_trophy.c`, `trophies.jsonl`); надо решить, чьё хранение оставить, а уведомления/иконки взять |
| [#360](https://github.com/deadinside28/bloodborne_pc/pull/360) | wheat32 | большие графические кучи для текстур-паков (автоопределение пака, `BB_DMEM_MB=9152`) | закрывает #240 и #283 (тот же автор подтвердил обходной путь); проверить, не конфликтует ли с нашим ростом кучи GraphicsPrivate по требованию |
| [#296](https://github.com/deadinside28/bloodborne_pc/pull/296) | whp199 | bbnet20: понятные сообщения входа, без паролей в логах, аренда UPnP | не влит; полезно (пароли в логах, которые прикладывают к issues); проверить конфликты с портом BBSC 0.0.2 от 25.10 |
| [#347](https://github.com/deadinside28/bloodborne_pc/pull/347) | Sarxzer | французский перевод лаунчера | простой; конфликт только с правками i18n |
| [#156](https://github.com/deadinside28/bloodborne_pc/pull/156) | AyrtonSanabio | португальский (бразильский) перевод лаунчера | то же |
| [#165](https://github.com/deadinside28/bloodborne_pc/pull/165) | AndiMajore | `BB_FAULT_WAIT=1` — процесс ждёт gdb при падении | маленький; пути файлов от 0.51 (`gpu/shim/common/assert.cpp`), переносить вручную |
| [#79](https://github.com/deadinside28/bloodborne_pc/pull/79) | Convez | сборка через nix flake | по желанию |
| [#8](https://github.com/deadinside28/bloodborne_pc/pull/8) | yogesh1239 | FSR 4.1.1 FP8 на RDNA4 | FP8-вариант для RDNA4 у нас уже есть (сделан отдельно) — скорее всего закрыть, указав это |

## Issues: исправлено локально после 0.6 (закрыть при релизе, попросить перепроверить)

| Issue | Что | Что сделано |
|---|---|---|
| [#307](https://github.com/deadinside28/bloodborne_pc/issues/307) | номер версии в лаунчере | файл VERSION (0.7-dev), лаунчер/run.sh/run.py/Windows |
| [#304](https://github.com/deadinside28/bloodborne_pc/issues/304), [#334](https://github.com/deadinside28/bloodborne_pc/issues/334) | меню оверлея по L3+R3 срабатывает в бою | открывается только при удержании L3+R3 0.8 с (#334 просит ещё отключение/переназначение — частично) |
| [#290](https://github.com/deadinside28/bloodborne_pc/issues/290) | описания клавиш не на английском | 15 строк дополнены |
| [#299](https://github.com/deadinside28/bloodborne_pc/issues/299), [#366](https://github.com/deadinside28/bloodborne_pc/issues/366) | выбор иконок кнопок | `button_prompts=auto/playstation/xbox/keyboard` в bbport.ini и лаунчере (#366 — неверные иконки до первого нажатия; принудительный выбор обходит) |
| [#291](https://github.com/deadinside28/bloodborne_pc/issues/291) | рамки на Deck при 1280×800 | полноэкранный режим принудительно под gamescope |
| [#269](https://github.com/deadinside28/bloodborne_pc/issues/269), [#321](https://github.com/deadinside28/bloodborne_pc/issues/321) | китайский текст, японская озвучка, языки из-за региона | таблицы языков по регионам найдены; `game_text_language`/`game_voice_language` (японская озвучка на слух не проверена) |
| [#234](https://github.com/deadinside28/bloodborne_pc/issues/234), [#328](https://github.com/deadinside28/bloodborne_pc/issues/328) | мыло на Wayland при 200 % | SDL high pixel density + `SDL_VIDEO_WAYLAND_SCALE_TO_DISPLAY=1` по умолчанию |
| [#282](https://github.com/deadinside28/bloodborne_pc/issues/282), [#313](https://github.com/deadinside28/bloodborne_pc/issues/313) | BBSC на Linux: нет набора колоколов, `?GoodsName?` | чтение образа игры для BBSC (своя таблица low-range) + PR #342/#346; ты подтвердил «соединение появилось» 22.10 |
| [#336](https://github.com/deadinside28/bloodborne_pc/issues/336) | краш 0x2169edc (GX query) | PR #343 |
| [#213](https://github.com/deadinside28/bloodborne_pc/issues/213) | зависание при выборе апскейлера в меню (Deck) | контексты FSR 4/4.1.1 создаются в фоне |
| [#85](https://github.com/deadinside28/bloodborne_pc/issues/85) | FSR 3.1 мусор на ROG Ally | PR #293 |
| [#289](https://github.com/deadinside28/bloodborne_pc/issues/289) | краш с mako FG | слой Anti-Lag на Linux только при `BB_ANTI_LAG=1` (в коде помечено #289) |
| [#212](https://github.com/deadinside28/bloodborne_pc/issues/212) | моды не откатываются | бэкап состояния модов (`mods.last`) |
| [#358](https://github.com/deadinside28/bloodborne_pc/issues/358) | Deck: переключение мышь/геймпад дёргает камеру | события клавиатуры/мыши от Steam Input отбрасываются, пока оверлей закрыт (23.10) — вероятно, оно |
| [#345](https://github.com/deadinside28/bloodborne_pc/issues/345) | FSR 4.1.2 / драйверные DLL для RDNA2 | модели INT8 RDNA2 / Lite из `amdxcffx64.dll` (23.10, тобой не проверено) |
| [#368](https://github.com/deadinside28/bloodborne_pc/issues/368), [#253](https://github.com/deadinside28/bloodborne_pc/issues/253) | «взрыв вершин» после возврата во Сон | привязки через границу кусков растущей кучи читали устаревшую копию; исправлено 25.10 (`LayerStitch`), не проверено |

## Issues: вероятно уже исправлено, нужна перепроверка у авторов

- Жесты не открываются: [#241](https://github.com/deadinside28/bloodborne_pc/issues/241), [#188](https://github.com/deadinside28/bloodborne_pc/issues/188), [#180](https://github.com/deadinside28/bloodborne_pc/issues/180), [#196](https://github.com/deadinside28/bloodborne_pc/issues/196) — в текущей сборке работают (touchpad left / Back).
- Падение при старте на NVIDIA, RIP=0: [#181](https://github.com/deadinside28/bloodborne_pc/issues/181), [#219](https://github.com/deadinside28/bloodborne_pc/issues/219), [#229](https://github.com/deadinside28/bloodborne_pc/issues/229) — исправление через `GBM_BACKENDS_PATH` (19.10), на NVIDIA не проверено; новые комментарии 21–23.10 всё ещё на 0.6.
- [#238](https://github.com/deadinside28/bloodborne_pc/issues/238) Хемвик: перестают грузиться объекты — кэш ресурсов рендера (исправлено 19.10).
- [#155](https://github.com/deadinside28/bloodborne_pc/issues/155) обвал FPS после создания персонажа (= смена сохранения, исправлено).
- [#227](https://github.com/deadinside28/bloodborne_pc/issues/227) copy shader 1.65 мс — больше не в топе.
- [#231](https://github.com/deadinside28/bloodborne_pc/issues/231) Vulkan в `bbport_dlss` — исправлено.
- [#92](https://github.com/deadinside28/bloodborne_pc/issues/92) мерцание клинка Людвига (0.4) — исчезает с новой моделью, теперь она по умолчанию.
- [#88](https://github.com/deadinside28/bloodborne_pc/issues/88) VK_KHR_surface (0.4) = #172, закрыт.
- [#143](https://github.com/deadinside28/bloodborne_pc/issues/143) DoF и мерцание неба (0.4) — гонка ConstRam исправлена 07.10.
- [#11](https://github.com/deadinside28/bloodborne_pc/issues/11) мерцают иконки предметов (0.3).
- Старые версии 0.2–0.51 без новых сообщений: [#41](https://github.com/deadinside28/bloodborne_pc/issues/41), [#42](https://github.com/deadinside28/bloodborne_pc/issues/42), [#47](https://github.com/deadinside28/bloodborne_pc/issues/47), [#40](https://github.com/deadinside28/bloodborne_pc/issues/40), [#61](https://github.com/deadinside28/bloodborne_pc/issues/61), [#4](https://github.com/deadinside28/bloodborne_pc/issues/4), [#56](https://github.com/deadinside28/bloodborne_pc/issues/56), [#98](https://github.com/deadinside28/bloodborne_pc/issues/98), [#97](https://github.com/deadinside28/bloodborne_pc/issues/97), [#28](https://github.com/deadinside28/bloodborne_pc/issues/28), [#111](https://github.com/deadinside28/bloodborne_pc/issues/111), [#169](https://github.com/deadinside28/bloodborne_pc/issues/169), [#139](https://github.com/deadinside28/bloodborne_pc/issues/139), [#118](https://github.com/deadinside28/bloodborne_pc/issues/118), [#200](https://github.com/deadinside28/bloodborne_pc/issues/200), [#20](https://github.com/deadinside28/bloodborne_pc/issues/20), [#7](https://github.com/deadinside28/bloodborne_pc/issues/7) — закрыть с просьбой открыть заново на новой версии.

## Issues: открыто, нужна работа

**Падения и зависания**
- [#32](https://github.com/deadinside28/bloodborne_pc/issues/32) **главный открытый краш**: 0x263b8e7, адрес 0x5300000000 (порча пула, все отчёты на AMD, в т.ч. Deck). Новая зацепка (rcopra, 21.10): весь сеанс кэш текстур выше критической отметки, 241–1032 вытеснения прямо перед падением → путь вытеснения GC. Двухчасовые прогоны на Deck не поймали.
- [#369](https://github.com/deadinside28/bloodborne_pc/issues/369) **новый (25.10)**: assert «flip interrupt out of order [videoout.cpp:394]» через ~1 с после автосохранения (RTX 2080 SUPER, 0.6, DLSS, 60 FPS, Hyprland, второй монитор). Подробный отчёт.
- [#355](https://github.com/deadinside28/bloodborne_pc/issues/355) краш 0.6 (только лог; в логе `m27_9999.tpf.dcx not found`, как в #240).
- [#340](https://github.com/deadinside28/bloodborne_pc/issues/340), [#341](https://github.com/deadinside28/bloodborne_pc/issues/341), [#359](https://github.com/deadinside28/bloodborne_pc/issues/359) Intel UHD 620 / ноутбуки без дискретной карты: device lost при submit, код 23.
- [#262](https://github.com/deadinside28/bloodborne_pc/issues/262) краш в `ucrtbase.dll` (bb-probe.exe, 18.10) — это Windows-форк; новый комментарий 25.10 с Intel 12700K тоже про Windows.
- [#261](https://github.com/deadinside28/bloodborne_pc/issues/261) краш при запуске (NVIDIA 595 CreateDevice).
- [#184](https://github.com/deadinside28/bloodborne_pc/issues/184) NVIDIA Xid 109 в Гаоле/Ярнаме только с DoF + motion blur (новое подтверждение на 0.6, 25.10).
- [#344](https://github.com/deadinside28/bloodborne_pc/issues/344) мёртвые враги проваливаются под землю — есть и в оригинале, но у нас чаще (физика при высоком FPS?).

**Память и производительность**
- [#335](https://github.com/deadinside28/bloodborne_pc/issues/335) «утечка» VRAM за час на RTX 5060 Ti 8 GB.
- [#332](https://github.com/deadinside28/bloodborne_pc/issues/332) RX 580: плохой frame pacing из-за вытеснения текстур (обход от игрока: `BB_GUEST_GPU_MEMORY=0`).
- [#315](https://github.com/deadinside28/bloodborne_pc/issues/315) одно ядро 100 %, GPU 50 % (Гаскойн, Амелия, Хемвик) — упор в стадию A/B на CPU.
- [#240](https://github.com/deadinside28/bloodborne_pc/issues/240), [#283](https://github.com/deadinside28/bloodborne_pc/issues/283) краш/мерцание в клинике Йосефки с текстур-паками → PR #360.
- Polaris (RX 570/580): [#131](https://github.com/deadinside28/bloodborne_pc/issues/131), [#105](https://github.com/deadinside28/bloodborne_pc/issues/105) — старая модель удалена, остаётся только новая.

**Графика**
- [#308](https://github.com/deadinside28/bloodborne_pc/issues/308) размытие мира после спавна — воспроизведено 25.10: motion blur самой игры с неверной предыдущей камерой; гипотеза — устаревшее VRAM-зеркало константного буфера. Не исправлено.
- [#365](https://github.com/deadinside28/bloodborne_pc/issues/365) артефакты дыма Boom Hammer (FSR 4.1.1, Deck) — возможно, reactive mask (исправление 25.10, не проверено).
- [#237](https://github.com/deadinside28/bloodborne_pc/issues/237) мерцание краёв с DLSS FG.
- [#194](https://github.com/deadinside28/bloodborne_pc/issues/194) красный/жёлтый экран в Запретном лесу.
- [#182](https://github.com/deadinside28/bloodborne_pc/issues/182) AMD BC-250: оранжевая кайма, серые квадраты.
- [#18](https://github.com/deadinside28/bloodborne_pc/issues/18) гостинг (сбор мест).

**Управление и прочее**
- [#361](https://github.com/deadinside28/bloodborne_pc/issues/361) задние кнопки и тачпады Deck — только через Steam Input.
- [#325](https://github.com/deadinside28/bloodborne_pc/issues/325) камера на Xbox-геймпаде при 90+ FPS — не воспроизводится; у автора нужен `BB_PAD_TRACE=2 BB_CAMLOG=1`.
- [#208](https://github.com/deadinside28/bloodborne_pc/issues/208) вертикальная скорость камеры выше горизонтальной — так в самой игре (338°/с против 159°/с); скорости X/Y уже есть в лаунчере → ответить.
- [#354](https://github.com/deadinside28/bloodborne_pc/issues/354) THD: переподключение из главного меню не работает.
- [#356](https://github.com/deadinside28/bloodborne_pc/issues/356) пресеты персонажа не сохраняются, модель зависает.
- [#362](https://github.com/deadinside28/bloodborne_pc/issues/362) нет .desktop при сборке AppImage из исходников через `packaging/appimage.sh` (по журналу 21.10 в нашем AppImage он есть — проверить именно этот путь сборки); старые: [#287](https://github.com/deadinside28/bloodborne_pc/issues/287), [#132](https://github.com/deadinside28/bloodborne_pc/issues/132), [#133](https://github.com/deadinside28/bloodborne_pc/issues/133).
- Steam overlay / MangoHud в AppImage: [#217](https://github.com/deadinside28/bloodborne_pc/issues/217), [#77](https://github.com/deadinside28/bloodborne_pc/issues/77), [#66](https://github.com/deadinside28/bloodborne_pc/issues/66) — отключено намеренно (нужен libGL хоста).
- [#350](https://github.com/deadinside28/bloodborne_pc/issues/350) «failed to start the embedded python interpreter» — по словам автора, лаунчер испортился при обновлении во время игры; похоже на чужую сборку.

## Запросы функций

[#363](https://github.com/deadinside28/bloodborne_pc/issues/363) лаунчер на RmlUi · [#357](https://github.com/deadinside28/bloodborne_pc/issues/357) мод от глюка Изза · [#319](https://github.com/deadinside28/bloodborne_pc/issues/319) иконки DS3 · [#298](https://github.com/deadinside28/bloodborne_pc/issues/298) 16:10 (уже работает, ответили в комментариях) · [#233](https://github.com/deadinside28/bloodborne_pc/issues/233) предкомпиляция шейдеров (сделано: рецепты) · [#220](https://github.com/deadinside28/bloodborne_pc/issues/220), [#201](https://github.com/deadinside28/bloodborne_pc/issues/201) меню читов (было в форке) · [#198](https://github.com/deadinside28/bloodborne_pc/issues/198) DLC на CUSA00900 · [#176](https://github.com/deadinside28/bloodborne_pc/issues/176), [#148](https://github.com/deadinside28/bloodborne_pc/issues/148) запуск без лаунчера (`--play`) · [#168](https://github.com/deadinside28/bloodborne_pc/issues/168) GTAO (сделан VBAO) · [#160](https://github.com/deadinside28/bloodborne_pc/issues/160) описание репозитория · [#150](https://github.com/deadinside28/bloodborne_pc/issues/150) ключ трофеев (PR #364) · [#94](https://github.com/deadinside28/bloodborne_pc/issues/94) DLSS/FSR (сделано) · [#73](https://github.com/deadinside28/bloodborne_pc/issues/73) комбинации кнопок · [#72](https://github.com/deadinside28/bloodborne_pc/issues/72) macOS (есть форк) · [#46](https://github.com/deadinside28/bloodborne_pc/issues/46) LSFG-VK (в комментариях: работает через `LSFGVK_PROFILE`) · [#37](https://github.com/deadinside28/bloodborne_pc/issues/37), [#21](https://github.com/deadinside28/bloodborne_pc/issues/21) Windows и другие платформы (Windows-сборка готовится) · [#25](https://github.com/deadinside28/bloodborne_pc/issues/25) система сборки · [#16](https://github.com/deadinside28/bloodborne_pc/issues/16) карта скоростей игры для motion vectors · [#255](https://github.com/deadinside28/bloodborne_pc/issues/255), [#58](https://github.com/deadinside28/bloodborne_pc/issues/58) физика при FPS выше 30/60 (движок игры).

## Не про этот проект / мета

- Форки: [#250](https://github.com/deadinside28/bloodborne_pc/issues/250) (Jgamrot), [#242](https://github.com/deadinside28/bloodborne_pc/issues/242), [#114](https://github.com/deadinside28/bloodborne_pc/issues/114) (Windows-форки), [#262](https://github.com/deadinside28/bloodborne_pc/issues/262).
- Неподдерживаемые CPU без BMI1/MOVBE: [#221](https://github.com/deadinside28/bloodborne_pc/issues/221), [#26](https://github.com/deadinside28/bloodborne_pc/issues/26), [#230](https://github.com/deadinside28/bloodborne_pc/issues/230) (Xeon/i7-3770; в #26 есть патч probe.c от игроков).
- Поддержка: [#329](https://github.com/deadinside28/bloodborne_pc/issues/329) (libcuda, ответ дан: #135), [#82](https://github.com/deadinside28/bloodborne_pc/issues/82), [#87](https://github.com/deadinside28/bloodborne_pc/issues/87), [#107](https://github.com/deadinside28/bloodborne_pc/issues/107), [#187](https://github.com/deadinside28/bloodborne_pc/issues/187), [#189](https://github.com/deadinside28/bloodborne_pc/issues/189), [#216](https://github.com/deadinside28/bloodborne_pc/issues/216) (не воспроизводится).
- Мета и вопросы: [#222](https://github.com/deadinside28/bloodborne_pc/issues/222) (твоё объявление — закрепить?), [#300](https://github.com/deadinside28/bloodborne_pc/issues/300) (авторство PR — сделать при пуше), [#65](https://github.com/deadinside28/bloodborne_pc/issues/65), [#23](https://github.com/deadinside28/bloodborne_pc/issues/23), [#5](https://github.com/deadinside28/bloodborne_pc/issues/5).

## Что взять в работу (предложение)

1. [#369](https://github.com/deadinside28/bloodborne_pc/issues/369) — assert порядка flip-прерываний после автосохранения: свежий, воспроизводимый, с полным отчётом.
2. [#32](https://github.com/deadinside28/bloodborne_pc/issues/32) — порча пула: проверить зацепку с вытеснением GC (кэш весь сеанс выше критической отметки).
3. [#308](https://github.com/deadinside28/bloodborne_pc/issues/308) — размытие после телепорта: проверить гипотезу с устаревшим зеркалом константного буфера.
4. PR [#360](https://github.com/deadinside28/bloodborne_pc/pull/360) и [#296](https://github.com/deadinside28/bloodborne_pc/pull/296) — посмотреть и влить, если не конфликтуют.
5. PR [#364](https://github.com/deadinside28/bloodborne_pc/pull/364) — решить про трофеи (их уведомления поверх нашего хранения?).
