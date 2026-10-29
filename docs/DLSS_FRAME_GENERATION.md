# DLSS Frame Generation and NVIDIA Reflex on Linux (2026-10-18)

## DLSS Frame Generation (2026-10-18)

NVIDIA's DLSS Frame Generation runs natively on Linux through the DLSS bridge, on a GeForce RTX 40
or newer with NVIDIA's `libnvidia-ngx-dlssg.so.<version>` from the DLSS SDK (310.7 or newer).
Between each two of the game's frames one generated frame is shown (RTX 40), or up to three
(RTX 50, *Multiplier* 3x/4x). Works with every upscaler (DLSS or FSR) and every preset except Off.

- **Where:** the overlay menu (Insert or L3+R3) → *Frame generation → DLSS (NVIDIA RTX 40/50)*;
  the launcher → *Upscaler → Frame generation*; `BB_FRAME_GEN=dlss`, `BB_FRAME_GEN_MULTIPLIER=2..4`.
  `bbport.ini`: `frame_generation=off|fsr3|dlss`, `frame_generation_multiplier=2..4`.
- **The library:** `DLSS_SDK_ROOT=... bash build.sh` copies it next to `bb-probe` with the Super
  Resolution one (the AppImage carries both); the launcher's *DLSS: NVIDIA's library → Choose a
  file…* takes it too, into `<user>/dlss/`. The game searches both folders.
- **Inputs:** the finished frame (UI included), the frame before its UI, a copy of the scene
  depth, the motion vectors bbport computes (camera and object motion), the jitter and the game's
  cameras as D3D clip-space matrices (`gpu/upscale/camera_matrices.h`, tested without a GPU by
  `camera-matrices-test`). A camera that stood still smeared the generated frames on every turn;
  the matrices let DLSS separate the camera's motion from the objects'.
- **Without it:** on other GPUs, without the library or the bridge, or with `BB_DLSS=0`, the menu
  entry says why, and a saved DLSS choice is kept but does nothing. If a generation fails, the
  real frame is shown twice instead of garbage, and DLSS Frame Generation switches itself off
  with the reason in the menu.
- **Bridge ABI 2** (`gpu/dlss_bridge`, MIT): `FrameGenMaxCount`, `FrameGenCreate`,
  `FrameGenEvaluate`, `FrameGenRelease`, and `ConfigureDirectories` (several folders for NGX to
  search). An ABI 1 bridge still gives DLSS Super Resolution.
- FSR 3.1 frame generation stays hidden until its artifacts are fixed (`BB_FRAME_GEN=1` shows it);
  the presenter now paces up to three generated frames evenly before the real one.

Not yet verified on hardware by the author (no NVIDIA GPU at hand): the Vulkan path follows the
Windows port's working implementation (Supermedo/bloodborne_pc) and NVIDIA's Linux SDK; reports
from RTX 40/50 owners are welcome.

## Low latency: NVIDIA Reflex through VK_NV_low_latency2 (2026-10-18)

On NVIDIA's driver (550 and newer) the game's thread sleeps at the start of each of its frames
until the driver wants the frame started (`vkLatencySleepNV`, at the frame-rate hooks' pacer
step; without the hooks right after the flip), and the frame's simulation, render submissions and
present carry one ID through the driver's markers, `VkLatencySubmissionPresentIdNV` and
`VK_KHR_present_id`. Generated frames are presented out of band. `gpu/render/vk_low_latency`.

- *Overlay menu → Output → Low latency (NVIDIA Reflex)* and the launcher: Off, On (default where
  available), On + Boost. `bbport.ini`: `low_latency=off|on|boost`; `BB_LOW_LATENCY`. With DLSS
  frame generation it is always at least On (the overlay shows that and greys Off out; the
  launcher's row follows). Nothing changes on GPUs and drivers without the extension; the entry
  says so on an NVIDIA GPU with an old driver.
- Not yet measured on hardware by the author; what to compare on an RTX card: input lag with the
  setting Off and On at the same frame rate, with and without frame generation.

## Русский

Генерация кадров DLSS (DLSS Frame Generation) работает нативно на Linux через мост DLSS, на
GeForce RTX 40 и новее с библиотекой NVIDIA `libnvidia-ngx-dlssg.so.<версия>` из DLSS SDK (310.7
или новее). Между каждыми двумя кадрами игры показывается один сгенерированный (RTX 40) или до
трёх (RTX 50, *Множитель* 3x/4x). Работает с любым апскейлером (DLSS или FSR) и любым пресетом,
кроме «Выкл».

- **Где:** меню-оверлей (Insert или L3+R3) → *Генерация кадров → DLSS (NVIDIA RTX 40/50)*;
  лаунчер → *Апскейлер → Генерация кадров*; `BB_FRAME_GEN=dlss`, `BB_FRAME_GEN_MULTIPLIER=2..4`.
  `bbport.ini`: `frame_generation=off|fsr3|dlss`, `frame_generation_multiplier=2..4`.
- **Библиотека:** `DLSS_SDK_ROOT=... bash build.sh` копирует её рядом с `bb-probe` вместе с
  библиотекой Super Resolution (AppImage несёт обе); в лаунчере её можно выбрать так же, *DLSS:
  библиотека NVIDIA → Выбрать файл…*, в `<user>/dlss/`. Игра ищет в обеих папках.
- **Входы:** готовый кадр (с интерфейсом), кадр до интерфейса, копия глубины сцены, векторы
  движения bbport (камера и объекты), jitter и камеры игры в виде матриц clip-пространства D3D
  (`gpu/upscale/camera_matrices.h`, тест `camera-matrices-test` без GPU).
- **Без неё:** на других видеокартах, без библиотеки или моста, с `BB_DLSS=0` пункт меню говорит,
  почему недоступен; сохранённый выбор DLSS остаётся, но ничего не делает. Если генерация не
  удалась, вместо мусора дважды показывается настоящий кадр, и генерация DLSS выключается с
  причиной в меню.
- **ABI 2 моста** (`gpu/dlss_bridge`, MIT): `FrameGenMaxCount`, `FrameGenCreate`,
  `FrameGenEvaluate`, `FrameGenRelease` и `ConfigureDirectories`. Мост ABI 1 по-прежнему даёт
  DLSS Super Resolution.
- Генерация кадров FSR 3.1 остаётся скрытой, пока не исправлены её артефакты (`BB_FRAME_GEN=1`);
  презентер теперь показывает до трёх сгенерированных кадров равномерно перед настоящим.

На железе автором не проверено (нет видеокарты NVIDIA под рукой): путь Vulkan повторяет рабочую
реализацию порта для Windows (Supermedo/bloodborne_pc) и SDK NVIDIA для Linux; нужны отчёты
владельцев RTX 40/50.
