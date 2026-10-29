# Contributions in 0.7

Pull requests and outside work merged into the local tree since 0.6 (2026-10-19), then the ones
taken into earlier releases. No release commit so far named its contributors, so the 0.7 commit
carries trailers for all of them (below), and GitHub credits each author (GitHub noreply
addresses, from the PRs' authors).

## Since 0.6

| Work | Author | What |
|---|---|---|
| PR #157 | Qiming-Liu | Simplified Chinese launcher translation |
| PR #225 | AndiMajore | swapchain: accept FIFO Relaxed |
| PR #260 | adds39939 | DLSS Frame Generation and NVIDIA Reflex (VK_NV_low_latency2) |
| PR #293 | iannitram | FSR 3.1 never in protected memory (#85) |
| PR #339 | maiconjsv | test_online takes the module path |
| PR #342 | whp199 | BBSC: hunter's kit, Kin Bell signs and summons in every area (merged without its memory.h part) |
| PR #343 | whp199 | guard GX occlusion queries without a handle (crash 0x2169edc, #336) |
| PR #346 | DominickCobb-rs | BBSC off: vanilla bells back |
| PR #347 | Sarxzer | French launcher translation |
| PR #360 | wheat32 | larger graphics heaps for texture packs (#240) |
| #352 / PR #353 | ArenaMMORPG | mouse auto-rotation NOPs (the rest measured, not taken) |
| commit 15cd9b7 (vigil-trophies) | guutol (Luís Gustavo P. Bomfim) | trophies (`runtime_trophy.c`, `trophies.jsonl`) |
| archive bbport-vbao | redditor69 (Discord) | GT-VBAO |
| SrShadowy/bloodborne_pc `testing` | SrShadowy, Teefpyre | play-time carry, hit collector cap, CharaId guard, light grid option, video skip, touchpad, mouse X invert, shader catalog, BBSC 0.0.2 relay and Linux fixes |

## Earlier releases (0.4 – 0.6)

| Work | Author | What | Release |
|---|---|---|---|
| PR #10 | SrDeTs | Brazilian Portuguese launcher translation | 0.4 |
| PR #36 | IgorWarzocha | language selector in the in-game menu | 0.4 |
| PR #43 | itachi-re | repository URL in the README's clone command | 0.4 |
| PR #71 | Bainianzzz | per-controller stick neutral, inner/outer dead zone | 0.5 |
| PR #75 | Qiming-Liu | Simplified Chinese launcher translation | 0.5 |
| PR #78 | hideoneric | Copy and Export log buttons | 0.5 |
| PR #80 | guitaripod | FSR 4 v07: BB_FSR4_STATS pipeline fix | 0.5 |
| PR #83 | unTanana | DLSS Super Resolution on Linux | 0.5 |
| PR #84 | unTanana | add-on folders reported as installed (#33) | 0.5 |
| PR #112 | AndiMajore | fsr4cap finds system-wide Proton | 0.5 |
| PR #113 | AndiMajore | Dear ImGui private to libbbgpu.so | 0.5 |
| PR #89 | unTanana | FSR 4.1.1 recording from the AppImage on SteamOS | 0.51 |
| PR #93 | andrR89 | swapchain retired through oldSwapchain (NVIDIA Wayland crash) | 0.51 |
| PR #117 | ayozetr | game check: every retail release and the 60 FPS eboot | 0.51 |
| PR #120 | JuanraCM | launcher arguments through the nix-shell fallback | 0.51 |
| PR #123 | unTanana | every retail release (CUSA00900 and others) runs the same 1.09 profile | 0.51 |
| PR #108 | SaeedShow | native mouse look and Souls-like bindings (the base of ours) | 0.51 |
| PR #193 | msisovic | guest thread handles below 40 bits (loading deadlocks) | 0.6-pre1 |
| PR #166 | AndiMajore | a SnormNz buffer sharp no longer stops the port | 0.6-pre1 |
| PR #153, #154 | AyrtonSanabio | the in-game menu's language from the launcher (#134) | 0.6-pre1 |
| PR #206 | axpico | wait for evicted textures before failing on full VRAM (#163, #173) | 0.6 |
| PR #210 | whp199 | detiler: tile into VRAM, one copy to host | 0.6 |
| PR #211 | whp199 | no submissions while the swapchain is recreated | 0.6 |
| PR #214 | whp199 | guest hooks with every memory model (#162) | 0.6 |
| PR #224 | AndiMajore | GX per-frame command heap 256 KiB -> 1 MiB (#162) | 0.6 |
| Supermedo/bloodborne_pc (fork) | Supermedo, Awaken (Awakenchan) | the online module's network stack: shadPS4 0.20 network/NP/shadNet put together by Awaken, The Hunter's Dream backend; the Windows launcher's package layout | 0.6, 0.7 |

The same fixes as ours were also proposed in PR #137 (Aryxa, libcuda for DLSS in the AppImage),
PR #140 (Obrikash, device without the optional DLSS extensions) and PR #235 (noelmiller, the
client version for shadNet); they are credited too.

Not merged (open on GitHub, to decide): #364 AndiMajore (trophies per user and in-game toasts;
overlaps guutol's port), #296 whp199 (bbnet20 login notices, no secrets in logs), #165 / #167 /
#223 / #226 AndiMajore, #156 AyrtonSanabio, #138 Ifjy, #127 GerasimosKan.

```
Co-authored-by: Qiming-Liu <68600416+Qiming-Liu@users.noreply.github.com>
Co-authored-by: AndiMajore <18685571+AndiMajore@users.noreply.github.com>
Co-authored-by: adds39939 <37377420+adds39939@users.noreply.github.com>
Co-authored-by: iannitram <4624371+iannitram@users.noreply.github.com>
Co-authored-by: maiconjsv <209799748+maiconjsv@users.noreply.github.com>
Co-authored-by: whp199 <64566213+whp199@users.noreply.github.com>
Co-authored-by: DominickCobb-rs <138266282+DominickCobb-rs@users.noreply.github.com>
Co-authored-by: Sarxzer <80457483+Sarxzer@users.noreply.github.com>
Co-authored-by: wheat32 <5769329+wheat32@users.noreply.github.com>
Co-authored-by: ArenaMMORPG <38107216+ArenaMMORPG@users.noreply.github.com>
Co-authored-by: SrShadowy <43967231+SrShadowy@users.noreply.github.com>
Co-authored-by: Gustavo Bomfim <186610075+guutol@users.noreply.github.com>
Co-authored-by: Teefpyre <185977176+Teefpyre@users.noreply.github.com>
Co-authored-by: SrDeTs <101432439+SrDeTs@users.noreply.github.com>
Co-authored-by: IgorWarzocha <634445+IgorWarzocha@users.noreply.github.com>
Co-authored-by: itachi-re <117917529+itachi-re@users.noreply.github.com>
Co-authored-by: Bainianzzz <95992848+Bainianzzz@users.noreply.github.com>
Co-authored-by: hideoneric <75436909+hideoneric@users.noreply.github.com>
Co-authored-by: guitaripod <47460844+guitaripod@users.noreply.github.com>
Co-authored-by: unTanana <22821635+unTanana@users.noreply.github.com>
Co-authored-by: andrR89 <3686269+andrR89@users.noreply.github.com>
Co-authored-by: ayozetr <53948812+ayozetr@users.noreply.github.com>
Co-authored-by: JuanraCM <44867213+JuanraCM@users.noreply.github.com>
Co-authored-by: SaeedShow <213490996+SaeedShow@users.noreply.github.com>
Co-authored-by: msisovic <95937882+msisovic@users.noreply.github.com>
Co-authored-by: axpico <119961943+axpico@users.noreply.github.com>
Co-authored-by: Aryxa <101952436+Aryxa@users.noreply.github.com>
Co-authored-by: Obrikash <73013361+Obrikash@users.noreply.github.com>
Co-authored-by: noelmiller <4983138+noelmiller@users.noreply.github.com>
Co-authored-by: AyrtonSanabio <142481010+AyrtonSanabio@users.noreply.github.com>
Co-authored-by: Supermedo <28518750+Supermedo@users.noreply.github.com>
Co-authored-by: Awaken <93601997+Awakenchan@users.noreply.github.com>
```

guutol and Teefpyre: their GitHub noreply addresses (ids from the GitHub API). redditor69 has no
GitHub account known to us: credited in the notes and in the overlay text.

After the release, the merged PRs can be closed with thanks (as after 0.6); #364 and #296 need a
decision first.
