// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: log classes (the part of the GPU library a message comes from).
#pragma once

namespace Common::Log::Class {

constexpr auto Common = "Common";
constexpr auto Common_Filesystem = "Common.Filesystem";
constexpr auto Common_Memory = "Common.Memory";
constexpr auto Config = "Config";
constexpr auto Core = "Core";
constexpr auto Debug = "Debug";
constexpr auto Frontend = "Frontend";
constexpr auto ImGui = "ImGui";
constexpr auto Input = "Input";
constexpr auto Kernel = "Kernel";
constexpr auto Kernel_Event = "Kernel.Event";
constexpr auto KeyManager = "KeyManager";
constexpr auto Lib = "Lib";
constexpr auto Lib_AvPlayer = "Lib.AvPlayer";
constexpr auto Lib_GnmDriver = "Lib.GnmDriver";
constexpr auto Lib_Kernel = "Lib.Kernel";
constexpr auto Lib_VideoOut = "Lib.VideoOut";
constexpr auto Lib_Videodec = "Lib.Videodec";
constexpr auto Loader = "Loader";
constexpr auto Log = "Log";
constexpr auto Render = "Render";
constexpr auto Render_Recompiler = "Render.Recompiler";
constexpr auto Render_Vulkan = "Render.Vulkan";
constexpr auto Tty = "Tty";

// The online module (gpu/bbnet).
constexpr auto Lib_Http = "Lib.Http";
constexpr auto Lib_Http2 = "Lib.Http2";
constexpr auto Lib_Net = "Lib.Net";
constexpr auto Lib_NetCtl = "Lib.NetCtl";
constexpr auto Lib_NpAuth = "Lib.NpAuth";
constexpr auto Lib_NpCommerce = "Lib.NpCommerce";
constexpr auto Lib_NpCommon = "Lib.NpCommon";
constexpr auto Lib_NpManager = "Lib.NpManager";
constexpr auto Lib_NpMatching = "Lib.NpMatching";
constexpr auto Lib_NpMatching2 = "Lib.NpMatching2";
constexpr auto Lib_NpPartner = "Lib.NpPartner";
constexpr auto Lib_NpParty = "Lib.NpParty";
constexpr auto Lib_NpProfileDialog = "Lib.NpProfileDialog";
constexpr auto Lib_NpScore = "Lib.NpScore";
constexpr auto Lib_NpSignaling = "Lib.NpSignaling";
constexpr auto Lib_NpSnsFacebookDialog = "Lib.NpSnsFacebookDialog";
constexpr auto Lib_NpTrophy = "Lib.NpTrophy";
constexpr auto Lib_NpTus = "Lib.NpTus";
constexpr auto Lib_NpWebApi = "Lib.NpWebApi";
constexpr auto Lib_NpWebApi2 = "Lib.NpWebApi2";
constexpr auto Lib_Ssl = "Lib.Ssl";
constexpr auto Lib_Ssl2 = "Lib.Ssl2";
constexpr auto NpHandler = "NpHandler";
constexpr auto ShadNet = "ShadNet";

// More of shadPS4 0.20's classes (the online module, gpu/bbnet20).
constexpr auto Core_Devices = "Core.Devices";
constexpr auto Core_Linker = "Core.Linker";
constexpr auto IPC = "IPC";
constexpr auto Kernel_Fs = "Kernel.Fs";
constexpr auto Kernel_Pthread = "Kernel.Pthread";
constexpr auto Kernel_Sce = "Kernel.Sce";
constexpr auto Kernel_Vmm = "Kernel.Vmm";
constexpr auto Lib_Ajm = "Lib.Ajm";
constexpr auto Lib_AppContent = "Lib.AppContent";
constexpr auto Lib_Audio3d = "Lib.Audio3d";
constexpr auto Lib_AudioIn = "Lib.AudioIn";
constexpr auto Lib_AudioOut = "Lib.AudioOut";
constexpr auto Lib_Camera = "Lib.Camera";
constexpr auto Lib_CommonDlg = "Lib.CommonDlg";
constexpr auto Lib_CompanionHttpd = "Lib.CompanionHttpd";
constexpr auto Lib_CompanionUtil = "Lib.CompanionUtil";
constexpr auto Lib_ContentExport = "Lib.ContentExport";
constexpr auto Lib_DiscMap = "Lib.DiscMap";
constexpr auto Lib_ErrorDialog = "Lib.ErrorDialog";
constexpr auto Lib_Fiber = "Lib.Fiber";
constexpr auto Lib_Font = "Lib.Font";
constexpr auto Lib_FontFt = "Lib.FontFt";
constexpr auto Lib_GameLiveStreaming = "Lib.GameLiveStreaming";
constexpr auto Lib_Hmd = "Lib.Hmd";
constexpr auto Lib_HmdSetupDialog = "Lib.HmdSetupDialog";
constexpr auto Lib_Ime = "Lib.Ime";
constexpr auto Lib_ImeDialog = "Lib.ImeDialog";
constexpr auto Lib_InvitationDialog="Lib.InvitationDialog";
constexpr auto Lib_Jpeg = "Lib.Jpeg";
constexpr auto Lib_LibcInternal = "Lib.LibcInternal";
constexpr auto Lib_Mouse = "Lib.Mouse";
constexpr auto Lib_Move = "Lib.Move";
constexpr auto Lib_MsgDlg = "Lib.MsgDlg";
constexpr auto Lib_Ngs2 = "Lib.Ngs2";
constexpr auto Lib_NpUtility = "Lib.NpUtility";
constexpr auto Lib_Pad = "Lib.Pad";
constexpr auto Lib_PlayGo = "Lib.PlayGo";
constexpr auto Lib_PlayGoDialog = "Lib.PlayGoDialog";
constexpr auto Lib_Png = "Lib.Png";
constexpr auto Lib_Random = "Lib.Random";
constexpr auto Lib_RazorCpu = "Lib.RazorCpu";
constexpr auto Lib_Remoteplay = "Lib.Remoteplay";
constexpr auto Lib_Rtc = "Lib.Rtc";
constexpr auto Lib_Rudp = "Lib.Rudp";
constexpr auto Lib_SaveData = "Lib.SaveData";
constexpr auto Lib_SaveDataDialog = "Lib.SaveDataDialog";
constexpr auto Lib_Screenshot = "Lib.Screenshot";
constexpr auto Lib_SharePlay = "Lib.SharePlay";
constexpr auto Lib_SigninDialog = "Lib.SigninDialog";
constexpr auto Lib_SysModule = "Lib.SysModule";
constexpr auto Lib_SystemGesture = "Lib.SystemGesture";
constexpr auto Lib_SystemService = "Lib.SystemService";
constexpr auto Lib_Usbd = "Lib.Usbd";
constexpr auto Lib_UserService = "Lib.UserService";
constexpr auto Lib_Vdec2 = "Lib.Vdec2";
constexpr auto Lib_Vdecsw = "Lib.Vdecsw";
constexpr auto Lib_VideoRecording = "Lib.VideoRecording";
constexpr auto Lib_Voice = "Lib.Voice";
constexpr auto Lib_VrTracker = "Lib.VrTracker";
constexpr auto Lib_WebBrowserDialog = "Lib.WebBrowserDialog";
constexpr auto Lib_Zlib = "Lib.Zlib";

} // namespace Common::Log::Class
