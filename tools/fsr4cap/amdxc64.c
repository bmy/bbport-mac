// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: MIT
//
// amdxc64.dll for recording AMD's FSR 4.1 on Windows (tools/fsr4cap/build_assets_win.py): AMD's
// upscaler DLL asks the driver's D3D12 extension module for these interfaces before it offers
// FSR 4.1. Under Proton, Wine's amdxc64 answers; on Windows the recording runs fsr4cap.exe on
// vkd3d-proton (D3D12 over Vulkan, any GPU) with this module next to it, so that every GPU gives
// the same recording as Proton does. What it answers (the interfaces' ids and layouts are AMD's):
//   IAmdExtStub1 / IAmdExtStub2  present, nothing to do;
//   IAmdExtFfxApi                UpdateFfxApiProvider goes to the driver's FSR module,
//                                amdxcffx64.dll (FSR 4.1.2's models: the copy driver_model.py
//                                made sits next to fsr4cap.exe), when FSR4_UPGRADE=1;
//   IAmdExtAntiLagApi            asked of the D3D12 device (vkd3d-proton implements it).
#define COBJMACROS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <unknwn.h>

#include <stdlib.h>
#include <string.h>

static const GUID IID_IAmdExtFfxApi = {0xb58d6601, 0x7401, 0x4234, {0x81, 0x80, 0x6f, 0xeb, 0xfc, 0x0e, 0x48, 0x4c}};
static const GUID IID_IAmdExtAntiLagApi = {0x44085fbe, 0xe839, 0x40c5, {0xbf, 0x38, 0x0e, 0xbc, 0x5a, 0xb4, 0xd0, 0xa6}};
static const GUID IID_IAmdExtStub1 = {0x014937ec, 0x9288, 0x446f, {0xa9, 0xac, 0xd7, 0x5a, 0x8e, 0x3a, 0x98, 0x4f}};
static const GUID IID_IAmdExtStub2 = {0xba019d53, 0xccab, 0x4cbd, {0xb5, 0x6a, 0x72, 0x30, 0xed, 0x43, 0x30, 0xad}};

// IAmdExtFfxApi: IUnknown + UpdateFfxApiProvider(void* data, unsigned size).
typedef struct FfxApi FfxApi;
typedef struct {
    HRESULT(STDMETHODCALLTYPE* QueryInterface)(FfxApi*, REFIID, void**);
    ULONG(STDMETHODCALLTYPE* AddRef)(FfxApi*);
    ULONG(STDMETHODCALLTYPE* Release)(FfxApi*);
    HRESULT(STDMETHODCALLTYPE* UpdateFfxApiProvider)(FfxApi*, void*, unsigned int);
} FfxApiVtbl;
struct FfxApi {
    const FfxApiVtbl* vtbl;
    LONG ref;
};

static HRESULT STDMETHODCALLTYPE FfxQueryInterface(FfxApi* self, REFIID iid, void** out) {
    (void)self;
    (void)iid;
    *out = NULL;
    return E_NOINTERFACE;
}
static ULONG STDMETHODCALLTYPE FfxAddRef(FfxApi* self) {
    return (ULONG)InterlockedIncrement(&self->ref);
}
static ULONG STDMETHODCALLTYPE FfxRelease(FfxApi* self) {
    const LONG ref = InterlockedDecrement(&self->ref);
    if (ref == 0) {
        free(self);
    }
    return (ULONG)ref;
}
typedef HRESULT(__stdcall* UpdateFfxApiProviderFn)(void*, unsigned int);
static HRESULT STDMETHODCALLTYPE FfxUpdateProvider(FfxApi* self, void* data, unsigned int size) {
    (void)self;
    // As Wine's: only with FSR4_UPGRADE=1 (the recording of the driver's models sets it).
    const char* upgrade = getenv("FSR4_UPGRADE");
    if (!upgrade || strcmp(upgrade, "1") != 0) {
        return E_NOINTERFACE;
    }
    HMODULE module = LoadLibraryA("amdxcffx64.dll");
    if (!module) {
        return E_NOINTERFACE;
    }
    UpdateFfxApiProviderFn update = (UpdateFfxApiProviderFn)(void*)GetProcAddress(module, "UpdateFfxApiProvider");
    return update ? update(data, size) : E_NOINTERFACE;
}
static const FfxApiVtbl ffx_vtbl = {FfxQueryInterface, FfxAddRef, FfxRelease, FfxUpdateProvider};

// IAmdExtStub2: IUnknown + three methods the DLL calls and nothing needs.
typedef struct Stub2 Stub2;
typedef struct {
    HRESULT(STDMETHODCALLTYPE* QueryInterface)(Stub2*, REFIID, void**);
    ULONG(STDMETHODCALLTYPE* AddRef)(Stub2*);
    ULONG(STDMETHODCALLTYPE* Release)(Stub2*);
    void(STDMETHODCALLTYPE* Stub)(Stub2*);
    void(STDMETHODCALLTYPE* Stub2)(Stub2*, unsigned int);
    void(STDMETHODCALLTYPE* Stub3)(Stub2*);
} Stub2Vtbl;
struct Stub2 {
    const Stub2Vtbl* vtbl;
    LONG ref;
};
static HRESULT STDMETHODCALLTYPE Stub2QueryInterface(Stub2* self, REFIID iid, void** out) {
    (void)self;
    (void)iid;
    *out = NULL;
    return E_NOINTERFACE;
}
static ULONG STDMETHODCALLTYPE Stub2AddRef(Stub2* self) {
    return (ULONG)InterlockedIncrement(&self->ref);
}
static ULONG STDMETHODCALLTYPE Stub2Release(Stub2* self) {
    const LONG ref = InterlockedDecrement(&self->ref);
    if (ref == 0) {
        free(self);
    }
    return (ULONG)ref;
}
static void STDMETHODCALLTYPE Stub2Nothing(Stub2* self) {
    (void)self;
}
static void STDMETHODCALLTYPE Stub2NothingWith(Stub2* self, unsigned int value) {
    (void)self;
    (void)value;
}
static const Stub2Vtbl stub2_vtbl = {Stub2QueryInterface, Stub2AddRef, Stub2Release,
                                     Stub2Nothing, Stub2NothingWith, Stub2Nothing};

// IAmdExtStub1: IUnknown + QueryInterface2(void*, REFIID, void**), which hands out IAmdExtStub2.
typedef struct Stub1 Stub1;
typedef struct {
    HRESULT(STDMETHODCALLTYPE* QueryInterface)(Stub1*, REFIID, void**);
    ULONG(STDMETHODCALLTYPE* AddRef)(Stub1*);
    ULONG(STDMETHODCALLTYPE* Release)(Stub1*);
    HRESULT(STDMETHODCALLTYPE* QueryInterface2)(Stub1*, void*, REFIID, void**);
} Stub1Vtbl;
struct Stub1 {
    const Stub1Vtbl* vtbl;
};
static HRESULT STDMETHODCALLTYPE Stub1QueryInterface2(Stub1* self, void* unused, REFIID iid, void** out) {
    (void)self;
    (void)unused;
    *out = NULL;
    if (!IsEqualGUID(iid, &IID_IAmdExtStub2)) {
        return E_NOINTERFACE;
    }
    Stub2* stub = calloc(1, sizeof(*stub));
    if (!stub) {
        return E_OUTOFMEMORY;
    }
    stub->vtbl = &stub2_vtbl;
    stub->ref = 1;
    *out = stub;
    return S_OK;
}
static HRESULT STDMETHODCALLTYPE Stub1QueryInterface(Stub1* self, REFIID iid, void** out) {
    return Stub1QueryInterface2(self, NULL, iid, out);
}
static ULONG STDMETHODCALLTYPE Stub1AddRef(Stub1* self) {
    (void)self;
    return 2;
}
static ULONG STDMETHODCALLTYPE Stub1Release(Stub1* self) {
    (void)self;
    return 1;
}
static const Stub1Vtbl stub1_vtbl = {Stub1QueryInterface, Stub1AddRef, Stub1Release, Stub1QueryInterface2};
static Stub1 stub1 = {&stub1_vtbl};

__declspec(dllexport) HRESULT __cdecl AmdExtD3DCreateInterface(IUnknown* outer, REFIID iid, void** out) {
    *out = NULL;
    if (IsEqualGUID(iid, &IID_IAmdExtFfxApi)) {
        FfxApi* ffx = calloc(1, sizeof(*ffx));
        if (!ffx) {
            return E_OUTOFMEMORY;
        }
        ffx->vtbl = &ffx_vtbl;
        ffx->ref = 1;
        *out = ffx;
        return S_OK;
    }
    if (IsEqualGUID(iid, &IID_IAmdExtAntiLagApi)) {
        return outer ? IUnknown_QueryInterface(outer, &IID_IAmdExtAntiLagApi, out) : E_NOINTERFACE;
    }
    if (IsEqualGUID(iid, &IID_IAmdExtStub1)) {
        *out = &stub1;
        return S_OK;
    }
    return E_NOINTERFACE;
}

__declspec(dllexport) HMODULE WINAPI AmdGetDxcModuleHandle(void) {
    return GetModuleHandleA(NULL);
}

// Exported as the driver's module does; never called by the upscaler DLL.
__declspec(dllexport) HRESULT WINAPI GetSettingsBlobsAll(void) {
    return E_NOTIMPL;
}
__declspec(dllexport) HRESULT WINAPI OpenAdapter12(void) {
    return E_NOTIMPL;
}
__declspec(dllexport) HRESULT WINAPI OpenShimInterface(void) {
    return E_NOTIMPL;
}
