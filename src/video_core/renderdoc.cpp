// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/logging/formatter.h"
#include "core/emulator_settings.h"
#include "video_core/renderdoc.h"

#include <atomic>
#include <renderdoc_app.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <filesystem>

namespace VideoCore {

enum class CaptureState {
    Idle,
    StartRequested,
    InProgress,
    StopRequested,
};
static std::atomic<CaptureState> capture_state{CaptureState::Idle};
static std::atomic<u32> screenshot_game_only_count{0};
static std::atomic<u32> screenshot_with_overlays_count{0};

RENDERDOC_API_1_6_0* rdoc_api{};

void LoadRenderDoc() {
    if (!EmulatorSettings.IsRenderdocEnabled()) {
        return;
    }

#ifdef WIN32

    // Check if we are running by RDoc GUI
    HMODULE mod = GetModuleHandleW(L"renderdoc.dll");
    if (!mod) {
        static constexpr wchar_t RenderDocPath[] = LR"(C:\Program Files\RenderDoc\renderdoc.dll)";
        mod = LoadLibraryW(RenderDocPath);
        if (!mod) {
            LOG_ERROR(Render,
                      "Cannot load RenderDoc from C:\\Program Files\\RenderDoc "
                      "(Windows error {})",
                      GetLastError());
        }
    }

    if (mod) {
        const auto RENDERDOC_GetAPI =
            reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(mod, "RENDERDOC_GetAPI"));
        if (!RENDERDOC_GetAPI) {
            LOG_ERROR(Render, "Loaded RenderDoc module does not export RENDERDOC_GetAPI");
            return;
        }
        const s32 ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
        ASSERT(ret == 1);
    }
#else
#ifdef ANDROID
    static constexpr const char RENDERDOC_LIB[] = "libVkLayer_GLES_RenderDoc.so";
#else
    static constexpr const char RENDERDOC_LIB[] = "librenderdoc.so";
#endif
    // Check if we are running by RDoc GUI
    void* mod = dlopen(RENDERDOC_LIB, RTLD_NOW | RTLD_NOLOAD);
    if (!mod && EmulatorSettings.IsRenderdocEnabled()) {
        // If enabled in config, try to load RDoc runtime in offline mode
        if ((mod = dlopen(RENDERDOC_LIB, RTLD_NOW))) {
            const auto RENDERDOC_GetAPI =
                reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"));
            const s32 ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
            ASSERT(ret == 1);
        } else {
            LOG_ERROR(Render, "Cannot load RenderDoc: {}", dlerror());
        }
    }
#endif
    if (rdoc_api) {
        // Disable default capture keys as they suppose to trigger present-to-present capturing
        // and it is not what we want
        rdoc_api->SetCaptureKeys(nullptr, 0);

        // Also remove rdoc crash handler
        rdoc_api->UnloadCrashHandler();
    }
}

void StartCapture() {
    if (!rdoc_api) {
        return;
    }

    CaptureState expected = CaptureState::StartRequested;
    if (capture_state.compare_exchange_strong(expected, CaptureState::InProgress,
                                              std::memory_order_acq_rel)) {
        rdoc_api->StartFrameCapture(nullptr, nullptr);
        LOG_INFO(Render, "RenderDoc multi-frame capture started; press Home again to save");
    }
}

void EndCapture() {
    if (!rdoc_api) {
        return;
    }

    CaptureState expected = CaptureState::StopRequested;
    if (capture_state.compare_exchange_strong(expected, CaptureState::Idle,
                                              std::memory_order_acq_rel)) {
        if (rdoc_api->EndFrameCapture(nullptr, nullptr) == 0) {
            LOG_ERROR(Render, "RenderDoc failed to end capture");
        } else {
            LOG_INFO(Render, "RenderDoc multi-frame capture saved");
        }
    }
}

void ToggleCapture() {
    CaptureState state = capture_state.load(std::memory_order_acquire);
    for (;;) {
        switch (state) {
        case CaptureState::Idle:
            if (capture_state.compare_exchange_weak(state, CaptureState::StartRequested,
                                                    std::memory_order_acq_rel)) {
                LOG_INFO(Render, "RenderDoc capture requested");
                return;
            }
            break;
        case CaptureState::StartRequested:
            if (capture_state.compare_exchange_weak(state, CaptureState::Idle,
                                                    std::memory_order_acq_rel)) {
                LOG_INFO(Render, "RenderDoc capture request cancelled");
                return;
            }
            break;
        case CaptureState::InProgress:
            if (capture_state.compare_exchange_weak(state, CaptureState::StopRequested,
                                                    std::memory_order_acq_rel)) {
                LOG_INFO(Render, "RenderDoc capture save requested");
                return;
            }
            break;
        case CaptureState::StopRequested:
            return;
        }
    }
}

void SetOutputDir(const std::filesystem::path& path, const std::string& prefix) {
    if (!rdoc_api) {
        return;
    }
    LOG_WARNING(Common, "RenderDoc capture path: {}", (path / prefix).string());
    rdoc_api->SetCaptureFilePathTemplate(fmt::UTF((path / prefix).u8string()).data.data());
}

bool IsRenderDocLoaded() {
    return rdoc_api != nullptr;
}

void RequestScreenshot(const ScreenshotRequest request) {
    switch (request) {
    case ScreenshotRequest::GameOnly:
        screenshot_game_only_count.fetch_add(1, std::memory_order_relaxed);
        break;
    case ScreenshotRequest::WithOverlays:
        screenshot_with_overlays_count.fetch_add(1, std::memory_order_relaxed);
        break;
    case ScreenshotRequest::None:
    default:
        break;
    }
}

u32 ConsumeGameOnlyScreenshotRequests() {
    return screenshot_game_only_count.exchange(0, std::memory_order_acq_rel);
}

u32 ConsumeWithOverlaysScreenshotRequests() {
    return screenshot_with_overlays_count.exchange(0, std::memory_order_acq_rel);
}

ScreenshotRequests ConsumeScreenshotRequests() {
    return ScreenshotRequests{
        .game_only_count = ConsumeGameOnlyScreenshotRequests(),
        .with_overlays_count = ConsumeWithOverlaysScreenshotRequests(),
    };
}

} // namespace VideoCore
