// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/debug.h"
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/singleton.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/devtools/layer.h"
#include "core/emulator_settings.h"
#include "core/libraries/system/systemservice.h"
#include "core/startup_progress.h"
#include "imgui/friends_layer.h"
#include "imgui/invitation_prompt_layer.h"
#include "imgui/notifications_layer.h"
#include "imgui/renderer/imgui_core.h"
#include "imgui/renderer/imgui_impl_vulkan.h"
#include "imgui/renderer/texture_manager.h"
#include "imgui/shadnet_notifications_layer.h"
#include "imgui/startup_loading.h"
#include "sdl_window.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/image.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csetjmp>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <system_error>
#include <thread>
#include <vector>
#include <imgui.h>
#include <png.h>
#include <vk_mem_alloc.h>

namespace Vulkan {

bool CanBlitToSwapchain(const vk::PhysicalDevice physical_device, vk::Format format) {
    const vk::FormatProperties props{physical_device.getFormatProperties(format)};
    return static_cast<bool>(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst);
}

[[nodiscard]] vk::ImageSubresourceLayers MakeImageSubresourceLayers() {
    return vk::ImageSubresourceLayers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlit(s32 frame_width, s32 frame_height, s32 dst_width,
                                          s32 dst_height, s32 offset_x, s32 offset_y) {
    return vk::ImageBlit{
        .srcSubresource = MakeImageSubresourceLayers(),
        .srcOffsets =
            std::array{
                vk::Offset3D{
                    .x = 0,
                    .y = 0,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = frame_width,
                    .y = frame_height,
                    .z = 1,
                },
            },
        .dstSubresource = MakeImageSubresourceLayers(),
        .dstOffsets =
            std::array{
                vk::Offset3D{
                    .x = offset_x,
                    .y = offset_y,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = offset_x + dst_width,
                    .y = offset_y + dst_height,
                    .z = 1,
                },
            },
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlitStretch(s32 frame_width, s32 frame_height,
                                                 s32 swapchain_width, s32 swapchain_height) {
    return MakeImageBlit(frame_width, frame_height, swapchain_width, swapchain_height, 0, 0);
}

/// SHADPS4_VK_RECORD_THREAD=0 makes the command processor record its Vulkan commands itself
/// again instead of handing them to the recording thread.
static bool DrawRecordingThreadEnabled() {
    const char* env = std::getenv("SHADPS4_VK_RECORD_THREAD");
    if (env != nullptr && env[0] == '0') {
        LOG_INFO(Render_Vulkan, "Vulkan recording thread disabled by SHADPS4_VK_RECORD_THREAD");
        return false;
    }
    return true;
}

static vk::Rect2D FitImage(s32 frame_width, s32 frame_height, s32 swapchain_width,
                           s32 swapchain_height) {
    float frame_aspect = static_cast<float>(frame_width) / frame_height;
    float swapchain_aspect = static_cast<float>(swapchain_width) / swapchain_height;

    u32 dst_width = swapchain_width;
    u32 dst_height = swapchain_height;

    if (frame_aspect > swapchain_aspect) {
        dst_height = static_cast<s32>(swapchain_width / frame_aspect);
    } else {
        dst_width = static_cast<s32>(swapchain_height * frame_aspect);
    }

    const s32 offset_x = (swapchain_width - dst_width) / 2;
    const s32 offset_y = (swapchain_height - dst_height) / 2;

    return vk::Rect2D{{offset_x, offset_y}, {dst_width, dst_height}};
}

[[nodiscard]] vk::ImageBlit MakeImageBlitFit(s32 frame_width, s32 frame_height, s32 swapchain_width,
                                             s32 swapchain_height) {
    const auto& dst_rect = FitImage(frame_width, frame_height, swapchain_width, swapchain_height);

    return MakeImageBlit(frame_width, frame_height, dst_rect.extent.width, dst_rect.extent.height,
                         dst_rect.offset.x, dst_rect.offset.y);
}

enum class ScreenshotKind : u8 {
    GameOnly,
    WithOverlays,
};

struct ScreenshotReadback {
    ScreenshotKind kind{};
    std::vector<std::filesystem::path> paths{};
    VideoCore::Buffer buffer;
    u32 width{};
    u32 height{};
    vk::Format format{};
    bool hdr_encoded{};

    ScreenshotReadback(const Instance& instance, Scheduler& scheduler, ScreenshotKind kind_,
                       std::vector<std::filesystem::path> paths_, const u32 width_,
                       const u32 height_, const vk::Format format_, const bool hdr_encoded_)
        : kind{kind_}, paths{std::move(paths_)},
          buffer{instance,
                 scheduler,
                 VideoCore::MemoryUsage::Download,
                 0,
                 vk::BufferUsageFlagBits::eTransferDst,
                 static_cast<u64>(width_) * static_cast<u64>(height_) * 4},
          width{width_}, height{height_}, format{format_}, hdr_encoded{hdr_encoded_} {}
};

static std::string SanitizeFilenameComponent(std::string value) {
    for (char& c : value) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '_' && c != '-') {
            c = '_';
        }
    }
    if (value.empty()) {
        return "UNKNOWN";
    }
    return value;
}

static std::vector<std::filesystem::path> BuildScreenshotPaths(const ScreenshotKind kind,
                                                               const u32 count) {
    static std::atomic<u64> screenshot_sequence{0};
    std::vector<std::filesystem::path> paths{};
    if (count == 0) {
        return paths;
    }

    const auto& screenshots_dir = Common::FS::GetUserPath(Common::FS::PathType::ScreenshotsDir);
    std::filesystem::create_directories(screenshots_dir);

    const auto game_id =
        SanitizeFilenameComponent(std::string(Common::ElfInfo::Instance().GameSerial()));
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
        1000;

    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif

    std::ostringstream stamp;
    stamp << std::put_time(&local_tm, "%Y%m%d_%H%M%S") << '_' << std::setw(3) << std::setfill('0')
          << ms;

    const char* suffix = kind == ScreenshotKind::GameOnly ? "game" : "hud";
    const auto first_sequence = screenshot_sequence.fetch_add(count, std::memory_order_relaxed);

    paths.reserve(count);
    const auto stamp_str = stamp.str();
    for (u32 i = 0; i < count; ++i) {
        paths.emplace_back(screenshots_dir / fmt::format("{}_{}_{}_{:06}.png", game_id, stamp_str,
                                                         suffix, first_sequence + i));
    }

    return paths;
}

static float PqToNits(const float encoded) {
    // ST.2084 inverse EOTF
    constexpr float m1 = 2610.0f / 16384.0f;
    constexpr float m2 = 2523.0f / 32.0f;
    constexpr float c1 = 3424.0f / 4096.0f;
    constexpr float c2 = 2413.0f / 128.0f;
    constexpr float c3 = 2392.0f / 128.0f;

    const float v = std::clamp(encoded, 0.0f, 1.0f);
    const float vp = std::pow(v, 1.0f / m2);
    const float num = std::max(vp - c1, 0.0f);
    const float den = std::max(c2 - c3 * vp, 1e-6f);
    return 10000.0f * std::pow(num / den, 1.0f / m1);
}

static float ToneMapToSdrLinear(const float nits) {
    // Map absolute HDR luminance into SDR [0,1], preserving 100-nit white.
    constexpr float sdr_white_nits = 100.0f;
    const float x = std::max(nits, 0.0f) / sdr_white_nits;
    const float mapped = (2.0f * x) / (1.0f + x);
    return std::clamp(mapped, 0.0f, 1.0f);
}

static float LinearToSrgb(const float linear) {
    const float x = std::clamp(linear, 0.0f, 1.0f);
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

static const std::array<float, 1024>& GetPqDecodeNitsLut() {
    static const std::array<float, 1024> lut = [] {
        std::array<float, 1024> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = PqToNits(static_cast<float>(i) / 1023.0f);
        }
        return values;
    }();
    return lut;
}

static const std::array<u8, 1024>& GetUnorm10ToU8Lut() {
    static const std::array<u8, 1024> lut = [] {
        std::array<u8, 1024> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<u8>((i * 255u + 511u) / 1023u);
        }
        return values;
    }();
    return lut;
}

static void CopyImageToReadback(const CommandRecorder& cmdbuf, const vk::Image image,
                                const vk::ImageLayout layout, ScreenshotReadback& readback) {
    const vk::BufferImageCopy copy_region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {readback.width, readback.height, 1},
    };
    cmdbuf.copyImageToBuffer(image, layout, readback.buffer.Handle(), copy_region);
}

static bool ConvertReadbackToRgba8(const ScreenshotReadback& readback, std::vector<u8>& out_rgba) {
    const u64 pixel_count = static_cast<u64>(readback.width) * static_cast<u64>(readback.height);
    const u64 byte_size = pixel_count * 4;
    if (readback.buffer.mapped_data.size() < byte_size) {
        LOG_ERROR(Render_Vulkan, "Screenshot readback buffer size mismatch (have {}, need {})",
                  readback.buffer.mapped_data.size(), byte_size);
        return false;
    }

    const auto src =
        std::span<const u8>{readback.buffer.mapped_data.data(), static_cast<size_t>(byte_size)};
    out_rgba.resize(static_cast<size_t>(byte_size));

    switch (readback.format) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        std::memcpy(out_rgba.data(), src.data(), out_rgba.size());
        for (u64 i = 0; i < pixel_count; ++i) {
            out_rgba[static_cast<size_t>(i) * 4 + 3] = 255;
        }
        return true;
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            out_rgba[o + 0] = src[o + 2];
            out_rgba[o + 1] = src[o + 1];
            out_rgba[o + 2] = src[o + 0];
            out_rgba[o + 3] = 255;
        }
        return true;
    case vk::Format::eA2R10G10B10UnormPack32: {
        const auto& pq_decode_lut = GetPqDecodeNitsLut();
        const auto& unorm10_to_u8 = GetUnorm10ToU8Lut();

        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            const u32 packed = static_cast<u32>(src[o + 0]) | (static_cast<u32>(src[o + 1]) << 8) |
                               (static_cast<u32>(src[o + 2]) << 16) |
                               (static_cast<u32>(src[o + 3]) << 24);
            const u32 b = (packed >> 0) & 0x3FF;
            const u32 g = (packed >> 10) & 0x3FF;
            const u32 r = (packed >> 20) & 0x3FF;

            if (readback.hdr_encoded) {
                // Rec.2020 + PQ. Convert to SDR Rec.709 for PNG output.
                const float r2020 = pq_decode_lut[r];
                const float g2020 = pq_decode_lut[g];
                const float b2020 = pq_decode_lut[b];

                const float r709_nits = 1.6605f * r2020 - 0.5876f * g2020 - 0.0728f * b2020;
                const float g709_nits = -0.1246f * r2020 + 1.1329f * g2020 - 0.0083f * b2020;
                const float b709_nits = -0.0182f * r2020 - 0.1006f * g2020 + 1.1187f * b2020;

                const float r_srgb = LinearToSrgb(ToneMapToSdrLinear(r709_nits));
                const float g_srgb = LinearToSrgb(ToneMapToSdrLinear(g709_nits));
                const float b_srgb = LinearToSrgb(ToneMapToSdrLinear(b709_nits));

                out_rgba[o + 0] = static_cast<u8>(std::clamp(r_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 1] = static_cast<u8>(std::clamp(g_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 2] = static_cast<u8>(std::clamp(b_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
            } else {
                out_rgba[o + 0] = unorm10_to_u8[r];
                out_rgba[o + 1] = unorm10_to_u8[g];
                out_rgba[o + 2] = unorm10_to_u8[b];
            }
            out_rgba[o + 3] = 255;
        }
        return true;
    }
    case vk::Format::eA2B10G10R10UnormPack32: {
        const auto& pq_decode_lut = GetPqDecodeNitsLut();
        const auto& unorm10_to_u8 = GetUnorm10ToU8Lut();

        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            const u32 packed = static_cast<u32>(src[o + 0]) | (static_cast<u32>(src[o + 1]) << 8) |
                               (static_cast<u32>(src[o + 2]) << 16) |
                               (static_cast<u32>(src[o + 3]) << 24);
            const u32 r = (packed >> 0) & 0x3FF;
            const u32 g = (packed >> 10) & 0x3FF;
            const u32 b = (packed >> 20) & 0x3FF;

            if (readback.hdr_encoded) {
                // HDR swapchain path is Rec.2020 + PQ. Convert to SDR Rec.709 for PNG output.
                const float r2020 = pq_decode_lut[r];
                const float g2020 = pq_decode_lut[g];
                const float b2020 = pq_decode_lut[b];

                const float r709_nits = 1.6605f * r2020 - 0.5876f * g2020 - 0.0728f * b2020;
                const float g709_nits = -0.1246f * r2020 + 1.1329f * g2020 - 0.0083f * b2020;
                const float b709_nits = -0.0182f * r2020 - 0.1006f * g2020 + 1.1187f * b2020;

                const float r_srgb = LinearToSrgb(ToneMapToSdrLinear(r709_nits));
                const float g_srgb = LinearToSrgb(ToneMapToSdrLinear(g709_nits));
                const float b_srgb = LinearToSrgb(ToneMapToSdrLinear(b709_nits));

                out_rgba[o + 0] = static_cast<u8>(std::clamp(r_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 1] = static_cast<u8>(std::clamp(g_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 2] = static_cast<u8>(std::clamp(b_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
            } else {
                out_rgba[o + 0] = unorm10_to_u8[r];
                out_rgba[o + 1] = unorm10_to_u8[g];
                out_rgba[o + 2] = unorm10_to_u8[b];
            }
            out_rgba[o + 3] = 255;
        }
        return true;
    }
    default:
        LOG_WARNING(Render_Vulkan, "Unsupported screenshot format: {}",
                    vk::to_string(readback.format));
        return false;
    }
}

static bool WritePng(const std::filesystem::path& path, const std::span<const u8> rgba,
                     const u32 width, const u32 height) {
    Common::FS::IOFile file(path, Common::FS::FileAccessMode::Create);
    if (!file.IsOpen()) {
        return false;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png_ptr == nullptr) {
        return false;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == nullptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr)) != 0) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }

    png_init_io(png_ptr, file.file);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    thread_local std::vector<png_bytep> rows;
    rows.resize(height);
    for (u32 y = 0; y < height; ++y) {
        rows[y] = const_cast<png_bytep>(rgba.data() + static_cast<size_t>(y) * width * 4);
    }

    png_write_image(png_ptr, rows.data());
    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;
}

static void SavePendingScreenshots(const std::vector<ScreenshotReadback>& readbacks) {
    for (const auto& readback : readbacks) {
        if (readback.paths.empty()) {
            continue;
        }

        std::vector<u8> rgba;
        if (!ConvertReadbackToRgba8(readback, rgba)) {
            continue;
        }

        const auto& primary_path = readback.paths.front();
        if (!WritePng(primary_path, rgba, readback.width, readback.height)) {
            LOG_ERROR(Render_Vulkan, "Failed saving screenshot to {}", primary_path.string());
            continue;
        }

        LOG_INFO(Render_Vulkan, "Saved screenshot: {}", primary_path.string());

        std::ifstream file(primary_path, std::ios::binary);
        std::vector<u8> imgdata;
        if (file) {
            imgdata = std::vector<u8>(std::istreambuf_iterator<char>(file),
                                      std::istreambuf_iterator<char>());
        }
        shadNotifications::QueueNotification("Saved screenshot:\n" + primary_path.string(), 3.0f,
                                             shadNotifications::position::BottomRight, imgdata);

        for (size_t i = 1; i < readback.paths.size(); ++i) {
            const auto& path = readback.paths[i];
            std::error_code ec{};
            std::filesystem::copy_file(primary_path, path, std::filesystem::copy_options::none, ec);
            if (ec) {
                // Fallback for platforms/filesystems where copy_file can fail for transient
                // reasons.
                if (!WritePng(path, rgba, readback.width, readback.height)) {
                    LOG_ERROR(Render_Vulkan, "Failed saving screenshot to {}", path.string());
                    continue;
                }
            }

            LOG_INFO(Render_Vulkan, "Saved screenshot: {}", path.string());
            std::ifstream file(path, std::ios::binary);
            std::vector<u8> imgdata;
            if (file) {
                imgdata = std::vector<u8>(std::istreambuf_iterator<char>(file),
                                          std::istreambuf_iterator<char>());
            }
            shadNotifications::QueueNotification("Saved screenshot:\n" + path.string(), 3.0f,
                                                 shadNotifications::position::BottomRight, imgdata);
        }
    }
}

Presenter::Presenter(Frontend::WindowSDL& window_, AmdGpu::Liverpool* liverpool_)
    : window{window_}, liverpool{liverpool_},
      instance{window, EmulatorSettings.GetGpuId(), EmulatorSettings.IsVkValidationEnabled(),
               EmulatorSettings.IsVkCrashDiagnosticEnabled()},
      draw_scheduler{instance, true, DrawRecordingThreadEnabled()},
      present_scheduler{instance, false, false, true},
      swapchain{instance, window},
      rasterizer{std::make_unique<Rasterizer>(instance, draw_scheduler, liverpool)},
      texture_cache{rasterizer->GetTextureCache()},
      display_pacer{1'000'000'000 / static_cast<s64>(EmulatorSettings.GetVblankFrequency())},
      vrr_pacer{1'000'000'000 / static_cast<s64>(EmulatorSettings.GetVblankFrequency())} {
    gpu_frames_ahead = std::min(EmulatorSettings.GetGpuFramesAhead(), MaxGpuFramesAhead);
    const u32 num_images = swapchain.GetImageCount();
    // Four source frames cover all independent ownership states during a host stall: last shown,
    // active present, mailbox and next producer. Keep the old +1 policy for larger swapchains.
    const u32 num_frames = std::max(num_images + 1, 4U);
    const vk::Device device = instance.GetDevice();

    // Create presentation frames.
    present_frames.resize(num_frames);
    for (u32 i = 0; i < num_frames; i++) {
        Frame& frame = present_frames[i];
        frame.id = i;
        auto fence = Check<"create present done fence">(
            device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        frame.present_done = fence;
        free_queue.push(&frame);
    }
    recycle_thread = std::jthread([this](std::stop_token token) { RecycleThread(token); });
    if (swapchain.HasPresentWait()) {
        pacing_log = std::getenv("SHADPS4_PACING_LOG") != nullptr;
        present_ready_thread =
            std::jthread([this](std::stop_token token) { PresentReadyThread(token); });
        present_wait_thread =
            std::jthread([this](std::stop_token token) { PresentWaitThread(token); });
    }
    if (swapchain.HasLowLatency()) {
        const vk::StructureChain semaphore_chain = {
            vk::SemaphoreCreateInfo{},
            vk::SemaphoreTypeCreateInfo{
                .semaphoreType = vk::SemaphoreType::eTimeline,
                .initialValue = 0,
            },
        };
        reflex_semaphore = Check<"create Reflex sleep semaphore">(
            device.createSemaphoreUnique(semaphore_chain.get()));
        const u64 present_id = PresentIdOfFrame(gcp_frame_id);
        swapchain.SetLatencyMarker(present_id, vk::LatencyMarkerNV::eSimulationStart);
        draw_scheduler.SetLatencyPresentId(present_id);
        render_submit_pending = true;
        liverpool->SetGfxSubmitBeginHook([this] { BeginGuestRenderSubmit(); });
    }

    SetPostFxOptions(EmulatorSettings.GetUpscaler(), EmulatorSettings.GetAntiAliasing(),
                     EmulatorSettings.GetSharpening(), EmulatorSettings.GetRcasAttenuation());
    postfx_pass.Create(instance);
    pp_pass.Create(device, swapchain.GetSurfaceFormat().format);

    ImGui::Layer::AddLayer(Common::Singleton<Core::Devtools::Layer>::Instance());
    ImGui::Friends::Register();
    ImGui::ShadNetNotify::Register();
    ImGui::InvitationPrompt::Register();
}

Presenter::~Presenter() {
    ImGui::InvitationPrompt::Unregister();
    ImGui::ShadNetNotify::Unregister();
    ImGui::Friends::Unregister();
    ImGui::Layer::RemoveLayer(Common::Singleton<Core::Devtools::Layer>::Instance());

    recycle_thread.request_stop();
    recycle_cv.notify_all();
    recycle_thread.join();
    for (auto* thread : {&present_ready_thread, &present_wait_thread}) {
        if (thread->joinable()) {
            thread->request_stop();
            thread->join();
        }
    }

    draw_scheduler.Finish();
    present_scheduler.Finish();
    draw_scheduler.ResetCommandBuffer();
    present_scheduler.ResetCommandBuffer();

    const vk::Device device = instance.GetDevice();
    for (auto& frame : present_frames) {
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        device.destroyImageView(frame.image_view);
        device.destroyFence(frame.present_done);
    }
}

void Presenter::SyncPipelineCache() {
    rasterizer->GetPipelineCache().Sync();
}

void Presenter::PreloadPipelineCache() {
    rasterizer->GetPipelineCache().Preload();
}

void Presenter::ReturnFrame(Frame* frame) {
    if (frame == nullptr) {
        return;
    }
    std::scoped_lock lock{free_mutex};
    free_queue.push(frame);
    free_cv.notify_one();
}

void Presenter::RecycleFrameAsync(Frame* frame) {
    if (frame == nullptr) {
        return;
    }
    {
        std::scoped_lock lock{recycle_mutex};
        recycle_queue.push(frame);
    }
    recycle_cv.notify_one();
}

void Presenter::RecycleThread(std::stop_token token) {
    Common::SetCurrentThreadName("shadPS4:FrameRecycle");
    while (true) {
        Frame* frame;
        {
            std::unique_lock lock{recycle_mutex};
            recycle_cv.wait(lock, token, [this] { return !recycle_queue.empty(); });
            if (recycle_queue.empty()) {
                if (token.stop_requested()) {
                    return;
                }
                continue;
            }
            frame = recycle_queue.front();
            recycle_queue.pop();
        }

        if (frame->ready_semaphore && frame->ready_tick != 0) {
            const vk::SemaphoreWaitInfo wait_info = {
                .semaphoreCount = 1,
                .pSemaphores = &frame->ready_semaphore,
                .pValues = &frame->ready_tick,
            };
            const auto result =
                instance.GetDevice().waitSemaphores(wait_info, std::numeric_limits<u64>::max());
            ASSERT_MSG(result == vk::Result::eSuccess,
                       "Failed waiting for a superseded presentation frame: {}",
                       vk::to_string(result));
        }
        // A presented frame is still read by its presentation until the fence signals. The fence
        // of a frame that was never presented stays signaled from its previous use.
        const auto result = instance.GetDevice().waitForFences(frame->present_done, true,
                                                               std::numeric_limits<u64>::max());
        ASSERT_MSG(result == vk::Result::eSuccess, "Failed retiring presentation frame: {}",
                   vk::to_string(result));
        ReturnFrame(frame);
    }
}

Presenter::PresentTimingFeedback Presenter::GetPresentTimingFeedback() const {
    const u64 seq = correction_seq.load(std::memory_order_acquire);
    return {
        .last_present_call_ns = last_present_call_ns.load(std::memory_order_acquire),
        .present_call_period_ns = present_call_period_ns.load(std::memory_order_acquire),
        .generation = timing_generation.load(std::memory_order_acquire),
        .present_call_samples = present_call_samples.load(std::memory_order_acquire),
        .is_fifo = swapchain.IsFIFO(),
        .display_locked = display_locked.load(std::memory_order_acquire),
        .display_timed = UsesDisplayPacing(),
        .correction_seq = seq,
        .correction_ns = correction_ns.load(std::memory_order_acquire),
    };
}

void Presenter::ResetFifoTimingFeedback() {
    std::scoped_lock lock{feedback_mutex};
    ResetFifoTimingFeedbackLocked();
}

void Presenter::ClearPresentCallHistoryLocked() {
    last_present_call_ns.store(0, std::memory_order_release);
    present_call_period_ns.store(0, std::memory_order_release);
    present_call_samples.store(0, std::memory_order_release);
    present_call_period_history.fill(0);
    present_call_period_history_index = 0;
    present_call_period_history_size = 0;
}

void Presenter::ResetFifoTimingFeedbackLocked() {
    ClearPresentCallHistoryLocked();
    // The present wait thread resets the display pacer when it sees the new generation.
    display_locked.store(false, std::memory_order_release);
    {
        std::scoped_lock lock{pacing_mutex};
        pacing.locked = false;
        pacing.anchor_display_ns = 0;
    }
    timing_generation.fetch_add(1, std::memory_order_acq_rel);
}

void Presenter::SetPresentationEpoch(const u64 epoch) {
    std::scoped_lock lock{feedback_mutex};
    if (presentation_epoch == epoch) {
        return;
    }
    presentation_epoch = epoch;
    ResetFifoTimingFeedbackLocked();
}

void Presenter::RecordPresentCall(const u64 epoch) {
    if (epoch == 0) {
        return;
    }

    std::scoped_lock lock{feedback_mutex};
    if (epoch != presentation_epoch) {
        return;
    }

    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const s64 now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    const s64 previous_ns = last_present_call_ns.exchange(now_ns, std::memory_order_acq_rel);
    if (previous_ns == 0) {
        return;
    }

    const s64 sample = now_ns - previous_ns;
    if (sample <= 1'000'000 || sample >= 100'000'000) {
        // Discard intervals that cannot represent steady FIFO pacing and require a fresh sample
        // window before applying another guest-timer correction.
        ClearPresentCallHistoryLocked();
        return;
    }

    present_call_period_history[present_call_period_history_index] = sample;
    present_call_period_history_index =
        (present_call_period_history_index + 1) % PresentCallPeriodWindow;
    present_call_period_history_size =
        std::min(present_call_period_history_size + 1, PresentCallPeriodWindow);

    auto sorted = present_call_period_history;
    std::sort(sorted.begin(), sorted.begin() + present_call_period_history_size);
    const u32 middle = present_call_period_history_size / 2;
    const s64 median = present_call_period_history_size % 2 != 0
                           ? sorted[middle]
                           : (sorted[middle - 1] + sorted[middle]) / 2;
    present_call_period_ns.store(median, std::memory_order_release);
    present_call_samples.store(present_call_period_history_size, std::memory_order_release);
}

void Presenter::RecreateSwapchain() {
    swapchain.Recreate(window.GetWidth(), window.GetHeight());
    ResetFifoTimingFeedback();
}

bool Presenter::UsesDisplayPacing() const {
    return swapchain.HasPresentWait() && (swapchain.IsFIFO() || swapchain.IsMailbox());
}

u64 Presenter::NextPresentId(const Frame* frame, const bool is_reusing_frame) {
    if (!swapchain.HasPresentWait()) {
        return 0;
    }
    constexpr u64 RepresentMask = (u64{1} << PresentIdFrameShift) - 1;
    u64 present_id = 0;
    if (!is_reusing_frame && frame->frame_id != 0) {
        present_id = PresentIdOfFrame(frame->frame_id);
    } else if ((last_present_id & RepresentMask) != RepresentMask) {
        present_id = last_present_id + 1;
    }
    if (present_id <= last_present_id) {
        return 0;
    }
    last_present_id = present_id;
    return present_id;
}

bool Presenter::ShouldHoldFlip(const s64 tick_ns, const s64 next_tick_ns,
                               const u64 previous_frame_id) {
    if (previous_frame_id == 0) {
        return false;
    }
    std::scoped_lock lock{pacing_mutex};
    const s64 period = pacing.display_period_ns;
    if (!pacing.locked || period <= 0 || pacing.anchor_display_ns == 0) {
        return false;
    }
    // Pacing data from a display that stopped taking frames is stale.
    constexpr s64 StaleNs = 50'000'000;
    if (tick_ns - pacing.anchor_display_ns > 4 * period + StaleNs) {
        return false;
    }

    const u64 previous_id = PresentIdOfFrame(previous_frame_id);
    s64 previous_slot;
    if (pacing.displayed_present_id >= previous_id) {
        previous_slot = pacing.anchor_display_ns;
    } else if (pacing.ready_present_id == previous_id && pacing.ready_ns != 0) {
        // Finished and waiting for the display: it goes out at the first scanout after that.
        const s64 since_anchor = pacing.ready_ns - pacing.anchor_display_ns;
        const s64 refreshes = since_anchor <= 0 ? 0 : (since_anchor + period - 1) / period;
        previous_slot = pacing.anchor_display_ns + refreshes * period;
        // That scanout passed with an older frame, so the display queue holds a backlog. A
        // flip latched now would queue behind it and keep every later frame a refresh late.
        if (tick_ns > previous_slot + std::max<s64>(1'000'000, period / 4)) {
            backlog_holds.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    } else {
        // Still rendering, or never presented: the GPU paces the guest, not the display.
        return false;
    }

    // The display shows the next frame at the scanout after the previous one.
    const s64 next_slot = previous_slot + period;
    if (tick_ns + pacing.lead_ns > next_slot) {
        return false;
    }
    // Latching at the next vblank still makes that scanout, with a fresher frame.
    if (next_tick_ns + pacing.lead_ns <= next_slot) {
        slot_holds.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

void Presenter::PresentReadyThread(std::stop_token token) {
    Common::SetCurrentThreadName("shadPS4:PresentReady");
    // The wake-up time is the completion timestamp, so this thread must run as soon as it wakes.
    Common::SetCurrentThreadPriority(Common::ThreadPriority::VeryHigh);

    using namespace std::chrono;
    constexpr u64 WaitSliceNs = 10'000'000;
    constexpr u32 MaxWaitSlices = 50;

    while (!token.stop_requested()) {
        PendingPresentWait entry;
        {
            std::unique_lock lock{present_wait_mutex};
            ready_wait_cv.wait(lock, token, [this] { return !ready_wait_queue.empty(); });
            if (ready_wait_queue.empty()) {
                continue;
            }
            entry = ready_wait_queue.front();
            ready_wait_queue.pop_front();
        }

        const vk::Semaphore present_semaphore = present_scheduler.GetMasterSemaphore()->Handle();
        const vk::SemaphoreWaitInfo ready_wait = {
            .semaphoreCount = 1,
            .pSemaphores = &present_semaphore,
            .pValues = &entry.present_tick,
        };
        vk::Result ready_result = vk::Result::eTimeout;
        for (u32 i = 0; i < MaxWaitSlices && ready_result == vk::Result::eTimeout &&
                        !token.stop_requested();
             ++i) {
            ready_result = instance.GetDevice().waitSemaphores(ready_wait, WaitSliceNs);
        }
        entry.ready_ns = 0;
        if (ready_result == vk::Result::eSuccess) {
            entry.ready_ns =
                duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
            std::scoped_lock lock{pacing_mutex};
            pacing.ready_present_id = entry.present_id;
            pacing.ready_ns = entry.ready_ns;
        }

        {
            std::scoped_lock lock{present_wait_mutex};
            if (display_wait_queue.size() >= MaxPendingPresentWaits) {
                display_wait_queue.pop_front();
            }
            display_wait_queue.push_back(entry);
        }
        display_wait_cv.notify_one();
    }
}

void Presenter::PresentWaitThread(std::stop_token token) {
    Common::SetCurrentThreadName("shadPS4:PresentWait");
    // The wake-up time is the display timestamp, so this thread must run as soon as it wakes.
    Common::SetCurrentThreadPriority(Common::ThreadPriority::VeryHigh);

    using namespace std::chrono;
    const auto now_ns = [] {
        return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    };
    constexpr s64 MaxDisplayWaitNs = 500'000'000;
    constexpr s64 MaxSleepSliceNs = 50'000'000;
    constexpr s64 PollLeadNs = 1'500'000;
    constexpr s64 PollIntervalNs = 100'000;
    // A longer gap between the last miss and the hit is too coarse to date the display event.
    constexpr s64 MaxPollGapNs = 500'000;
    constexpr s64 LogIntervalNs = 5'000'000'000;
    s64 last_display_ns = 0;
    u64 pacer_generation = timing_generation.load(std::memory_order_acquire);
    s64 next_log_ns = now_ns() + LogIntervalNs;

    while (!token.stop_requested()) {
        PendingPresentWait entry;
        {
            std::unique_lock lock{present_wait_mutex};
            display_wait_cv.wait(lock, token, [this] { return !display_wait_queue.empty(); });
            if (display_wait_queue.empty()) {
                continue;
            }
            entry = display_wait_queue.front();
            display_wait_queue.pop_front();
        }

        // Polled instead of waited on: a blocking wait would hold the swapchain away from acquire,
        // present and the latency markers. Polling starts shortly before the frame can reach the
        // display: once it is ready and presented, one display period after the previous frame.
        vk::Result display_result = vk::Result::eTimeout;
        s64 not_displayed_ns = std::max(entry.ready_ns, entry.present_ns);
        s64 display_ns = 0;
        if (entry.ready_ns != 0) {
            const s64 period_ns =
                std::max(static_cast<s64>(window.GetDisplayRefreshPeriodNs()), 2 * PollLeadNs);
            const s64 give_up_ns = now_ns() + MaxDisplayWaitNs;
            s64 expected_ns = std::max(last_display_ns + period_ns, not_displayed_ns);
            while (!token.stop_requested()) {
                const s64 now = now_ns();
                if (now >= give_up_ns) {
                    break;
                }
                if (now < expected_ns - PollLeadNs) {
                    Common::AccurateSleep(nanoseconds{std::min(expected_ns - PollLeadNs - now,
                                                               MaxSleepSliceNs)},
                                          nullptr, false);
                    continue;
                }
                if (now > expected_ns + PollLeadNs) {
                    expected_ns += period_ns; // Missed this refresh, try the next one.
                    continue;
                }
                display_result = swapchain.PollPresent(entry.swapchain_serial, entry.present_id);
                display_ns = now_ns();
                if (display_result != vk::Result::eTimeout) {
                    break;
                }
                not_displayed_ns = display_ns;
                Common::AccurateSleep(nanoseconds{PollIntervalNs}, nullptr, false);
            }
        }
        const bool displayed = display_result == vk::Result::eSuccess ||
                               display_result == vk::Result::eSuboptimalKHR;
        // Only a short gap since the last poll that missed the frame dates the display event.
        const bool timed = displayed && display_ns - not_displayed_ns <= MaxPollGapNs;
        if (timed) {
            display_ns = not_displayed_ns + (display_ns - not_displayed_ns) / 2;
        }
        if (displayed) {
            // The lower bound keeps the next prediction early, so its polls cannot start late.
            last_display_ns = timed ? display_ns : not_displayed_ns;
        } else {
            display_ns = now_ns();
        }

        const u64 generation = timing_generation.load(std::memory_order_acquire);
        std::optional<DisplayPacer::Result> result;
        {
            std::scoped_lock lock{pacing_mutex};
            if (generation != pacer_generation) {
                pacer_generation = generation;
                display_pacer.Reset();
                pacing.locked = false;
                pacing.anchor_display_ns = 0;
            }
            // Abandoned waits release the frame for the vblank thread too.
            pacing.displayed_present_id = std::max(pacing.displayed_present_id, entry.present_id);
            if (timed) {
                display_pacer.SetNominalDisplayPeriod(window.GetDisplayRefreshPeriodNs());
                // A frame is available to the display once it is both finished and presented;
                // VRR pacing presents finished frames late on purpose.
                result = display_pacer.AddSample(
                    entry.latch_ns, std::max(entry.ready_ns, entry.present_ns), display_ns);
                pacing.locked = result->locked;
                pacing.display_period_ns = display_pacer.DisplayPeriod();
                pacing.lead_ns = display_pacer.Lead();
                pacing.anchor_display_ns = display_ns;
            }
        }
        if (result) {
            display_locked.store(result->locked, std::memory_order_release);
            correction_ns.store(result->correction_ns, std::memory_order_release);
            correction_seq.fetch_add(1, std::memory_order_acq_rel);
        }

        if (pacing_log && display_ns >= next_log_ns) {
            next_log_ns = display_ns + LogIntervalNs;
            DisplayPacer::Stats stats;
            PacingState state;
            {
                std::scoped_lock lock{pacing_mutex};
                stats = display_pacer.TakeStats();
                state = pacing;
            }
            LOG_INFO(Render_Vulkan,
                     "Display pacing: locked={} period={:.3f}ms lead={:.2f}ms p95={:.2f}ms "
                     "samples={} coherent={}/{} held={}/{} holds slot={} backlog={} "
                     "corrections={:.2f}ms",
                     state.locked, static_cast<double>(state.display_period_ns) / 1e6,
                     static_cast<double>(state.lead_ns) / 1e6,
                     static_cast<double>(stats.production_p95_ns) / 1e6, stats.samples,
                     stats.coherent, stats.history, stats.held, stats.history,
                     slot_holds.exchange(0, std::memory_order_relaxed),
                     backlog_holds.exchange(0, std::memory_order_relaxed),
                     static_cast<double>(stats.corrections_ns) / 1e6);
        }
    }
}

void Presenter::EndGuestFrame() {
    const bool reflex = reflex_semaphore && swapchain.HasLowLatency();
    if (reflex) {
        swapchain.SetLatencyMarker(PresentIdOfFrame(gcp_frame_id),
                                   vk::LatencyMarkerNV::eRendersubmitEnd);
    }
    ++gcp_frame_id;
    // The next frame reaches the command processor with the next guest submission.
    render_submit_pending = true;
    draw_scheduler.SetLatencyPresentId(reflex ? PresentIdOfFrame(gcp_frame_id) : 0);
}

void Presenter::BeginGuestRenderSubmit() {
    if (!render_submit_pending) {
        return;
    }
    render_submit_pending = false;
    swapchain.SetLatencyMarker(PresentIdOfFrame(gcp_frame_id),
                               vk::LatencyMarkerNV::eRendersubmitStart);
}

void Presenter::WaitForReflex() {
    std::scoped_lock lock{reflex_mutex};
    // Each flip of the guest ends one of its frames, and the command processor ends them in the
    // same order, so both sides give a frame the same present id.
    const u64 present_id = PresentIdOfFrame(guest_frame_id++);
    if (!reflex_semaphore || !swapchain.HasLowLatency()) {
        return;
    }
    swapchain.SetLatencyMarker(present_id, vk::LatencyMarkerNV::eSimulationEnd);
    const u64 value = ++reflex_sleep_value;
    if (swapchain.LatencySleep(*reflex_semaphore, value)) {
        const vk::Semaphore semaphore = *reflex_semaphore;
        const vk::SemaphoreWaitInfo wait_info = {
            .semaphoreCount = 1,
            .pSemaphores = &semaphore,
            .pValues = &value,
        };
        constexpr u64 MaxSleepNs = 100'000'000;
        const auto result = instance.GetDevice().waitSemaphores(wait_info, MaxSleepNs);
        if (result != vk::Result::eSuccess && !reflex_timeout_logged) {
            reflex_timeout_logged = true;
            LOG_WARNING(Render_Vulkan, "NVIDIA Reflex sleep did not finish: {}",
                        vk::to_string(result));
        }
    }
    // The guest simulates its next frame once this returns.
    swapchain.SetLatencyMarker(PresentIdOfFrame(guest_frame_id),
                               vk::LatencyMarkerNV::eSimulationStart);
}

void Presenter::PaceVrrPresent(const u64 present_tick, const s64 latch_ns) {
    // A display with a cadence of its own is paced through the guest vblank instead.
    if (!EmulatorSettings.IsVrrPacingEnabled() || display_locked.load(std::memory_order_acquire)) {
        vrr_pacer.Reset();
        return;
    }

    using namespace std::chrono;
    const auto now_ns = [] {
        return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    };
    const VrrPacer::Schedule schedule = vrr_pacer.Plan(latch_ns);
    const vk::Semaphore present_semaphore = present_scheduler.GetMasterSemaphore()->Handle();
    const vk::SemaphoreWaitInfo ready_wait = {
        .semaphoreCount = 1,
        .pSemaphores = &present_semaphore,
        .pValues = &present_tick,
    };
    const s64 timeout = std::max<s64>(schedule.wait_limit_ns - now_ns(), 0);
    const bool ready = instance.GetDevice().waitSemaphores(ready_wait, static_cast<u64>(timeout)) ==
                       vk::Result::eSuccess;
    const s64 ready_ns = now_ns();
    vrr_pacer.AddSample(ready ? ready_ns : 0);
    if (!ready) {
        return;
    }

    // The waitable timer wakes up to about half a millisecond late, so the end of the wait spins.
    constexpr s64 SpinNs = 700'000;
    if (schedule.present_ns - ready_ns > SpinNs) {
        Common::AccurateSleep(nanoseconds{schedule.present_ns - ready_ns - SpinNs}, nullptr, false);
    }
    while (now_ns() < schedule.present_ns) {
        std::this_thread::yield();
    }
}

bool Presenter::IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const {
    return std::ranges::find(vo_buffers_addr, color_buffer.Address()) != vo_buffers_addr.cend();
}

void Presenter::RecreateFrame(Frame* frame, u32 width, u32 height) {
    const vk::Device device = instance.GetDevice();
    if (frame->imgui_texture) {
        ImGui::Vulkan::RemoveTexture(frame->imgui_texture);
    }
    if (frame->image_view) {
        device.destroyImageView(frame->image_view);
    }
    if (frame->image) {
        vmaDestroyImage(instance.GetAllocator(), frame->image, frame->allocation);
    }

    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const vk::ImageCreateInfo image_info = {
        .flags = vk::ImageCreateFlagBits::eMutableFormat,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &frame->allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}",
                     vk::to_string(vk::Result{result}));
        UNREACHABLE();
    }
    frame->image = vk::Image{unsafe_image};
    SetObjectName(device, frame->image, "Frame image #{}", frame->id);

    const vk::ImageViewCreateInfo view_info = {
        .image = frame->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    auto view = Check<"create frame image view">(device.createImageView(view_info));
    frame->image_view = view;
    frame->width = width;
    frame->height = height;

    frame->imgui_texture = ImGui::Vulkan::AddTexture(view, vk::ImageLayout::eShaderReadOnlyOptimal);
    frame->is_hdr = swapchain.GetHDR();
}

Frame* Presenter::PrepareLastFrame() {
    if (last_submit_frame == nullptr) {
        return nullptr;
    }

    Frame* frame = last_submit_frame;

    while (true) {
        vk::Result result = instance.GetDevice().waitForFences(frame->present_done, false,
                                                               std::numeric_limits<u64>::max());
        if (result == vk::Result::eSuccess) {
            break;
        }
        if (result == vk::Result::eTimeout) {
            continue;
        }
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
    }

    // The prior presentation leaves the source image shader-readable. Re-presentation only reads
    // it again, so no layout round-trip or extra queue submission is necessary.
    return frame;
}

static vk::Format GetFrameViewFormat(const Libraries::VideoOut::PixelFormat format) {
    switch (format) {
    case Libraries::VideoOut::PixelFormat::A8B8G8R8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A8R8G8B8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A2R10G10B10:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2R10G10B10UnormPack32;
    default:
        break;
    }
    UNREACHABLE_MSG("Unknown format={}", static_cast<u32>(format));
    return {};
}

Frame* Presenter::PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                               VAddr cpu_address) {
    auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
    const auto image_id = texture_cache.FindImage(desc);
    texture_cache.UpdateImage(image_id);

    Frame* frame = GetRenderFrame();

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange{frame_subresources},
    };

    draw_scheduler.EndRendering();
    const auto cmdbuf = draw_scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    VideoCore::ImageViewInfo view_info{};
    view_info.format = GetFrameViewFormat(attribute.attrib.pixel_format);
    // Exclude alpha from output frame to avoid blending with UI.
    view_info.mapping.a = vk::ComponentSwizzle::eOne;

    auto& image = texture_cache.GetImage(image_id);
    auto image_view = *image.FindView(view_info).image_view;
    const vk::Extent2D image_size = {image.info.size.width, image.info.size.height};
    expected_ratio = static_cast<float>(image_size.width) / static_cast<float>(image_size.height);

    const u32 capture_game_only_count = VideoCore::ConsumeGameOnlyScreenshotRequests();
    std::vector<ScreenshotReadback> pending_screenshots;
    if (capture_game_only_count > 0) {
        pending_screenshots.reserve(1);
        const bool hdr_encoded =
            attribute.attrib.pixel_format == Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq;
        pending_screenshots.emplace_back(
            instance, draw_scheduler, ScreenshotKind::GameOnly,
            BuildScreenshotPaths(ScreenshotKind::GameOnly, capture_game_only_count),
            image_size.width, image_size.height, view_info.format, hdr_encoded);
        auto& readback = pending_screenshots.back();

        // Capture the guest output before any host-side scaling (FSR/PP) is applied.
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
        CopyImageToReadback(cmdbuf, image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                            readback);
    }

    // Continue with host-side passes that draw the displayed (scaled) frame.
    image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {});

    const bool input_linear = view_info.format == vk::Format::eR8G8B8A8Srgb ||
                              view_info.format == vk::Format::eB8G8R8A8Srgb;
    const auto filtered = postfx_pass.Render(draw_scheduler, image_view, image_size,
                                            {frame->width, frame->height}, GetPostFxOptions(), input_linear);
    {
        const GpuTimingContext timing{draw_scheduler,
                                      {.kind = GpuWork::PostProcess,
                                       .resource0 = GpuHandle(frame->image),
                                       .resource1 = GpuHandle(filtered.view)}};
        pp_pass.Render(cmdbuf, filtered.view, filtered.size, *frame, pp_settings);
    }

    DebugState.game_resolution = {image_size.width, image_size.height};
    DebugState.output_resolution = {frame->width, frame->height};

    std::shared_ptr<std::vector<ScreenshotReadback>> deferred_screenshots{};
    if (!pending_screenshots.empty()) {
        deferred_screenshots =
            std::make_shared<std::vector<ScreenshotReadback>>(std::move(pending_screenshots));
        draw_scheduler.DeferPriorityOperation(
            [deferred_screenshots]() { SavePendingScreenshots(*deferred_screenshots); });
    }

    // Flush frame creation commands.
    frame->ready_semaphore = draw_scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = draw_scheduler.CurrentTick();
    frame->frame_id = gcp_frame_id;
    frame->latch_ns = 0;
    SubmitInfo info{};
    draw_scheduler.EndGpuTimingFrame();
    draw_scheduler.Flush(info);

    // When the GPU is the slower side, the command processor would otherwise run ahead until
    // every presentation frame is taken. Finished frames then queue behind the GPU: the present
    // call blocks in the driver, pending frames are superseded after the GPU rendered them and
    // the command processor stalls waiting for a frame. Bound the backlog here instead.
    const u64 frame_number = guest_frame_count++;
    guest_frame_ticks[frame_number % guest_frame_ticks.size()] = frame->ready_tick;
    if (gpu_frames_ahead != 0 && frame_number >= gpu_frames_ahead) {
        const u64 oldest_tick =
            guest_frame_ticks[(frame_number - gpu_frames_ahead) % guest_frame_ticks.size()];
        draw_scheduler.Wait(oldest_tick);
    }
    EndGuestFrame();
    return frame;
}

void Presenter::SetPostFxOptions(int upscaler, int aa, int sharpening, int attenuation) {
    const u32 options = static_cast<u32>(upscaler == 1) |
                        (static_cast<u32>(aa == 1 || aa == 3 || aa == 4 ? aa : 0) << 2) |
                        (static_cast<u32>(std::clamp(sharpening, 0, 1)) << 5) |
                        (static_cast<u32>(std::clamp(attenuation, 0, 3000)) << 6);
    postfx_options.store(options, std::memory_order_release);
}

HostPasses::PostFxPass::Settings Presenter::GetPostFxOptions() const {
    const u32 options = postfx_options.load(std::memory_order_acquire);
    return {static_cast<int>(options & 3), static_cast<int>((options >> 2) & 7),
            static_cast<int>((options >> 5) & 1), static_cast<int>(options >> 6)};
}

Frame* Presenter::PrepareBlankFrame(bool present_thread) {
    // Request a free presentation frame.
    Frame* frame = GetRenderFrame();

    auto& scheduler = present_thread ? present_scheduler : draw_scheduler;
    scheduler.EndRendering();

    const auto cmdbuf = scheduler.CommandBuffer();

    constexpr vk::ImageSubresourceRange simple_subresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const auto post_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const vk::RenderingAttachmentInfo attachment = {
        .imageView = frame->image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .extent = {frame->width, frame->height},
            },
        .layerCount = 1,
        .colorAttachmentCount = 1u,
        .pColorAttachments = &attachment,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    cmdbuf.beginRendering(rendering_info);
    cmdbuf.endRendering();

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &post_barrier,
    });

    // Flush frame creation commands.
    frame->ready_semaphore = scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    frame->frame_id = present_thread ? 0 : gcp_frame_id;
    frame->latch_ns = 0;
    SubmitInfo info{};
    scheduler.EndGpuTimingFrame();
    scheduler.Flush(info);
    if (!present_thread) {
        EndGuestFrame();
    }
    return frame;
}

void Presenter::Present(Frame* frame, bool is_reusing_frame, const u64 presentation_epoch) {
    if (presentation_epoch != 0) {
        std::scoped_lock lock{feedback_mutex};
        if (presentation_epoch != this->presentation_epoch) {
            if (!is_reusing_frame) {
                RecycleFrameAsync(frame);
            }
            return;
        }
    }

    // Recreate the swapchain if the window was resized.
    if (window.GetWidth() != swapchain.GetWidth() || window.GetHeight() != swapchain.GetHeight()) {
        RecreateSwapchain();
    }

    if (!swapchain.AcquireNextImage()) {
        RecreateSwapchain();
        if (!swapchain.AcquireNextImage()) {
            // User resizes the window too fast and GPU can't keep up. Skip this frame.
            LOG_WARNING(Render_Vulkan, "Skipping frame!");
            if (!is_reusing_frame) {
                RecycleFrameAsync(frame);
            }
            return;
        }
    }

    // Reset fence for queue submission. Do it here instead of GetRenderFrame() because we may
    // skip frame because of slow swapchain recreation. If a frame skip occurs, we skip signal
    // the frame's present fence and future GetRenderFrame() call will hang waiting for this frame.
    const auto reset_result = instance.GetDevice().resetFences(frame->present_done);
    ASSERT_MSG(reset_result == vk::Result::eSuccess,
               "Unexpected error resetting present done fence: {}", vk::to_string(reset_result));

    ImGuiID dockId = ImGui::Core::NewFrame(is_reusing_frame);

    const vk::Image swapchain_image = swapchain.Image();
    const vk::ImageView swapchain_image_view = swapchain.ImageView();

    auto& scheduler = present_scheduler;
    const auto cmdbuf = scheduler.CommandBuffer();
    // Presentation records on this thread; the overlay renderer needs the command buffer itself.
    const vk::CommandBuffer raw_cmdbuf = scheduler.RawCommandBuffer();
    const u32 capture_with_overlays_count = VideoCore::ConsumeWithOverlaysScreenshotRequests();
    std::vector<ScreenshotReadback> pending_screenshots;
    const bool is_guest_frame = !is_reusing_frame && frame->frame_id != 0;
    bool startup = startup_active;
    if (startup && !Core::Startup::progress.IsActive()) {
        // From here on presenting skips the startup screen with a plain flag.
        startup = startup_active = false;
        scheduler.DeferOperation([probe = std::move(startup_probe)] {});
    }
    if (startup && is_guest_frame) {
        Core::Startup::progress.GuestFlip();
    }
    const bool probe = startup && frame->frame_id != 0 && StartupProbeDue();
    if (capture_with_overlays_count > 0) {
        pending_screenshots.reserve(1);
    }

    if (EmulatorSettings.IsVkHostMarkersEnabled()) {
        cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
            .pLabelName = "Present",
        });
    }

    {
        auto* profiler_ctx = instance.GetProfilerContext();
        TracyVkNamedZoneC(profiler_ctx, renderer_gpu_zone, raw_cmdbuf, "Host frame",
                          MarkersPalette::GpuMarkerColor, profiler_ctx != nullptr);

        const vk::Extent2D extent = swapchain.GetExtent();
        const vk::ImageMemoryBarrier swapchain_pre_barrier{
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::DependencyFlagBits::eByRegion, {}, {}, swapchain_pre_barrier);

        if (!is_reusing_frame || probe) {
            const vk::ImageMemoryBarrier frame_pre_barrier{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .oldLayout = is_reusing_frame ? vk::ImageLayout::eShaderReadOnlyOptimal
                                              : vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = frame->image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   probe ? vk::PipelineStageFlagBits::eFragmentShader |
                                               vk::PipelineStageFlagBits::eComputeShader
                                         : vk::PipelineStageFlagBits::eFragmentShader,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, frame_pre_barrier);
        }
        if (probe) {
            const auto format = swapchain.GetSurfaceFormat().format;
            startup_probe->Record(cmdbuf, frame->image_view, {frame->width, frame->height},
                                  format == vk::Format::eB8G8R8A8Srgb ||
                                      format == vk::Format::eR8G8B8A8Srgb ||
                                      format == vk::Format::eA8B8G8R8SrgbPack32);
        }

        bool swapchain_copied_for_screenshot = false;

        { // Draw the game
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0f});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
            ImGui::SetNextWindowDockID(dockId, ImGuiCond_Once);
            if (ImGui::Begin("Display##game_display", nullptr, ImGuiWindowFlags_NoNav)) {
                auto game_texture = frame->imgui_texture;
                auto game_width = frame->width;
                auto game_height = frame->height;

                // The splash also stays up while the startup panel shows.
                if (Libraries::SystemService::IsSplashVisible() ||
                    (startup && EmulatorSettings.IsShowSplash())) {
                    if (!splash_img.has_value()) {
                        splash_img.emplace();
                        const auto& splash_data = Common::ElfInfo::Instance().GetSplashData();
                        if (!splash_data.empty()) {
                            splash_img = ImGui::RefCountedTexture::DecodePngTexture(splash_data);
                        }
                    }
                    if (auto& splash_image = this->splash_img.value()) {
                        auto [im_id, width, height] = splash_image.GetTexture();
                        game_texture = im_id;
                        game_width = width;
                        game_height = height;
                    }
                }

                ImVec2 contentArea = ImGui::GetContentRegionAvail();
                SetExpectedGameSize((s32)contentArea.x, (s32)contentArea.y);

                const auto imgRect =
                    FitImage(game_width, game_height, (s32)contentArea.x, (s32)contentArea.y);
                ImVec2 offset{
                    static_cast<float>(imgRect.offset.x),
                    static_cast<float>(imgRect.offset.y),
                };
                ImVec2 size{
                    static_cast<float>(imgRect.extent.width),
                    static_cast<float>(imgRect.extent.height),
                };

                ImGui::SetCursorPos(ImGui::GetCursorStartPos() + offset);
                ImGui::Image(game_texture, size);
                if (startup) {
                    ImGui::DrawStartupLoading(ImGui::GetWindowPos() + ImGui::GetCursorStartPos(),
                                              contentArea, Common::ElfInfo::Instance().Title(),
                                              Core::Startup::progress.GetStage(),
                                              Core::Startup::progress.ElapsedMs());
                }

                if (EmulatorSettings.IsNullGPU()) {
                    Core::Devtools::Layer::DrawNullGpuNotice();
                }
            }
            ImGui::End();
            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor();
        }
        {
            const GpuTimingScope timing{scheduler, GpuWork::Composition,
                                        GpuHandle(swapchain_image_view), GpuHandle(frame->image)};
            ImGui::Core::Render(raw_cmdbuf, swapchain_image_view, swapchain.GetExtent());
        }

        if (capture_with_overlays_count > 0) {
            pending_screenshots.emplace_back(
                instance, scheduler, ScreenshotKind::WithOverlays,
                BuildScreenshotPaths(ScreenshotKind::WithOverlays, capture_with_overlays_count),
                extent.width, extent.height,
                swapchain.GetHDR() ? vk::Format::eA2B10G10R10UnormPack32
                                   : swapchain.GetSurfaceFormat().format,
                swapchain.GetHDR());
            auto& readback = pending_screenshots.back();

            const vk::ImageMemoryBarrier to_transfer{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };

            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, to_transfer);
            CopyImageToReadback(cmdbuf, swapchain_image, vk::ImageLayout::eTransferSrcOptimal,
                                readback);
            swapchain_copied_for_screenshot = true;
        }

        const vk::AccessFlags post_src_access_mask =
            swapchain_copied_for_screenshot ? vk::AccessFlagBits::eTransferRead
                                            : vk::AccessFlagBits::eColorAttachmentWrite;
        const vk::ImageLayout post_old_layout = swapchain_copied_for_screenshot
                                                    ? vk::ImageLayout::eTransferSrcOptimal
                                                    : vk::ImageLayout::eColorAttachmentOptimal;
        const vk::ImageMemoryBarrier post_barrier{
            .srcAccessMask = post_src_access_mask,
            .dstAccessMask = vk::AccessFlagBits::eNone,
            .oldLayout = post_old_layout,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);

        if (profiler_ctx) {
            TracyVkCollect(profiler_ctx, raw_cmdbuf);
        }
    }
    if (EmulatorSettings.IsVkHostMarkersEnabled()) {
        cmdbuf.endDebugUtilsLabelEXT();
    }

    // Flush vulkan commands.
    std::shared_ptr<std::vector<ScreenshotReadback>> deferred_screenshots{};
    if (!pending_screenshots.empty()) {
        deferred_screenshots =
            std::make_shared<std::vector<ScreenshotReadback>>(std::move(pending_screenshots));
        scheduler.DeferPriorityOperation(
            [deferred_screenshots]() { SavePendingScreenshots(*deferred_screenshots); });
    }

    const u64 present_id = NextPresentId(frame, is_reusing_frame);
    const bool mark_latency = is_guest_frame && present_id != 0 && swapchain.HasLowLatency();
    SubmitInfo info{};
    if (mark_latency) {
        info.latency_present_id = present_id;
    }
    info.AddWait(swapchain.GetImageAcquiredSemaphore(), 1,
                 vk::PipelineStageFlagBits::eColorAttachmentOutput);
    info.AddWait(frame->ready_semaphore, frame->ready_tick,
                 probe ? vk::PipelineStageFlagBits::eFragmentShader |
                             vk::PipelineStageFlagBits::eComputeShader
                       : vk::PipelineStageFlagBits::eFragmentShader);
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    info.AddSignal(frame->present_done);
    // The command processor publishes a frame without waiting for its submission. A wait on
    // the queue for a value that is submitted to it only later can stall the queue.
    if (frame->ready_semaphore == draw_scheduler.GetMasterSemaphore()->Handle()) {
        draw_scheduler.WaitSubmitted(frame->ready_tick);
    }
    const u64 present_tick = scheduler.CurrentTick();
    ImGui::Core::TextureManager::EndFrame(scheduler);
    scheduler.EndGpuTimingFrame();
    scheduler.Flush(info);
    if (probe) {
        startup_probe_tick = present_tick;
    }
    if (is_guest_frame && frame->latch_ns != 0) {
        PaceVrrPresent(present_tick, frame->latch_ns);
    }
    // Present to swapchain.
    const s64 present_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
    if (mark_latency) {
        swapchain.SetLatencyMarker(present_id, vk::LatencyMarkerNV::ePresentStart);
    }
    const u64 swapchain_serial = swapchain.GetSerial();
    const bool present_succeeded = swapchain.Present(present_id);
    if (mark_latency) {
        swapchain.SetLatencyMarker(present_id, vk::LatencyMarkerNV::ePresentEnd);
    }
    if (present_succeeded && is_guest_frame && present_id != 0 && frame->latch_ns != 0 &&
        UsesDisplayPacing()) {
        {
            std::scoped_lock lock{present_wait_mutex};
            if (ready_wait_queue.size() >= MaxPendingPresentWaits) {
                ready_wait_queue.pop_front();
            }
            ready_wait_queue.push_back({
                .present_id = present_id,
                .present_tick = present_tick,
                .swapchain_serial = swapchain_serial,
                .latch_ns = frame->latch_ns,
                .ready_ns = 0,
                .present_ns = present_ns,
            });
        }
        ready_wait_cv.notify_one();
    }
    if (!present_succeeded) {
        // An out-of-date present may not consume its wait semaphore or signal a maintenance
        // fence. Finish only this scheduler timeline before retiring the old swapchain resources;
        // the wait happens on the host presentation thread and never on the guest vblank thread.
        present_scheduler.Finish();
        RecreateSwapchain();
    } else if (!is_reusing_frame) {
        RecordPresentCall(presentation_epoch);
    }

    if (!is_reusing_frame) {
        // The previous frame returns to the pool once the GPU is done presenting it. Waiting for
        // that here would pace this thread by the GPU: while the GPU runs a frame behind the
        // guest, each present would block for a whole frame and newer frames would be superseded
        // after the GPU already rendered them.
        if (present_succeeded) {
            RecycleFrameAsync(last_submit_frame);
            last_submit_frame = frame;
        } else {
            RecycleFrameAsync(frame);
        }
        DebugState.IncFlipFrameNum();
    }
}

bool Presenter::StartupProbeDue() {
    auto& progress = Core::Startup::progress;
    if (startup_probe_tick != 0) {
        if (!present_scheduler.IsFree(startup_probe_tick)) {
            return false;
        }
        startup_probe_tick = 0;
        if (startup_probe->Visible()) {
            progress.SetStage(Core::Startup::Stage::Complete);
            return false;
        }
    }
    const s64 now = progress.ElapsedMs();
    if (now < startup_next_probe_ms) {
        return false;
    }
    startup_next_probe_ms = now + 250;
    if (!startup_probe) {
        startup_probe = std::make_unique<HostPasses::StartupProbe>(instance);
    }
    return true;
}

Frame* Presenter::GetRenderFrame() {
    // Wait for free presentation frames
    Frame* frame;
    {
        std::unique_lock lock{free_mutex};
        free_cv.wait(lock, [this] { return !free_queue.empty(); });
        LOG_DEBUG(Render_Vulkan, "Got render frame, remaining {}", free_queue.size() - 1);

        // Take the frame from the queue
        frame = free_queue.front();
        free_queue.pop();
    }

    if (frame->width != expected_frame_width || frame->height != expected_frame_height ||
        frame->is_hdr != swapchain.GetHDR()) {
        RecreateFrame(frame, expected_frame_width, expected_frame_height);
    }

    return frame;
}

void Presenter::SetExpectedGameSize(s32 width, s32 height) {
    const float ratio = (float)width / (float)height;

    expected_frame_height = height;
    expected_frame_width = width;
    if (ratio > expected_ratio) {
        expected_frame_width = static_cast<s32>(height * expected_ratio);
    } else {
        expected_frame_height = static_cast<s32>(width / expected_ratio);
    }
}

} // namespace Vulkan
