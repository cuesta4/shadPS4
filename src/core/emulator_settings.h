// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream> // Windows static guest red-zone protection
#include <string>
#include <vector>
#include "common/types.h"
#include "core/cpu_patches.h" // Windows static guest red-zone protection

#define EmulatorSettings (*EmulatorSettingsImpl::GetInstance())

enum HideCursorState : int {
    Never,
    Idle,
    Always,
};

enum UsbBackendType : int {
    Real,
    SkylandersPortal,
    InfinityBase,
    DimensionsToypad,
};

enum GpuReadbacksMode : int {
    Disabled,
    Relaxed,
    Precise,
};

// Windows static guest red-zone protection
std::ostream& operator<<(std::ostream& output, WindowsGuestRedZoneProtectionMode mode);

enum class ConfigMode {
    Default,
    Global,
    Clean,
};

enum AudioBackend : int {
    SDL,
    OpenAL,
    // Add more backends as needed
};

enum OpenALHrtfMode : int {
    HrtfAuto, // Let OpenAL Soft decide (on for headphone-like stereo outputs)
    HrtfOn,   // Force HRTF binaural rendering
    HrtfOff,  // Never use HRTF
};

enum OpenALOutputMode : int {
    OutputAuto,       // Let OpenAL Soft negotiate with the device
    OutputStereo,     // Force stereo output
    OutputQuad,       // Force quadraphonic output
    OutputSurround51, // Force 5.1 surround output
    OutputSurround71, // Force 7.1 surround output
};

template <typename T>
struct Setting {
    T default_value{};
    T value{};
    std::optional<T> game_specific_value{};

    Setting() = default;
    // Single-argument ctor: initialises both default_value and value so
    // that CleanMode can always recover the intended factory default.
    /*implicit*/ Setting(T init) : default_value(std::move(init)), value(default_value) {}

    /// Return the active value under the given mode.
    T get(ConfigMode mode = ConfigMode::Default) const {
        switch (mode) {
        case ConfigMode::Default:
            return game_specific_value.value_or(value);
        case ConfigMode::Global:
            return value;
        case ConfigMode::Clean:
            return default_value;
        }
        return value;
    }

    /// Write v to the base layer.
    /// Set proper value as base or game_specific
    void set(const T& v, bool game_specific = false) {
        if (game_specific) {
            game_specific_value = v;
        } else {
            value = v;
        }
    }

    /// Discard the game-specific override; subsequent get(Default) will
    /// fall back to the base value.
    void reset_game_specific() {
        game_specific_value = std::nullopt;
    }
};

struct OverrideItem;

// -------------------------------
// Support types
// -------------------------------
struct GameInstallDir {
    std::filesystem::path path;
    bool enabled;
};

// -------------------------------
// General settings
// -------------------------------
struct GeneralSettings {
    Setting<std::vector<GameInstallDir>> install_dirs;
    Setting<std::filesystem::path> addon_install_dir;
    Setting<std::filesystem::path> home_dir;
    Setting<std::filesystem::path> sys_modules_dir;
    Setting<std::filesystem::path> font_dir;

    Setting<int> volume_slider{100};
    Setting<bool> neo_mode{false};
    Setting<bool> dev_kit_mode{false};
    Setting<int> extra_dmem_in_mbytes{0};
    Setting<int> extra_fmem_in_mbytes{0};
    Setting<u32> app0_read_bandwidth_mibps{0};
    Setting<bool> app0_read_disable_time_stretching{false};
    Setting<bool> shad_net_enabled{false};
    Setting<bool> trophy_popup_disabled{false};
    Setting<double> trophy_notification_duration{6.0};
    Setting<std::string> trophy_notification_side{"right"};
    Setting<bool> show_splash{false};
    Setting<bool> connected_to_network{false};
    Setting<bool> discord_rpc_enabled{false};
    Setting<bool> show_fps_counter{false};
    Setting<int> console_language{1};
    Setting<int> big_picture_scale{1000};
    Setting<std::string> shadnet_server{"srv.shadps4.net:31313"};
    Setting<std::string> shadnet_webapi_server{"http://srv.shadps4.net:31315"};
    Setting<std::string> signaling_info{};
    Setting<bool> enable_upnp{true};

    // return a vector of override descriptors (runtime, but tiny)
    std::vector<OverrideItem> GetOverrideableFields() const;
};

// -------------------------------
// Log settings
// -------------------------------
struct LogSettings {
    Setting<bool> append{false}; // specific
    Setting<bool> enable{true};  // specific
    Setting<std::string> filter{""};
    Setting<std::string> flush_level{""};
    Setting<u32> max_skip_duration{5'000};
    Setting<bool> separate{false}; // specific
    Setting<unsigned long long> size_limit{100_MB};
    Setting<bool> skip_duplicate{true};
    Setting<bool> sync{true};
#ifdef _WIN32
    Setting<std::string> type{"wincolor"};
#endif

    // return a vector of override descriptors (runtime, but tiny)
    std::vector<OverrideItem> GetOverrideableFields() const;
};

// -------------------------------
// Debug settings
// -------------------------------
struct DebugSettings {
    Setting<bool> debug_dump{false};         // specific
    Setting<bool> shader_collect{false};     // specific
    Setting<std::string> config_version{""}; // specific

    std::vector<OverrideItem> GetOverrideableFields() const;
};

// -------------------------------
// Input settings
// -------------------------------

struct InputSettings {
    Setting<int> cursor_state{HideCursorState::Idle};      // specific
    Setting<int> cursor_hide_timeout{5};                   // specific
    Setting<int> usb_device_backend{UsbBackendType::Real}; // specific
    Setting<bool> use_special_pad{false};
    Setting<int> special_pad_class{1};
    Setting<bool> motion_controls_enabled{true}; // specific
    Setting<bool> use_unified_input_config{true};
    Setting<std::string> default_controller_id{""};
    Setting<bool> background_controller_input{false}; // specific
    Setting<bool> ime_accessibility_enabled{false};   // specific
    Setting<bool> ime_url_mail_short_panel{false};    // specific
    Setting<bool> is_circle_enter{false};             // specific
    Setting<s32> camera_id{-1};
    Setting<bool> use_mice_as_mice{false};

    std::vector<OverrideItem> GetOverrideableFields() const;
};
// -------------------------------
// Audio settings
// -------------------------------
struct AudioSettings {
    Setting<u32> audio_backend{AudioBackend::SDL};
    Setting<std::string> sdl_mic_device{"Default Device"};
    Setting<std::string> sdl_main_output_device{"Default Device"};
    Setting<std::string> sdl_padSpk_output_device{"Default Device"};
    Setting<std::string> openal_mic_device{"Default Device"};
    Setting<std::string> openal_main_output_device{"Default Device"};
    Setting<std::string> openal_padSpk_output_device{"Default Device"};
    Setting<u32> openal_hrtf{OpenALHrtfMode::HrtfAuto};
    Setting<u32> openal_output_mode{OpenALOutputMode::OutputAuto};

    std::vector<OverrideItem> GetOverrideableFields() const;
};

// Windows static guest red-zone protection
struct WindowsGuestRedZoneProtectionSettings {
    Setting<WindowsGuestRedZoneProtectionMode> windows_guest_red_zone_protection_mode{
        WindowsGuestRedZoneProtectionMode::Disabled};

    std::vector<OverrideItem> GetOverrideableFields() const;
};

// -------------------------------
// GPU settings
// -------------------------------
struct GPUSettings {
    Setting<u32> window_width{1280};
    Setting<u32> window_height{720};
    Setting<u32> internal_screen_width{1280};
    Setting<u32> internal_screen_height{720};
    Setting<bool> null_gpu{false};
    Setting<bool> copy_gpu_buffers{false};
    Setting<u32> readbacks_mode{GpuReadbacksMode::Disabled};
    Setting<bool> readback_linear_images_enabled{false};
    Setting<bool> direct_memory_access_enabled{false};
    Setting<bool> dump_shaders{false};
    Setting<bool> patch_shaders{false};
    Setting<u32> vblank_frequency{60};
    Setting<bool> full_screen{false};
    Setting<std::string> full_screen_mode{"Windowed"};
    Setting<std::string> present_mode{"Mailbox"};
    Setting<bool> enable_reflex{false};
    // Presents each finished frame at an even delay after its vblank on a VRR display.
    Setting<bool> vrr_pacing{false};
    Setting<bool> hdr_allowed{false};
    Setting<int> upscaler{0};
    Setting<int> anti_aliasing{0};
    Setting<int> sharpening{0};
    Setting<bool> fsr_enabled{false};
    Setting<bool> rcas_enabled{true};
    Setting<int> rcas_attenuation{250};
    // TODO add overrides
    std::vector<OverrideItem> GetOverrideableFields() const;
};
// -------------------------------
// Vulkan settings
// -------------------------------
struct VulkanSettings {
    Setting<s32> gpu_id{-1};
    Setting<bool> renderdoc_enabled{false};
    Setting<bool> vkvalidation_enabled{false};
    Setting<bool> vkvalidation_core_enabled{true};
    Setting<bool> vkvalidation_sync_enabled{false};
    Setting<bool> vkvalidation_gpu_enabled{false};
    Setting<bool> vkcrash_diagnostic_enabled{false};
    Setting<bool> vkhost_markers{false};
    Setting<bool> vkguest_markers{false};
    Setting<bool> pipeline_cache_enabled{false};
    Setting<bool> pipeline_cache_archived{false};
    Setting<bool> async_shader_recompiling{false};
    Setting<bool> use_nv_raw_access_chains{false};
    Setting<bool> force_uniform_buffers{false};
    // Guest frames the GPU may lag behind the command processor; 0 leaves the GPU unbounded.
    Setting<u32> gpu_frames_ahead{2};
    std::vector<OverrideItem> GetOverrideableFields() const;
};

// -------------------------------
// Main manager
// -------------------------------
class EmulatorSettingsImpl {
public:
    EmulatorSettingsImpl();
    ~EmulatorSettingsImpl();

    static std::shared_ptr<EmulatorSettingsImpl> GetInstance();
    static void SetInstance(std::shared_ptr<EmulatorSettingsImpl> instance);

    bool Save(const std::string& serial = "");
    bool Load(const std::string& serial = "");
    void SetDefaultValues();
    bool TransferSettings();

    // Config mode
    ConfigMode GetConfigMode() const {
        return m_configMode;
    }
    void SetConfigMode(ConfigMode mode) {
        m_configMode = mode;
    }

    //
    // Game-specific override management
    /// Clears all per-game overrides.  Call this when a game exits so
    /// the emulator reverts to global settings.
    void ClearGameSpecificOverrides();

    /// Reset a single field's game-specific override by its JSON ke
    void ResetGameSpecificValue(const std::string& key);

    // general accessors
    bool AddGameInstallDir(const std::filesystem::path& dir, bool enabled = true);
    std::vector<std::filesystem::path> GetGameInstallDirs() const;
    void SetAllGameInstallDirs(const std::vector<GameInstallDir>& dirs);
    void RemoveGameInstallDir(const std::filesystem::path& dir);
    void SetGameInstallDirEnabled(const std::filesystem::path& dir, bool enabled);
    void SetGameInstallDirs(const std::vector<std::filesystem::path>& dirs_config);
    const std::vector<bool> GetGameInstallDirsEnabled();
    const std::vector<GameInstallDir>& GetAllGameInstallDirs() const;

    std::filesystem::path GetHomeDir();
    void SetHomeDir(const std::filesystem::path& dir);
    std::filesystem::path GetSysModulesDir();
    void SetSysModulesDir(const std::filesystem::path& dir);
    std::filesystem::path GetFontsDir();
    void SetFontsDir(const std::filesystem::path& dir);
    std::filesystem::path GetAddonInstallDir();
    void SetAddonInstallDir(const std::filesystem::path& dir);

private:
    GeneralSettings m_general{};
    LogSettings m_log{};
    DebugSettings m_debug{};
    InputSettings m_input{};
    AudioSettings m_audio{};
    // Windows static guest red-zone protection
    WindowsGuestRedZoneProtectionSettings m_windows_guest_red_zone_protection{};
    GPUSettings m_gpu{};
    VulkanSettings m_vulkan{};
    ConfigMode m_configMode{ConfigMode::Default};

    // Runtime-only override: when true, IsShadNetEnabled() reports false for the
    // rest of this run regardless of the persisted setting
    std::atomic<bool> m_shadnet_session_disabled{false};

    bool m_loaded{false};

    static std::shared_ptr<EmulatorSettingsImpl> s_instance;
    static std::mutex s_mutex;

    static void PrintChangedSummary(const std::vector<std::string>& changed);

public:
    // Add these getters to access overrideable fields
    std::vector<OverrideItem> GetGeneralOverrideableFields() const;
    std::vector<OverrideItem> GetDebugOverrideableFields() const;
    std::vector<OverrideItem> GetInputOverrideableFields() const;
    std::vector<OverrideItem> GetAudioOverrideableFields() const;
    // Windows static guest red-zone protection
    std::vector<OverrideItem> GetWindowsGuestRedZoneProtectionOverrideableFields() const;
    std::vector<OverrideItem> GetGPUOverrideableFields() const;
    std::vector<OverrideItem> GetVulkanOverrideableFields() const;
    std::vector<std::string> GetAllOverrideableKeys() const;

#define SETTING_FORWARD(group, Name, field)                                                        \
    auto Get##Name() const {                                                                       \
        return (group).field.get(m_configMode);                                                    \
    }                                                                                              \
    void Set##Name(const decltype((group).field.value)& v, bool specific = false) {                \
        (group).field.set(v, specific);                                                            \
    }
#define SETTING_FORWARD_BOOL(group, Name, field)                                                   \
    bool Is##Name() const {                                                                        \
        return (group).field.get(m_configMode);                                                    \
    }                                                                                              \
    void Set##Name(bool v, bool specific = false) {                                                \
        (group).field.set(v, specific);                                                            \
    }
#define SETTING_FORWARD_BOOL_READONLY(group, Name, field)                                          \
    bool Is##Name() const {                                                                        \
        return (group).field.get(m_configMode);                                                    \
    }

    // General settings
    SETTING_FORWARD(m_general, VolumeSlider, volume_slider)
    SETTING_FORWARD_BOOL(m_general, Neo, neo_mode)
    SETTING_FORWARD_BOOL(m_general, DevKit, dev_kit_mode)
    SETTING_FORWARD(m_general, ExtraDmemInMBytes, extra_dmem_in_mbytes)
    SETTING_FORWARD(m_general, ExtraFmemInMBytes, extra_fmem_in_mbytes)
    SETTING_FORWARD(m_general, App0ReadBandwidthMiBps, app0_read_bandwidth_mibps)
    SETTING_FORWARD_BOOL(m_general, App0ReadDisableTimeStretching,
                         app0_read_disable_time_stretching)
    bool IsShadNetEnabled() const {
        return m_general.shad_net_enabled.get(m_configMode) &&
               !m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    void SetShadNetEnabled(bool v, bool specific = false) {
        m_general.shad_net_enabled.set(v, specific);
    }
    bool IsShadNetEnabledSetting() const {
        return m_general.shad_net_enabled.get(m_configMode);
    }
    void SetShadNetSessionDisabled(bool v) {
        m_shadnet_session_disabled.store(v, std::memory_order_relaxed);
    }
    bool IsShadNetSessionDisabled() const {
        return m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    SETTING_FORWARD_BOOL(m_general, TrophyPopupDisabled, trophy_popup_disabled)
    SETTING_FORWARD(m_general, TrophyNotificationDuration, trophy_notification_duration)
    SETTING_FORWARD(m_general, TrophyNotificationSide, trophy_notification_side)
    SETTING_FORWARD_BOOL(m_general, ShowSplash, show_splash)
    SETTING_FORWARD_BOOL(m_general, ConnectedToNetwork, connected_to_network)
    SETTING_FORWARD_BOOL(m_general, DiscordRPCEnabled, discord_rpc_enabled)
    SETTING_FORWARD_BOOL(m_general, ShowFpsCounter, show_fps_counter)
    SETTING_FORWARD(m_general, ConsoleLanguage, console_language)
    SETTING_FORWARD(m_general, BigPictureScale, big_picture_scale)
    SETTING_FORWARD(m_general, ShadNetServer, shadnet_server)
    SETTING_FORWARD(m_general, ShadNetWebApiServer, shadnet_webapi_server)
    SETTING_FORWARD(m_general, SignalingInfo, signaling_info)
    SETTING_FORWARD_BOOL(m_general, UPnPEnabled, enable_upnp)

    // Log settings
    SETTING_FORWARD_BOOL(m_log, LogAppend, append)
    SETTING_FORWARD_BOOL(m_log, LogEnable, enable)
    SETTING_FORWARD(m_log, LogFilter, filter)
    SETTING_FORWARD(m_log, LogFlushLevel, flush_level)
    SETTING_FORWARD(m_log, LogMaxSkipDuration, max_skip_duration)
    SETTING_FORWARD_BOOL(m_log, LogSeparate, separate)
    SETTING_FORWARD(m_log, LogSizeLimit, size_limit)
    SETTING_FORWARD_BOOL(m_log, LogSkipDuplicate, skip_duplicate)
    SETTING_FORWARD_BOOL(m_log, LogSync, sync)
#ifdef _WIN32
    SETTING_FORWARD(m_log, LogType, type)
#endif

    // Audio settings
    SETTING_FORWARD(m_audio, AudioBackend, audio_backend)
    SETTING_FORWARD(m_audio, SDLMicDevice, sdl_mic_device)
    SETTING_FORWARD(m_audio, SDLMainOutputDevice, sdl_main_output_device)
    SETTING_FORWARD(m_audio, SDLPadSpkOutputDevice, sdl_padSpk_output_device)
    SETTING_FORWARD(m_audio, OpenALMicDevice, openal_mic_device)
    SETTING_FORWARD(m_audio, OpenALMainOutputDevice, openal_main_output_device)
    SETTING_FORWARD(m_audio, OpenALPadSpkOutputDevice, openal_padSpk_output_device)
    SETTING_FORWARD(m_audio, OpenALHrtf, openal_hrtf)
    SETTING_FORWARD(m_audio, OpenALOutputMode, openal_output_mode)

    // Windows static guest red-zone protection
    SETTING_FORWARD(m_windows_guest_red_zone_protection, WindowsGuestRedZoneProtectionMode,
                    windows_guest_red_zone_protection_mode)

    // Debug settings
    SETTING_FORWARD_BOOL(m_debug, DebugDump, debug_dump)
    SETTING_FORWARD_BOOL(m_debug, ShaderCollect, shader_collect)
    SETTING_FORWARD(m_debug, ConfigVersion, config_version)

    // GPU Settings
    SETTING_FORWARD_BOOL(m_gpu, NullGPU, null_gpu)
    SETTING_FORWARD_BOOL(m_gpu, DumpShaders, dump_shaders)
    SETTING_FORWARD_BOOL(m_gpu, CopyGpuBuffers, copy_gpu_buffers)
    SETTING_FORWARD_BOOL(m_gpu, FullScreen, full_screen)
    SETTING_FORWARD(m_gpu, FullScreenMode, full_screen_mode)
    SETTING_FORWARD(m_gpu, PresentMode, present_mode)
    SETTING_FORWARD_BOOL(m_gpu, ReflexEnabled, enable_reflex)
    SETTING_FORWARD_BOOL(m_gpu, VrrPacingEnabled, vrr_pacing)
    SETTING_FORWARD(m_gpu, WindowHeight, window_height)
    SETTING_FORWARD(m_gpu, WindowWidth, window_width)
    SETTING_FORWARD(m_gpu, InternalScreenHeight, internal_screen_height)
    SETTING_FORWARD(m_gpu, InternalScreenWidth, internal_screen_width)
    SETTING_FORWARD_BOOL(m_gpu, HdrAllowed, hdr_allowed)
    SETTING_FORWARD(m_gpu, Upscaler, upscaler)
    SETTING_FORWARD(m_gpu, AntiAliasing, anti_aliasing)
    SETTING_FORWARD(m_gpu, Sharpening, sharpening)
    SETTING_FORWARD_BOOL(m_gpu, FsrEnabled, fsr_enabled)
    SETTING_FORWARD_BOOL(m_gpu, RcasEnabled, rcas_enabled)
    SETTING_FORWARD(m_gpu, RcasAttenuation, rcas_attenuation)
    SETTING_FORWARD(m_gpu, ReadbacksMode, readbacks_mode)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackLinearImagesEnabled, readback_linear_images_enabled)
    SETTING_FORWARD_BOOL(m_gpu, DirectMemoryAccessEnabled, direct_memory_access_enabled)
    SETTING_FORWARD_BOOL_READONLY(m_gpu, PatchShaders, patch_shaders)

    u32 GetVblankFrequency() {
        if (m_gpu.vblank_frequency.value < 30) {
            return 30;
        }
        return m_gpu.vblank_frequency.get();
    }
    void SetVblankFrequency(const u32& v, bool is_specific = false) {
        u32 val = v < 30 ? 30 : v;
        if (is_specific) {
            m_gpu.vblank_frequency.game_specific_value = val;
        } else {
            m_gpu.vblank_frequency.value = val;
        }
    }

    // Input Settings
    SETTING_FORWARD(m_input, CursorState, cursor_state)
    SETTING_FORWARD(m_input, CursorHideTimeout, cursor_hide_timeout)
    SETTING_FORWARD(m_input, UsbDeviceBackend, usb_device_backend)
    SETTING_FORWARD_BOOL(m_input, MotionControlsEnabled, motion_controls_enabled)
    SETTING_FORWARD_BOOL(m_input, BackgroundControllerInput, background_controller_input)
    SETTING_FORWARD_BOOL(m_input, ImeAccessibilityEnabled, ime_accessibility_enabled)
    SETTING_FORWARD_BOOL(m_input, ImeUrlMailShortPanel, ime_url_mail_short_panel)
    SETTING_FORWARD(m_input, DefaultControllerId, default_controller_id)
    SETTING_FORWARD_BOOL(m_input, UsingSpecialPad, use_special_pad)
    SETTING_FORWARD(m_input, SpecialPadClass, special_pad_class)
    SETTING_FORWARD_BOOL(m_input, UseUnifiedInputConfig, use_unified_input_config)
    SETTING_FORWARD(m_input, CameraId, camera_id)
    SETTING_FORWARD_BOOL(m_input, CircleEnter, is_circle_enter)
    SETTING_FORWARD_BOOL(m_input, MiceUsedAsMice, use_mice_as_mice)

    // Vulkan settings
    SETTING_FORWARD(m_vulkan, GpuId, gpu_id)
    SETTING_FORWARD_BOOL(m_vulkan, RenderdocEnabled, renderdoc_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationEnabled, vkvalidation_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationCoreEnabled, vkvalidation_core_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationSyncEnabled, vkvalidation_sync_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationGpuEnabled, vkvalidation_gpu_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkCrashDiagnosticEnabled, vkcrash_diagnostic_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkHostMarkersEnabled, vkhost_markers)
    SETTING_FORWARD_BOOL(m_vulkan, VkGuestMarkersEnabled, vkguest_markers)
    SETTING_FORWARD_BOOL(m_vulkan, PipelineCacheEnabled, pipeline_cache_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, PipelineCacheArchived, pipeline_cache_archived)
    SETTING_FORWARD_BOOL(m_vulkan, AsyncShaderRecompiling, async_shader_recompiling)
    SETTING_FORWARD_BOOL(m_vulkan, NvRawAccessChainsEnabled, use_nv_raw_access_chains)
    SETTING_FORWARD_BOOL(m_vulkan, UniformBufferShadersEnabled, force_uniform_buffers)
    SETTING_FORWARD(m_vulkan, GpuFramesAhead, gpu_frames_ahead)

#undef SETTING_FORWARD
#undef SETTING_FORWARD_BOOL
#undef SETTING_FORWARD_BOOL_READONLY
};
