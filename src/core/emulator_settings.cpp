// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <common/path_util.h>
#include <common/scm_rev.h>
#include <toml.hpp>
#include "common/logging/formatter.h"
#include "common/logging/log.h"
#include "emulator_settings.h"
#include "emulator_settings_serialization.h"
#include "emulator_state.h"

#include <SDL3/SDL_messagebox.h>

using json = nlohmann::json;

std::ostream& operator<<(std::ostream& output, WindowsGuestRedZoneProtectionMode mode) {
    return output << nlohmann::json(mode).get<std::string>();
}

std::vector<OverrideItem> GeneralSettings::GetOverrideableFields() const {
    return std::vector<OverrideItem>{
        make_override<GeneralSettings>("volume_slider", &GeneralSettings::volume_slider),
        make_override<GeneralSettings>("neo_mode", &GeneralSettings::neo_mode),
        make_override<GeneralSettings>("dev_kit_mode", &GeneralSettings::dev_kit_mode),
        make_override<GeneralSettings>("extra_dmem_in_mbytes",
                                       &GeneralSettings::extra_dmem_in_mbytes),
        make_override<GeneralSettings>("extra_fmem_in_mbytes",
                                       &GeneralSettings::extra_fmem_in_mbytes),
        make_override<GeneralSettings>("app0_read_bandwidth_mibps",
                                       &GeneralSettings::app0_read_bandwidth_mibps),
        make_override<GeneralSettings>("app0_read_disable_time_stretching",
                                       &GeneralSettings::app0_read_disable_time_stretching),
        make_override<GeneralSettings>("shad_net_enabled", &GeneralSettings::shad_net_enabled),
        make_override<GeneralSettings>("trophy_popup_disabled",
                                       &GeneralSettings::trophy_popup_disabled),
        make_override<GeneralSettings>("trophy_notification_duration",
                                       &GeneralSettings::trophy_notification_duration),
        make_override<GeneralSettings>("show_splash", &GeneralSettings::show_splash),
        make_override<GeneralSettings>("trophy_notification_side",
                                       &GeneralSettings::trophy_notification_side),
        make_override<GeneralSettings>("connected_to_network",
                                       &GeneralSettings::connected_to_network),
        make_override<GeneralSettings>("console_language", &GeneralSettings::console_language),
        make_override<GeneralSettings>("shadnet_server", &GeneralSettings::shadnet_server),
        make_override<GeneralSettings>("shadnet_webapi_server",
                                       &GeneralSettings::shadnet_webapi_server),
        make_override<GeneralSettings>("signaling_info", &GeneralSettings::signaling_info),
        make_override<GeneralSettings>("enable_upnp", &GeneralSettings::enable_upnp)};
}

std::vector<OverrideItem> LogSettings::GetOverrideableFields() const {
    return std::vector<OverrideItem>{
        make_override<LogSettings>("append", &LogSettings::append),
        make_override<LogSettings>("enable", &LogSettings::enable),
        make_override<LogSettings>("filter", &LogSettings::filter),
        make_override<LogSettings>("flush_level", &LogSettings::flush_level),
        make_override<LogSettings>("max_skip_duration", &LogSettings::max_skip_duration),
        make_override<LogSettings>("separate", &LogSettings::separate),
        make_override<LogSettings>("size_limit", &LogSettings::size_limit),
        make_override<LogSettings>("skip_duplicate", &LogSettings::skip_duplicate),
        make_override<LogSettings>("sync", &LogSettings::sync),
#ifdef _WIN32
        make_override<LogSettings>("type", &LogSettings::type),
#endif
    };
}

std::vector<OverrideItem> DebugSettings::GetOverrideableFields() const {
    return std::vector<OverrideItem>{
        make_override<DebugSettings>("debug_dump", &DebugSettings::debug_dump),
        make_override<DebugSettings>("shader_collect", &DebugSettings::shader_collect)};
}

std::vector<OverrideItem> InputSettings::GetOverrideableFields() const {
    return std::vector<OverrideItem>{
        make_override<InputSettings>("cursor_state", &InputSettings::cursor_state),
        make_override<InputSettings>("cursor_hide_timeout",
                                     &InputSettings::cursor_hide_timeout),
        make_override<InputSettings>("usb_device_backend", &InputSettings::usb_device_backend),
        make_override<InputSettings>("motion_controls_enabled",
                                     &InputSettings::motion_controls_enabled),
        make_override<InputSettings>("background_controller_input",
                                     &InputSettings::background_controller_input),
        make_override<InputSettings>("ime_accessibility_enabled",
                                     &InputSettings::ime_accessibility_enabled),
        make_override<InputSettings>("ime_url_mail_short_panel",
                                     &InputSettings::ime_url_mail_short_panel),
        make_override<InputSettings>("is_circle_enter", &InputSettings::is_circle_enter),
        make_override<InputSettings>("camera_id", &InputSettings::camera_id),
        make_override<InputSettings>("use_mice_as_mice", &InputSettings::use_mice_as_mice)};
}

std::vector<OverrideItem> AudioSettings::GetOverrideableFields() const {
    return std::vector<OverrideItem>{
        make_override<AudioSettings>("audio_backend", &AudioSettings::audio_backend),
        make_override<AudioSettings>("sdl_mic_device", &AudioSettings::sdl_mic_device),
        make_override<AudioSettings>("sdl_main_output_device",
                                     &AudioSettings::sdl_main_output_device),
        make_override<AudioSettings>("sdl_padSpk_output_device",
                                     &AudioSettings::sdl_padSpk_output_device),
        make_override<AudioSettings>("openal_mic_device", &AudioSettings::openal_mic_device),
        make_override<AudioSettings>("openal_main_output_device",
                                     &AudioSettings::openal_main_output_device),
        make_override<AudioSettings>("openal_padSpk_output_device",
                                     &AudioSettings::openal_padSpk_output_device),
        make_override<AudioSettings>("openal_hrtf", &AudioSettings::openal_hrtf),
        make_override<AudioSettings>("openal_output_mode", &AudioSettings::openal_output_mode)};
}

std::vector<OverrideItem> WindowsGuestRedZoneProtectionSettings::GetOverrideableFields() const {
    return std::vector<OverrideItem>{make_override<WindowsGuestRedZoneProtectionSettings>(
        "windows_guest_red_zone_protection_mode",
        &WindowsGuestRedZoneProtectionSettings::windows_guest_red_zone_protection_mode)};
}

std::vector<OverrideItem> GPUSettings::GetOverrideableFields() const {
    return std::vector<OverrideItem>{
        make_override<GPUSettings>("null_gpu", &GPUSettings::null_gpu),
        make_override<GPUSettings>("copy_gpu_buffers", &GPUSettings::copy_gpu_buffers),
        make_override<GPUSettings>("full_screen", &GPUSettings::full_screen),
        make_override<GPUSettings>("full_screen_mode", &GPUSettings::full_screen_mode),
        make_override<GPUSettings>("present_mode", &GPUSettings::present_mode),
        make_override<GPUSettings>("enable_reflex", &GPUSettings::enable_reflex),
        make_override<GPUSettings>("vrr_pacing", &GPUSettings::vrr_pacing),
        make_override<GPUSettings>("window_height", &GPUSettings::window_height),
        make_override<GPUSettings>("window_width", &GPUSettings::window_width),
        make_override<GPUSettings>("hdr_allowed", &GPUSettings::hdr_allowed),
        make_override<GPUSettings>("upscaler", &GPUSettings::upscaler),
        make_override<GPUSettings>("anti_aliasing", &GPUSettings::anti_aliasing),
        make_override<GPUSettings>("sharpening", &GPUSettings::sharpening),
        make_override<GPUSettings>("fsr_enabled", &GPUSettings::fsr_enabled),
        make_override<GPUSettings>("rcas_enabled", &GPUSettings::rcas_enabled),
        make_override<GPUSettings>("rcas_attenuation", &GPUSettings::rcas_attenuation),
        make_override<GPUSettings>("dump_shaders", &GPUSettings::dump_shaders),
        make_override<GPUSettings>("patch_shaders", &GPUSettings::patch_shaders),
        make_override<GPUSettings>("readbacks_mode", &GPUSettings::readbacks_mode),
        make_override<GPUSettings>("readback_linear_images_enabled",
                                   &GPUSettings::readback_linear_images_enabled),
        make_override<GPUSettings>("direct_memory_access_enabled",
                                   &GPUSettings::direct_memory_access_enabled),
        make_override<GPUSettings>("vblank_frequency", &GPUSettings::vblank_frequency),
    };
}

std::vector<OverrideItem> VulkanSettings::GetOverrideableFields() const {
    return std::vector<OverrideItem>{
        make_override<VulkanSettings>("gpu_id", &VulkanSettings::gpu_id),
        make_override<VulkanSettings>("renderdoc_enabled", &VulkanSettings::renderdoc_enabled),
        make_override<VulkanSettings>("vkvalidation_enabled",
                                      &VulkanSettings::vkvalidation_enabled),
        make_override<VulkanSettings>("vkvalidation_core_enabled",
                                      &VulkanSettings::vkvalidation_core_enabled),
        make_override<VulkanSettings>("vkvalidation_sync_enabled",
                                      &VulkanSettings::vkvalidation_sync_enabled),
        make_override<VulkanSettings>("vkvalidation_gpu_enabled",
                                      &VulkanSettings::vkvalidation_gpu_enabled),
        make_override<VulkanSettings>("vkcrash_diagnostic_enabled",
                                      &VulkanSettings::vkcrash_diagnostic_enabled),
        make_override<VulkanSettings>("vkhost_markers", &VulkanSettings::vkhost_markers),
        make_override<VulkanSettings>("vkguest_markers", &VulkanSettings::vkguest_markers),
        make_override<VulkanSettings>("pipeline_cache_enabled",
                                      &VulkanSettings::pipeline_cache_enabled),
        make_override<VulkanSettings>("pipeline_cache_archived",
                                      &VulkanSettings::pipeline_cache_archived),
        make_override<VulkanSettings>("async_shader_recompiling",
                                      &VulkanSettings::async_shader_recompiling),
        make_override<VulkanSettings>("use_nv_raw_access_chains",
                                      &VulkanSettings::use_nv_raw_access_chains),
        make_override<VulkanSettings>("force_uniform_buffers",
                                      &VulkanSettings::force_uniform_buffers),
        make_override<VulkanSettings>("gpu_frames_ahead", &VulkanSettings::gpu_frames_ahead),
    };
}

std::vector<OverrideItem> EmulatorSettingsImpl::GetGeneralOverrideableFields() const {
    return m_general.GetOverrideableFields();
}

std::vector<OverrideItem> EmulatorSettingsImpl::GetDebugOverrideableFields() const {
    return m_debug.GetOverrideableFields();
}

std::vector<OverrideItem> EmulatorSettingsImpl::GetInputOverrideableFields() const {
    return m_input.GetOverrideableFields();
}

std::vector<OverrideItem> EmulatorSettingsImpl::GetAudioOverrideableFields() const {
    return m_audio.GetOverrideableFields();
}

std::vector<OverrideItem>
EmulatorSettingsImpl::GetWindowsGuestRedZoneProtectionOverrideableFields() const {
    return m_windows_guest_red_zone_protection.GetOverrideableFields();
}

std::vector<OverrideItem> EmulatorSettingsImpl::GetGPUOverrideableFields() const {
    return m_gpu.GetOverrideableFields();
}

std::vector<OverrideItem> EmulatorSettingsImpl::GetVulkanOverrideableFields() const {
    return m_vulkan.GetOverrideableFields();
}

static void MigratePostFx(json& config, bool legacy_fsr, bool legacy_rcas) {
    if (!config.contains("GPU")) {
        return;
    }
    auto& gpu = config["GPU"];
    const bool enabled = gpu.value("fsr_enabled", legacy_fsr);
    if (!gpu.contains("upscaler") && gpu.contains("fsr_enabled")) {
        gpu["upscaler"] = enabled ? 1 : 0;
    }
    if (!gpu.contains("sharpening") &&
        (gpu.contains("fsr_enabled") || gpu.contains("rcas_enabled"))) {
        gpu["sharpening"] = enabled && gpu.value("rcas_enabled", legacy_rcas) ? 1 : 0;
    }
    if (const auto it = gpu.find("upscaler"); it != gpu.end() && *it != 0 && *it != 1) {
        *it = 0;
    }
    if (const auto it = gpu.find("anti_aliasing");
        it != gpu.end() && *it != 0 && *it != 1 && *it != 3 && *it != 4) {
        *it = 0;
    }
}

// ── Singleton storage ─────────────────────────────────────────────────
std::shared_ptr<EmulatorSettingsImpl> EmulatorSettingsImpl::s_instance = nullptr;
std::mutex EmulatorSettingsImpl::s_mutex;

namespace toml {
// why is it so hard to avoid exceptions with this library
template <typename T>
std::optional<T> get_optional(const toml::value& v, const std::string& key) {
    if (!v.is_table())
        return std::nullopt;
    const auto& tbl = v.as_table();
    auto it = tbl.find(key);
    if (it == tbl.end())
        return std::nullopt;

    if constexpr (std::is_same_v<T, int>) {
        if (it->second.is_integer()) {
            return static_cast<int>(toml::get<int>(it->second));
        }
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        if (it->second.is_integer()) {
            return static_cast<u32>(toml::get<unsigned int>(it->second));
        }
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        if (it->second.is_integer()) {
            return static_cast<long long>(toml::get<unsigned long long>(it->second));
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (it->second.is_floating()) {
            return toml::get<double>(it->second);
        }
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (it->second.is_string()) {
            return toml::get<std::string>(it->second);
        }
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
        if (it->second.is_string()) {
            return toml::get<std::string>(it->second);
        }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (it->second.is_boolean()) {
            return toml::get<bool>(it->second);
        }
    } else {
        static_assert([] { return false; }(), "Unsupported type in get_optional<T>");
    }

    return std::nullopt;
}

} // namespace toml

// ── Helpers ───────────────────────────────────────────────────────────

void EmulatorSettingsImpl::PrintChangedSummary(const std::vector<std::string>& changed) {
    if (changed.empty()) {
        return;
    }
    LOG_DEBUG(Config, "Game-specific overrides applied:");
    for (const auto& k : changed)
        LOG_DEBUG(Config, "    * {}", k);
}

// ── Singleton ────────────────────────────────────────────────────────
EmulatorSettingsImpl::EmulatorSettingsImpl() = default;

EmulatorSettingsImpl::~EmulatorSettingsImpl() {
    if (m_loaded)
        Save();
}

std::shared_ptr<EmulatorSettingsImpl> EmulatorSettingsImpl::GetInstance() {
    std::lock_guard lock(s_mutex);
    if (!s_instance)
        s_instance = std::make_shared<EmulatorSettingsImpl>();
    return s_instance;
}

void EmulatorSettingsImpl::SetInstance(std::shared_ptr<EmulatorSettingsImpl> instance) {
    std::lock_guard lock(s_mutex);
    s_instance = std::move(instance);
}

// --------------------
// General helpers
// --------------------
bool EmulatorSettingsImpl::AddGameInstallDir(const std::filesystem::path& dir, bool enabled) {
    for (const auto& d : m_general.install_dirs.value)
        if (d.path == dir)
            return false;
    m_general.install_dirs.value.push_back({dir, enabled});
    return true;
}

std::vector<std::filesystem::path> EmulatorSettingsImpl::GetGameInstallDirs() const {
    std::vector<std::filesystem::path> out;
    for (const auto& d : m_general.install_dirs.value)
        if (d.enabled)
            out.push_back(d.path);
    return out;
}

const std::vector<GameInstallDir>& EmulatorSettingsImpl::GetAllGameInstallDirs() const {
    return m_general.install_dirs.value;
}

void EmulatorSettingsImpl::SetAllGameInstallDirs(const std::vector<GameInstallDir>& dirs) {
    m_general.install_dirs.value = dirs;
}

void EmulatorSettingsImpl::RemoveGameInstallDir(const std::filesystem::path& dir) {
    auto iterator =
        std::find_if(m_general.install_dirs.value.begin(), m_general.install_dirs.value.end(),
                     [&dir](const GameInstallDir& install_dir) { return install_dir.path == dir; });
    if (iterator != m_general.install_dirs.value.end()) {
        m_general.install_dirs.value.erase(iterator);
    }
}

void EmulatorSettingsImpl::SetGameInstallDirEnabled(const std::filesystem::path& dir,
                                                    bool enabled) {
    auto iterator =
        std::find_if(m_general.install_dirs.value.begin(), m_general.install_dirs.value.end(),
                     [&dir](const GameInstallDir& install_dir) { return install_dir.path == dir; });
    if (iterator != m_general.install_dirs.value.end()) {
        iterator->enabled = enabled;
    }
}

void EmulatorSettingsImpl::SetGameInstallDirs(
    const std::vector<std::filesystem::path>& dirs_config) {
    m_general.install_dirs.value.clear();
    for (const auto& dir : dirs_config) {
        m_general.install_dirs.value.push_back({dir, true});
    }
}

const std::vector<bool> EmulatorSettingsImpl::GetGameInstallDirsEnabled() {
    std::vector<bool> enabled_dirs;
    for (const auto& dir : m_general.install_dirs.value) {
        enabled_dirs.push_back(dir.enabled);
    }
    return enabled_dirs;
}

std::filesystem::path EmulatorSettingsImpl::GetHomeDir() {
    if (m_general.home_dir.value.empty()) {
        return Common::FS::GetUserPath(Common::FS::PathType::HomeDir);
    }
    return m_general.home_dir.value;
}

void EmulatorSettingsImpl::SetHomeDir(const std::filesystem::path& dir) {
    m_general.home_dir.value = dir;
}

std::filesystem::path EmulatorSettingsImpl::GetSysModulesDir() {
    if (m_general.sys_modules_dir.value.empty()) {
        return Common::FS::GetUserPath(Common::FS::PathType::SysModuleDir);
    }
    return m_general.sys_modules_dir.value;
}

void EmulatorSettingsImpl::SetSysModulesDir(const std::filesystem::path& dir) {
    m_general.sys_modules_dir.value = dir;
}

std::filesystem::path EmulatorSettingsImpl::GetFontsDir() {
    if (m_general.font_dir.value.empty()) {
        return Common::FS::GetUserPath(Common::FS::PathType::FontsDir);
    }
    return m_general.font_dir.value;
}

void EmulatorSettingsImpl::SetFontsDir(const std::filesystem::path& dir) {
    m_general.font_dir.value = dir;
}

std::filesystem::path EmulatorSettingsImpl::GetAddonInstallDir() {
    if (m_general.addon_install_dir.value.empty()) {
        return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "addcont";
    }
    return m_general.addon_install_dir.value;
}

void EmulatorSettingsImpl::SetAddonInstallDir(const std::filesystem::path& dir) {
    m_general.addon_install_dir.value = dir;
}

// ── Game-specific override management ────────────────────────────────
void EmulatorSettingsImpl::ClearGameSpecificOverrides() {
    SettingsSerialization::ClearGroupOverrides(m_general);
    SettingsSerialization::ClearGroupOverrides(m_log);
    SettingsSerialization::ClearGroupOverrides(m_debug);
    SettingsSerialization::ClearGroupOverrides(m_input);
    SettingsSerialization::ClearGroupOverrides(m_audio);
    // Windows static guest red-zone protection
    SettingsSerialization::ClearGroupOverrides(m_windows_guest_red_zone_protection);
    SettingsSerialization::ClearGroupOverrides(m_gpu);
    SettingsSerialization::ClearGroupOverrides(m_vulkan);
}

void EmulatorSettingsImpl::ResetGameSpecificValue(const std::string& key) {
    // Walk every overrideable group until we find the matching key.
    auto tryGroup = [&key](auto& group) {
        for (auto& item : group.GetOverrideableFields()) {
            if (key == item.key) {
                item.reset_game_specific(&group);
                return true;
            }
        }
        return false;
    };
    if (tryGroup(m_general))
        return;
    if (tryGroup(m_log))
        return;
    if (tryGroup(m_debug))
        return;
    if (tryGroup(m_input))
        return;
    if (tryGroup(m_audio))
        return;
    // Windows static guest red-zone protection
    if (tryGroup(m_windows_guest_red_zone_protection))
        return;
    if (tryGroup(m_gpu))
        return;
    if (tryGroup(m_vulkan))
        return;
    LOG_WARNING(Config, "ResetGameSpecificValue: key '{}' not found", key);
}

bool EmulatorSettingsImpl::Save(const std::string& serial) {
    try {
        if (!serial.empty()) {
            const auto cfgDir = Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs);
            std::filesystem::create_directories(cfgDir);
            const auto path = cfgDir / (serial + ".json");

            json j = json::object();

            json generalObj = json::object();
            SettingsSerialization::SaveGroupGameSpecific(m_general, generalObj);
            j["General"] = generalObj;

            json logObj = json::object();
            SettingsSerialization::SaveGroupGameSpecific(m_log, logObj);
            j["Log"] = logObj;

            json debugObj = json::object();
            SettingsSerialization::SaveGroupGameSpecific(m_debug, debugObj);
            j["Debug"] = debugObj;

            json inputObj = json::object();
            SettingsSerialization::SaveGroupGameSpecific(m_input, inputObj);
            j["Input"] = inputObj;

            json audioObj = json::object();
            SettingsSerialization::SaveGroupGameSpecific(m_audio, audioObj);
            j["Audio"] = audioObj;

            // Windows static guest red-zone protection
            json windowsGuestRedZoneProtectionObj = json::object();
            SettingsSerialization::SaveGroupGameSpecific(m_windows_guest_red_zone_protection,
                                  windowsGuestRedZoneProtectionObj);
            j["WindowsGuestRedZoneProtection"] = windowsGuestRedZoneProtectionObj;

            json gpuObj = json::object();
            SettingsSerialization::SaveGroupGameSpecific(m_gpu, gpuObj);
            j["GPU"] = gpuObj;

            json vulkanObj = json::object();
            SettingsSerialization::SaveGroupGameSpecific(m_vulkan, vulkanObj);
            j["Vulkan"] = vulkanObj;

            std::ofstream out(path);
            if (!out) {
                LOG_ERROR(Config, "Failed to open game config for writing: {}", path.string());
                return false;
            }
            out << std::setw(2) << j;
            return !out.fail();

        } else {
            // ── Global config.json ─────────────────────────────────────
            const auto path =
                Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "config.json";

            SetConfigVersion(Common::g_scm_rev);

            json j;
            j["General"] = m_general;
            j["Log"] = m_log;
            j["Debug"] = m_debug;
            j["Input"] = m_input;
            j["Audio"] = m_audio;
            j["GPU"] = m_gpu;
            j["Vulkan"] = m_vulkan;

            // Read the existing file so we can preserve keys unknown to this build
            json existing = json::object();
            if (std::ifstream existingIn{path}; existingIn.good()) {
                try {
                    existingIn >> existing;
                } catch (...) {
                    existing = json::object();
                }
            }

            // Merge: update each section's known keys, but leave unknown keys intact
            for (auto& [section, val] : j.items()) {
                if (existing.contains(section) && existing[section].is_object() && val.is_object())
                    existing[section].update(val); // overwrites known keys, keeps unknown ones
                else
                    existing[section] = val;
            }

            std::ofstream out(path);
            if (!out) {
                LOG_ERROR(Config, "Failed to open config for writing: {}", path.string());
                return false;
            }
            out << std::setw(2) << existing;
            return !out.fail();
        }
    } catch (const std::exception& e) {
        LOG_ERROR(Config, "Error saving settings: {}", e.what());
        return false;
    }
}

// ── Load ──────────────────────────────────────────────────────────────

bool EmulatorSettingsImpl::Load(const std::string& serial) {
    // A newly loaded profile replaces, rather than extends, the previous profile.
    ClearGameSpecificOverrides(); // Windows static guest red-zone protection

    try {
        if (serial.empty()) {
            // ── Global config ──────────────────────────────────────────
            const auto userDir = Common::FS::GetUserPath(Common::FS::PathType::UserDir);
            const auto configPath = userDir / "config.json";

            if (std::ifstream in{configPath}; in.good()) {
                json gj;
                in >> gj;
                MigratePostFx(gj, m_gpu.fsr_enabled.get(), m_gpu.rcas_enabled.get());

                auto mergeGroup = [&gj](auto& group, const char* section) {
                    if (!gj.contains(section))
                        return;
                    json current = group;
                    current.update(gj.at(section));
                    group = current.get<std::remove_reference_t<decltype(group)>>();
                };

                mergeGroup(m_general, "General");
                mergeGroup(m_log, "Log");
                mergeGroup(m_debug, "Debug");
                mergeGroup(m_input, "Input");
                mergeGroup(m_audio, "Audio");
                mergeGroup(m_gpu, "GPU");
                mergeGroup(m_vulkan, "Vulkan");
            } else {
                if (std::filesystem::exists(Common::FS::GetUserPath(Common::FS::PathType::UserDir) /
                                            "config.toml")) {
                    SDL_MessageBoxButtonData btns[2]{
                        {0, 0, "Update"},
                        {0, 1, "Defaults"},
                    };
                    SDL_MessageBoxData msg_box{
                        0,
                        nullptr,
                        "Config Migration",
                        "The shadPS4 config backend has been updated, and you only have "
                        "the old version of the config. Do you wish to update it "
                        "automatically, or continue with the default config?",
                        2,
                        btns,
                        nullptr,
                    };
                    int result = 1;
                    SDL_ShowMessageBox(&msg_box, &result);
                    if (result == 0) {
                        if (TransferSettings()) {
                            m_loaded = true;
                            Save();
                            return true;
                        } else {
                            SDL_ShowSimpleMessageBox(0, "Config Migration",
                                                     "Error transferring settings, exiting.",
                                                     nullptr);
                            std::quick_exit(1);
                        }
                    }
                }
                SetDefaultValues();
                Save();
            }
            if (GetConfigVersion() != Common::g_scm_rev) {
                Save();
            }
            m_loaded = true;
            return true;
        } else {
            // ── Per-game override file ─────────────────────────────────
            // Never reloads global settings. Only applies
            // game_specific_value overrides on top of the already-loaded
            // base configuration.
            const auto gamePath =
                Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs) / (serial + ".json");

            if (!std::filesystem::exists(gamePath)) {
                return false;
            }

            std::ifstream in(gamePath);
            if (!in) {
                return false;
            }

            json gj;
            in >> gj;
            MigratePostFx(gj, m_gpu.upscaler.get() == 1, m_gpu.sharpening.get() == 1);

            std::vector<std::string> changed;

            // ApplyGroupOverrides now correctly stores values as
            // game_specific_value (see make_override in the header).
            // ConfigMode::Default will then resolve them at getter call
            // time without ever touching the base values.
            if (gj.contains("General"))
                SettingsSerialization::ApplyGroupOverrides(m_general, gj.at("General"), changed);
            if (gj.contains("Log"))
                SettingsSerialization::ApplyGroupOverrides(m_log, gj.at("Log"), changed);
            if (gj.contains("Debug"))
                SettingsSerialization::ApplyGroupOverrides(m_debug, gj.at("Debug"), changed);
            if (gj.contains("Input"))
                SettingsSerialization::ApplyGroupOverrides(m_input, gj.at("Input"), changed);
            if (gj.contains("Audio"))
                SettingsSerialization::ApplyGroupOverrides(m_audio, gj.at("Audio"), changed);
            // Windows static guest red-zone protection
            if (gj.contains("WindowsGuestRedZoneProtection"))
                SettingsSerialization::ApplyGroupOverrides(m_windows_guest_red_zone_protection,
                    gj.at("WindowsGuestRedZoneProtection"), changed);
            if (gj.contains("GPU"))
                SettingsSerialization::ApplyGroupOverrides(m_gpu, gj.at("GPU"), changed);
            if (gj.contains("Vulkan"))
                SettingsSerialization::ApplyGroupOverrides(m_vulkan, gj.at("Vulkan"), changed);

            PrintChangedSummary(changed);
            EmulatorState::GetInstance()->SetGameSpecifigConfigUsed(true);
            return true;
        }
    } catch (const std::exception& e) {
        LOG_ERROR(Config, "Error loading settings: {}", e.what());
        return false;
    }
}

void EmulatorSettingsImpl::SetDefaultValues() {
    m_general = GeneralSettings{};
    m_log = LogSettings{};
    m_debug = DebugSettings{};
    m_input = InputSettings{};
    m_audio = AudioSettings{};
    // Windows static guest red-zone protection
    m_windows_guest_red_zone_protection = WindowsGuestRedZoneProtectionSettings{};
    m_gpu = GPUSettings{};
    m_vulkan = VulkanSettings{};
}

bool EmulatorSettingsImpl::TransferSettings() {
    toml::value og_data;
    json new_data = json::object();
    try {
        auto path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "config.toml";
        std::ifstream ifs;
        ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        ifs.open(path, std::ios_base::binary);
        og_data = toml::parse(ifs, std::string{fmt::UTF(path.filename().u8string()).data});
    } catch (std::exception& ex) {
        fmt::print("Got exception trying to load config file. Exception: {}\n", ex.what());
        return false;
    }
    auto setFromToml = [&]<typename T>(Setting<T>& n, toml::value const& t, std::string k) {
        n = toml::get_optional<T>(t, k).value_or(n.default_value);
    };
    if (og_data.contains("General")) {
        const toml::value& general = og_data.at("General");
        auto& s = m_general;

        setFromToml(s.volume_slider, general, "volumeSlider");
        setFromToml(s.neo_mode, general, "isPS4Pro");
        setFromToml(s.dev_kit_mode, general, "isDevKit");
        setFromToml(s.trophy_popup_disabled, general, "isTrophyPopupDisabled");
        setFromToml(s.trophy_notification_duration, general, "trophyNotificationDuration");
        setFromToml(s.discord_rpc_enabled, general, "enableDiscordRPC");
        setFromToml(s.show_splash, general, "showSplash");
        setFromToml(s.trophy_notification_side, general, "sideTrophy");
        setFromToml(s.connected_to_network, general, "isConnectedToNetwork");
        setFromToml(s.sys_modules_dir, general, "sysModulesPath");
        setFromToml(s.font_dir, general, "fontsPath");
        // setFromToml(, general, "userName");
        // setFromToml(s.defaultControllerID, general, "defaultControllerID");
    }

    if (og_data.contains("Log")) {
        const toml::value& log = og_data.at("Log");
        auto& s = m_log;

        setFromToml(s.append, log, "append");
        setFromToml(s.enable, log, "enable");
        setFromToml(s.filter, log, "filter");
        setFromToml(s.max_skip_duration, log, "maxSkipDuration");
        setFromToml(s.separate, log, "separate");
        setFromToml(s.size_limit, log, "sizeLimit");
        setFromToml(s.skip_duplicate, log, "skipDuplicate");
        setFromToml(s.sync, log, "sync");
#ifdef _WIN32
        setFromToml(s.type, log, "type");
#endif
    }

    if (og_data.contains("General")) {
        const toml::value& general = og_data.at("General");
        auto& s = m_log;

        setFromToml(s.filter, general, "logFilter");
        setFromToml(s.skip_duplicate, general, "isIdenticalLogGrouped");
        Setting<std::string> logType("sync");
        setFromToml(logType, general, "logType");
        if (logType.get() == "sync") {
            s.sync = true;
        } else {
            s.sync = false;
        }
    }

    if (og_data.contains("Debug")) {
        const toml::value& debug = og_data.at("Debug");
        auto& s = m_log;

        setFromToml(s.enable, debug, "logEnabled");
        setFromToml(s.separate, debug, "isSeparateLogFilesEnabled");
    }

    if (og_data.contains("Input")) {
        const toml::value& input = og_data.at("Input");
        auto& s = m_input;

        setFromToml(s.cursor_state, input, "cursorState");
        setFromToml(s.cursor_hide_timeout, input, "cursorHideTimeout");
        setFromToml(s.use_special_pad, input, "useSpecialPad");
        setFromToml(s.special_pad_class, input, "specialPadClass");
        setFromToml(s.motion_controls_enabled, input, "isMotionControlsEnabled");
        setFromToml(s.use_unified_input_config, input, "useUnifiedInputConfig");
        setFromToml(s.background_controller_input, input, "backgroundControllerInput");
        setFromToml(s.ime_accessibility_enabled, input, "imeAccessibilityEnabled");
        setFromToml(s.ime_url_mail_short_panel, input, "imeUrlMailShortPanel");
        setFromToml(s.usb_device_backend, input, "usbDeviceBackend");
    }

    if (og_data.contains("Audio")) {
        const toml::value& audio = og_data.at("Audio");
        auto& s = m_audio;

        setFromToml(s.sdl_mic_device, audio, "micDevice");
        setFromToml(s.sdl_main_output_device, audio, "mainOutputDevice");
        setFromToml(s.sdl_padSpk_output_device, audio, "padSpkOutputDevice");
    }

    if (og_data.contains("GPU")) {
        const toml::value& gpu = og_data.at("GPU");
        auto& s = m_gpu;

        setFromToml(s.window_width, gpu, "screenWidth");
        setFromToml(s.window_height, gpu, "screenHeight");
        setFromToml(s.internal_screen_width, gpu, "internalScreenWidth");
        setFromToml(s.internal_screen_height, gpu, "internalScreenHeight");
        setFromToml(s.null_gpu, gpu, "nullGpu");
        setFromToml(s.copy_gpu_buffers, gpu, "copyGPUBuffers");
        setFromToml(s.readbacks_mode, gpu, "readbacksMode");
        setFromToml(s.readback_linear_images_enabled, gpu, "readbackLinearImages");
        setFromToml(s.direct_memory_access_enabled, gpu, "directMemoryAccess");
        setFromToml(s.dump_shaders, gpu, "dumpShaders");
        setFromToml(s.patch_shaders, gpu, "patchShaders");
        setFromToml(s.vblank_frequency, gpu, "vblankFrequency");
        setFromToml(s.full_screen, gpu, "Fullscreen");
        setFromToml(s.full_screen_mode, gpu, "FullscreenMode");
        setFromToml(s.present_mode, gpu, "presentMode");
        setFromToml(s.hdr_allowed, gpu, "allowHDR");
        setFromToml(s.fsr_enabled, gpu, "fsrEnabled");
        setFromToml(s.rcas_enabled, gpu, "rcasEnabled");
        setFromToml(s.rcas_attenuation, gpu, "rcasAttenuation");
        s.upscaler.set(s.fsr_enabled.get() ? 1 : 0);
        s.sharpening.set(s.fsr_enabled.get() && s.rcas_enabled.get() ? 1 : 0);
    }

    if (og_data.contains("Vulkan")) {
        const toml::value& vk = og_data.at("Vulkan");
        auto& s = m_vulkan;

        setFromToml(s.gpu_id, vk, "gpuId");
        setFromToml(s.vkvalidation_enabled, vk, "validation");
        setFromToml(s.vkvalidation_core_enabled, vk, "validation_core");
        setFromToml(s.vkvalidation_sync_enabled, vk, "validation_sync");
        setFromToml(s.vkvalidation_gpu_enabled, vk, "validation_gpu");
        setFromToml(s.vkcrash_diagnostic_enabled, vk, "crashDiagnostic");
        setFromToml(s.vkhost_markers, vk, "hostMarkers");
        setFromToml(s.vkguest_markers, vk, "guestMarkers");
        setFromToml(s.renderdoc_enabled, vk, "rdocEnable");
        setFromToml(s.pipeline_cache_enabled, vk, "pipelineCacheEnable");
        setFromToml(s.pipeline_cache_archived, vk, "pipelineCacheArchive");
    }

    if (og_data.contains("Debug")) {
        const toml::value& debug = og_data.at("Debug");
        auto& s = m_debug;

        setFromToml(s.debug_dump, debug, "DebugDump");
        setFromToml(s.shader_collect, debug, "CollectShader");
        setFromToml(m_general.show_fps_counter, debug, "showFpsCounter");
    }

    if (og_data.contains("Settings")) {
        const toml::value& settings = og_data.at("Settings");
        auto& s = m_general;
        setFromToml(s.console_language, settings, "consoleLanguage");
    }

    if (og_data.contains("GUI")) {
        const toml::value& gui = og_data.at("GUI");
        auto& s = m_general;

        // Transfer install directories
        try {
            const auto install_dir_array =
                toml::find_or<std::vector<std::string>>(gui, "installDirs", {});
            std::vector<bool> install_dirs_enabled;

            try {
                install_dirs_enabled = toml::find<std::vector<bool>>(gui, "installDirsEnabled");
            } catch (...) {
                // If it does not exist, assume that all are enabled.
                install_dirs_enabled.resize(install_dir_array.size(), true);
            }

            if (install_dirs_enabled.size() < install_dir_array.size()) {
                install_dirs_enabled.resize(install_dir_array.size(), true);
            }

            std::vector<GameInstallDir> settings_install_dirs;
            for (size_t i = 0; i < install_dir_array.size(); i++) {
                settings_install_dirs.push_back(
                    {std::filesystem::path{install_dir_array[i]}, install_dirs_enabled[i]});
            }
            s.install_dirs.value = settings_install_dirs;
        } catch (const std::exception& e) {
            LOG_WARNING(Config, "Failed to transfer install directories: {}", e.what());
        }

        // Transfer addon install directory
        try {
            std::string addon_install_dir_str;
            if (gui.contains("addonInstallDir")) {
                const auto& addon_value = gui.at("addonInstallDir");
                if (addon_value.is_string()) {
                    addon_install_dir_str = toml::get<std::string>(addon_value);
                    if (!addon_install_dir_str.empty()) {
                        s.addon_install_dir.value = std::filesystem::path{addon_install_dir_str};
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_WARNING(Config, "Failed to transfer addon install directory: {}", e.what());
        }
    }
    if (og_data.contains("General")) {
        const toml::value& general = og_data.at("General");
        auto& s = m_general;
        // Transfer sysmodules install directory
        try {
            std::string sysmodules_install_dir_str;
            if (general.contains("sysModulesPath")) {
                const auto& sysmodule_value = general.at("sysModulesPath");
                if (sysmodule_value.is_string()) {
                    sysmodules_install_dir_str = toml::get<std::string>(sysmodule_value);
                    if (!sysmodules_install_dir_str.empty()) {
                        s.sys_modules_dir.value = std::filesystem::path{sysmodules_install_dir_str};
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_WARNING(Config, "Failed to transfer sysmodules install directory: {}", e.what());
        }

        // Transfer font install directory
        try {
            std::string font_install_dir_str;
            if (general.contains("fontsPath")) {
                const auto& font_value = general.at("fontsPath");
                if (font_value.is_string()) {
                    font_install_dir_str = toml::get<std::string>(font_value);
                    if (!font_install_dir_str.empty()) {
                        s.font_dir.value = std::filesystem::path{font_install_dir_str};
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_WARNING(Config, "Failed to transfer font install directory: {}", e.what());
        }
    }

    return true;
}

std::vector<std::string> EmulatorSettingsImpl::GetAllOverrideableKeys() const {
    std::vector<std::string> keys;
    auto addGroup = [&keys](const auto& fields) {
        for (const auto& item : fields)
            keys.push_back(item.key);
    };
    addGroup(m_general.GetOverrideableFields());
    addGroup(m_log.GetOverrideableFields());
    addGroup(m_debug.GetOverrideableFields());
    addGroup(m_input.GetOverrideableFields());
    addGroup(m_audio.GetOverrideableFields());
    // Windows static guest red-zone protection
    addGroup(m_windows_guest_red_zone_protection.GetOverrideableFields());
    addGroup(m_gpu.GetOverrideableFields());
    addGroup(m_vulkan.GetOverrideableFields());
    return keys;
}
