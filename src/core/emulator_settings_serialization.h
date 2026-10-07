// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Serialization and override construction are private to settings persistence.
#include <exception>
#include <functional>
#include <sstream>
#include <string_view>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "core/emulator_settings.h"

// ── nlohmann helpers for std::filesystem::path ───────────────────────
namespace nlohmann {
template <>
struct adl_serializer<std::filesystem::path> {
    static void to_json(json& j, const std::filesystem::path& p) {
        const auto u8 = p.u8string();
        j = std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
    }
    static void from_json(const json& j, std::filesystem::path& p) {
        const std::string s = j.get<std::string>();
        p = std::filesystem::path(
            std::u8string_view(reinterpret_cast<const char8_t*>(s.data()), s.size()));
    }
};
} // namespace nlohmann

template <typename T>
void to_json(nlohmann::json& j, const Setting<T>& s) {
    j = s.value;
}

template <typename T>
void from_json(const nlohmann::json& j, Setting<T>& s) {
    s.value = j.get<T>();
}

struct OverrideItem {
    const char* key;
    std::function<void(void* group_ptr, const nlohmann::json& entry,
                       std::vector<std::string>& changed)>
        apply;
    /// Return the value that should be written to the per-game config file.
    /// Falls back to base value if no game-specific override is set.
    std::function<nlohmann::json(const void* group_ptr)> get_for_save;

    /// Clear game_specific_value for this field.
    std::function<void(void* group_ptr)> reset_game_specific;
};

template <typename Struct, typename T>
inline OverrideItem make_override(const char* key, Setting<T> Struct::* member) {
    return OverrideItem{
        key,
        [member, key](void* base, const nlohmann::json& entry, std::vector<std::string>& changed) {
            LOG_DEBUG(Config, "[make_override] Processing key: {}", key);
            LOG_DEBUG(Config, "[make_override] Entry JSON: {}", entry.dump());
            Struct* obj = reinterpret_cast<Struct*>(base);
            Setting<T>& dst = obj->*member;
            try {
                T newValue = entry.get<T>();
                LOG_DEBUG(Config, "[make_override] Parsed value: {}", newValue);
                LOG_DEBUG(Config, "[make_override] Current value: {}", dst.value);
                if (dst.value != newValue) {
                    std::ostringstream oss;
                    oss << key << " ( " << dst.value << " → " << newValue << " )";
                    changed.push_back(oss.str());
                    LOG_DEBUG(Config, "[make_override] Recorded change: {}", oss.str());
                }
                dst.game_specific_value = newValue;
                LOG_DEBUG(Config, "[make_override] Successfully updated {}", key);
            } catch (const std::exception& e) {
                LOG_ERROR(Config, "[make_override] ERROR parsing {}: {}", key, e.what());
                LOG_ERROR(Config, "[make_override] Entry was: {}", entry.dump());
                LOG_ERROR(Config, "[make_override] Type name: {}", entry.type_name());
            }
        },

        // --- get_for_save -------------------------------------------
        // Returns game_specific_value when present, otherwise base value.
        // This means a freshly-opened game-specific dialog still shows
        // useful (current-global) values rather than empty entries.
        [member](const void* base) -> nlohmann::json {
            const Struct* obj = reinterpret_cast<const Struct*>(base);
            const Setting<T>& src = obj->*member;
            return nlohmann::json(src.game_specific_value.value_or(src.value));
        },

        // --- reset_game_specific ------------------------------------
        [member](void* base) {
            Struct* obj = reinterpret_cast<Struct*>(base);
            (obj->*member).reset_game_specific();
        }};
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GameInstallDir, path, enabled)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GeneralSettings, install_dirs, addon_install_dir, home_dir,
                                   sys_modules_dir, font_dir, volume_slider, neo_mode, dev_kit_mode,
                                   extra_dmem_in_mbytes, shad_net_enabled, trophy_popup_disabled,
                                   trophy_notification_duration, show_splash,
                                   trophy_notification_side, connected_to_network,
                                   discord_rpc_enabled, show_fps_counter, console_language,
                                   big_picture_scale, shadnet_server)
#ifdef _WIN32
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LogSettings, append, enable, filter, max_skip_duration, separate,
                                   size_limit, skip_duplicate, sync, type)
#else
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LogSettings, append, enable, filter, max_skip_duration, separate,
                                   size_limit, skip_duplicate, sync)
#endif
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DebugSettings, debug_dump, shader_collect, config_version)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InputSettings, cursor_state, cursor_hide_timeout,
                                   usb_device_backend, use_special_pad, special_pad_class,
                                   motion_controls_enabled, use_unified_input_config,
                                   default_controller_id, background_controller_input,
                                   ime_accessibility_enabled, ime_url_mail_short_panel, camera_id,
                                   is_circle_enter)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AudioSettings, audio_backend, sdl_mic_device,
                                   sdl_main_output_device, sdl_padSpk_output_device,
                                   openal_mic_device, openal_main_output_device,
                                   openal_padSpk_output_device)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GPUSettings, window_width, window_height, internal_screen_width,
                                   internal_screen_height, null_gpu, copy_gpu_buffers,
                                   readbacks_mode, readback_linear_images_enabled,
                                   direct_memory_access_enabled, dump_shaders, patch_shaders,
                                   vblank_frequency, full_screen, full_screen_mode, present_mode,
                                   hdr_allowed, fsr_enabled, rcas_enabled, rcas_attenuation)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VulkanSettings, gpu_id, renderdoc_enabled, vkvalidation_enabled,
                                   vkvalidation_core_enabled, vkvalidation_sync_enabled,
                                   vkvalidation_gpu_enabled, vkcrash_diagnostic_enabled,
                                   vkhost_markers, vkguest_markers, pipeline_cache_enabled,
                                   pipeline_cache_archived)

namespace SettingsSerialization {
/// Apply overrideable fields from groupJson into group.game_specific_value.
template <typename Group>
void ApplyGroupOverrides(Group& group, const nlohmann::json& groupJson,
                         std::vector<std::string>& changed) {
    for (auto& item : group.GetOverrideableFields()) {
        if (!groupJson.contains(item.key))
            continue;
        item.apply(&group, groupJson.at(item.key), changed);
    }
}

// Write all overrideable fields from group into out (for game-specific save).
template <typename Group>
static void SaveGroupGameSpecific(const Group& group, nlohmann::json& out) {
    for (auto& item : group.GetOverrideableFields())
        out[item.key] = item.get_for_save(&group);
}

// Discard every game-specific override in group.
template <typename Group>
static void ClearGroupOverrides(Group& group) {
    for (auto& item : group.GetOverrideableFields())
        item.reset_game_specific(&group);
}

} // namespace SettingsSerialization
