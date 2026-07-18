// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/process_memory.h"

#include <algorithm>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "common/error.h"
#include "common/logging/log.h"

namespace Common {

namespace {

constexpr bool IsValidLimit(const u32 limit_mib) {
    return limit_mib == 0 || limit_mib == 10 * 1024 || limit_mib == 12 * 1024 ||
           limit_mib == 16 * 1024;
}

#ifdef _WIN32
struct WorkingSetPolicy {
    SIZE_T minimum{};
    SIZE_T maximum{};
    DWORD flags{};
};

std::mutex g_policy_mutex;
WorkingSetPolicy g_original_policy{};
bool g_original_policy_saved{};
bool g_artificial_limit_active{};

bool QueryPolicy(const HANDLE process, WorkingSetPolicy& policy) {
    return GetProcessWorkingSetSizeEx(process, &policy.minimum, &policy.maximum, &policy.flags);
}

bool RestorePolicy(const HANDLE process) {
    DWORD flags = g_original_policy.flags &
                  ~(QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
    if ((flags & QUOTA_LIMITS_HARDWS_MIN_ENABLE) == 0) {
        flags |= QUOTA_LIMITS_HARDWS_MIN_DISABLE;
    }
    if ((flags & QUOTA_LIMITS_HARDWS_MAX_ENABLE) == 0) {
        flags |= QUOTA_LIMITS_HARDWS_MAX_DISABLE;
    }
    return SetProcessWorkingSetSizeEx(process, g_original_policy.minimum, g_original_policy.maximum,
                                      flags);
}
#endif

} // Anonymous namespace

void ConfigureProcessWorkingSetLimit(u32 limit_mib) {
    if (!IsValidLimit(limit_mib)) {
        LOG_WARNING(Config,
                    "Ignoring unsupported artificial RAM limit {} MiB; expected 0, 10240, "
                    "12288, or 16384",
                    limit_mib);
        limit_mib = 0;
    }

#ifdef _WIN32
    const std::scoped_lock lock{g_policy_mutex};
    const HANDLE process = GetCurrentProcess();

    if (limit_mib == 0) {
        if (!g_artificial_limit_active) {
            return;
        }
        if (!RestorePolicy(process)) {
            LOG_WARNING(Core, "Unable to restore the Windows working-set policy: {}",
                        GetLastErrorMsg());
            return;
        }
        g_artificial_limit_active = false;
        LOG_INFO(Core, "Restored the original Windows working-set policy");
        return;
    }

    if (!g_original_policy_saved && !QueryPolicy(process, g_original_policy)) {
        LOG_WARNING(Core, "Unable to query the Windows working-set policy: {}", GetLastErrorMsg());
        return;
    }
    g_original_policy_saved = true;

    const SIZE_T maximum = static_cast<SIZE_T>(limit_mib) * 1_MB;
    const SIZE_T minimum = std::min(g_original_policy.minimum, maximum);
    constexpr DWORD Flags = QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE;
    if (!SetProcessWorkingSetSizeEx(process, minimum, maximum, Flags)) {
        LOG_WARNING(Core,
                    "Unable to enforce the artificial RAM limit as a Windows working-set "
                    "maximum (requested={} MiB): {}",
                    limit_mib, GetLastErrorMsg());
        return;
    }
    g_artificial_limit_active = true;

    WorkingSetPolicy effective_policy{};
    if (!QueryPolicy(process, effective_policy)) {
        LOG_WARNING(Core,
                    "Artificial RAM limit was accepted, but its Windows working-set policy "
                    "could not be verified: {}",
                    GetLastErrorMsg());
        return;
    }

    const bool hard_maximum = (effective_policy.flags & QUOTA_LIMITS_HARDWS_MAX_ENABLE) != 0;
    if (effective_policy.maximum != maximum || !hard_maximum) {
        LOG_WARNING(Core,
                    "Artificial RAM limit was not fully enforced: requested={} MiB "
                    "effective={} MiB hard_maximum={}",
                    limit_mib, effective_policy.maximum / 1_MB, hard_maximum);
        return;
    }

    LOG_INFO(Core, "Artificial RAM limit enforced as a Windows hard working-set maximum: {} MiB",
             limit_mib);
#else
    if (limit_mib != 0) {
        LOG_WARNING(Core,
                    "Artificial RAM limit {} MiB currently has no resident-memory enforcement "
                    "on this platform",
                    limit_mib);
    }
#endif
}

} // namespace Common
