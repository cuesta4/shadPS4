// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <tsl/robin_map.h>

#include "common/lru_cache.h"
#include "common/types.h"
#include "video_core/texture_cache/sampler.h"

namespace Vulkan {
class Instance;
}

namespace VideoCore {

class SamplerRepository {
public:
    explicit SamplerRepository(const Vulkan::Instance& instance);

    [[nodiscard]] vk::Sampler Get(const AmdGpu::Sampler& sampler,
                                  AmdGpu::BorderColorBuffer border_color_base, u64 tick);
    void Collect(u64 tick);

private:
    const Vulkan::Instance& instance;
    tsl::robin_map<u64, Sampler> samplers;
    Common::LeastRecentlyUsedCache<u64, u64> lru_cache;
    std::mutex mutex;
    u64 trigger_gc{};
    u64 pressure_gc{};
    u64 critical_gc{};
};

} // namespace VideoCore
