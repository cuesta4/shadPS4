// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <xxhash.h>

#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/texture_cache/sampler_repository.h"

namespace VideoCore {

SamplerRepository::SamplerRepository(const Vulkan::Instance& instance_) : instance{instance_} {
    const u64 maximum = instance.GetMaxSamplerAllocationCount();
    trigger_gc = maximum * 3 / 4;
    pressure_gc = maximum * 7 / 8;
    critical_gc = maximum * 15 / 16;
}

vk::Sampler SamplerRepository::Get(const AmdGpu::Sampler& sampler,
                                   AmdGpu::BorderColorBuffer border_color_base, u64 tick) {
    const u64 hash = XXH3_64bits(&sampler, sizeof(sampler));
    std::scoped_lock lock{mutex};
    const auto [it, inserted] = samplers.try_emplace(hash, instance, sampler, border_color_base);
    if (inserted) {
        it.value().lru_id = lru_cache.Insert(hash, tick);
    } else {
        lru_cache.Touch(it->second.lru_id, tick);
    }
    return it->second.Handle();
}

void SamplerRepository::Collect(u64 tick) {
    const u64 used = samplers.size();
    if (used < trigger_gc) {
        return;
    }

    std::scoped_lock lock{mutex};
    bool pressured{};
    bool aggressive{};
    u64 ticks_to_destroy{};
    size_t deletions{};

    const auto configure = [&](bool allow_aggressive) {
        pressured = used >= pressure_gc;
        aggressive = allow_aggressive && used >= critical_gc;
        ticks_to_destroy = aggressive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, tick);
        deletions = aggressive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](u64 hash) {
        if (deletions == 0) {
            return true;
        }
        --deletions;
        const auto it = samplers.find(hash);
        if (it != samplers.end()) {
            const size_t lru_id = it->second.lru_id;
            samplers.erase(it);
            lru_cache.Free(lru_id);
        }
        return false;
    };

    configure(false);
    lru_cache.ForEachItemBelow(tick - ticks_to_destroy, clean_up);
    if (used >= critical_gc) {
        configure(true);
        lru_cache.ForEachItemBelow(tick - ticks_to_destroy, clean_up);
    }
}

} // namespace VideoCore
