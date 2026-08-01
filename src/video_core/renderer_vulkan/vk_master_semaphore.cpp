// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>

#include "common/performance_telemetry.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

#include "common/assert.h"

namespace Vulkan {

constexpr u64 WAIT_TIMEOUT = std::numeric_limits<u64>::max();

MasterSemaphore::MasterSemaphore(const Instance& instance_) : instance{instance_} {
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);
}

MasterSemaphore::~MasterSemaphore() = default;

void MasterSemaphore::TelemetrySubmit(u64 tick) {
    if (!Common::PerformanceTelemetry::Enabled()) [[likely]] {
        return;
    }

    u64 now{};
    u64 idle_since{};
    {
        std::scoped_lock lock{telemetry_mutex};
        now = Common::PerformanceTelemetry::Timestamp();
        if (telemetry_idle_since_ns != 0 &&
            telemetry_completed_tick >= telemetry_submitted_tick) {
            idle_since = telemetry_idle_since_ns;
        }
        telemetry_submitted_tick = std::max(telemetry_submitted_tick, tick);
        telemetry_idle_since_ns = 0;
    }

    Common::PerformanceTelemetry::AddEnabled(
        Common::PerformanceTelemetry::Counter::DriverSubmitCalls, 1);
    Common::PerformanceTelemetry::RecordEnabled(
        Common::PerformanceTelemetry::EventType::DriverSubmit,
        reinterpret_cast<uintptr_t>(this), tick);
    if (idle_since != 0) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::GpuIdleGaps, 1);
        Common::PerformanceTelemetry::RecordDurationValueEnabled(
            Common::PerformanceTelemetry::Counter::GpuIdleGapNs,
            Common::PerformanceTelemetry::EventType::GpuIdleGap, now - idle_since,
            reinterpret_cast<uintptr_t>(this));
    }
}

void MasterSemaphore::TelemetryComplete(u64 tick) {
    if (!Common::PerformanceTelemetry::Enabled()) [[likely]] {
        return;
    }

    {
        std::scoped_lock lock{telemetry_mutex};
        const u64 now = Common::PerformanceTelemetry::Timestamp();
        telemetry_completed_tick = std::max(telemetry_completed_tick, tick);
        if (telemetry_submitted_tick != 0 &&
            telemetry_completed_tick >= telemetry_submitted_tick && telemetry_idle_since_ns == 0) {
            telemetry_idle_since_ns = now;
        }
    }
    Common::PerformanceTelemetry::RecordEnabled(
        Common::PerformanceTelemetry::EventType::TimelineComplete,
        reinterpret_cast<uintptr_t>(this), tick);
}

void MasterSemaphore::Refresh() {
    Common::PerformanceTelemetry::ScopedDuration poll_duration{
        Common::PerformanceTelemetry::Counter::TimelinePollNs};
    Common::PerformanceTelemetry::Add(
        Common::PerformanceTelemetry::Counter::TimelinePolls);
    const u64 previous_tick = gpu_tick.load(std::memory_order_relaxed);
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
    if (counter > previous_tick) {
        TelemetryComplete(counter);
    }
}

void MasterSemaphore::Wait(u64 tick) {
    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        return;
    }
    // Update the GPU tick and try again
    Refresh();
    if (IsFree(tick)) {
        return;
    }

    // If none of the above is hit, fallback to a regular wait
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::WaitCalls);
    Common::PerformanceTelemetry::ScopedDuration wait_duration{
        Common::PerformanceTelemetry::Counter::WaitNs,
        Common::PerformanceTelemetry::EventType::Wait, tick};
    while (instance.GetDevice().waitSemaphores(&wait_info, WAIT_TIMEOUT) != vk::Result::eSuccess) {
    }
    Refresh();
}

} // namespace Vulkan
