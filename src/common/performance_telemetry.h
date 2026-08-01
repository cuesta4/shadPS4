// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <filesystem>

#include "common/logging/log.h"
#include "common/types.h"

namespace Common::PerformanceTelemetry {

enum class Counter : u16 {
    Pm4Packets,
    Pm4Type2Packets,
    DcbBytes,
    CcbBytes,
    AcbBytes,
    GfxSubmits,
    AscSubmits,
    GcpWakes,
    QueueScans,
    QueueResumes,
    GcpActiveNs,
    GcpBlockedNs,
    QueueReadyNs,
    QueueResumeNs,
    IbDepthMax,
    Draws,
    Dispatches,
    DrawCpuNs,
    DispatchCpuNs,
    PipelineHits,
    PipelineMisses,
    PipelineCompileNs,
    RenderTargetHits,
    RenderTargetMisses,
    ImageTokenHits,
    ImageTokenMisses,
    DescriptorHits,
    DescriptorMisses,
    BufferTokenHits,
    BufferTokenMisses,
    StreamSliceHits,
    StreamSliceMisses,
    StagingBytes,
    BarrierCalls,
    CopyCalls,
    CopyBytes,
    TimelinePolls,
    TimelinePollNs,
    DriverSubmitCalls,
    DriverSubmitNs,
    DriverPresentCalls,
    DriverPresentNs,
    SubmitQueueDepthMax,
    WaitCalls,
    WaitNs,
    WritebackCalls,
    WritebackBytes,
    WritebackNs,
    PresentPrepareNs,
    PresentCpuNs,
    GpuIdleGaps,
    GpuIdleGapNs,
    Count,
};

enum class EventType : u16 {
    None,
    GcpActive,
    GcpBlocked,
    SubmitDone,
    TimelineComplete,
    DriverSubmit,
    Wait,
    Writeback,
    PresentPrepare,
    PresentCpu,
    DriverPresent,
    FramePresented,
    GpuIdleGap,
    Pm4Packet,
};

enum class Pm4Engine : u8 {
    Constant,
    Graphics,
    Compute,
};

[[nodiscard]] inline bool Enabled() noexcept {
    return Log::IsEnabled();
}

[[nodiscard]] inline u64 Timestamp() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

void AddEnabled(Counter counter, u64 value) noexcept;
void ObserveMaxEnabled(Counter counter, u64 value) noexcept;
void RecordEnabled(EventType type, u64 arg0, u64 arg1) noexcept;
void RecordDurationEnabled(Counter counter, EventType type, u64 start_ns, u64 arg0) noexcept;
void RecordDurationValueEnabled(Counter counter, EventType type, u64 duration, u64 arg0) noexcept;
void CountPm4PacketEnabled(Pm4Engine engine, u32 queue_id, u32 opcode, u32 depth,
                           uintptr_t address, u32 words) noexcept;

inline void Add(Counter counter, u64 value = 1) noexcept {
    if (Enabled()) [[unlikely]] {
        AddEnabled(counter, value);
    }
}

inline void ObserveMax(Counter counter, u64 value) noexcept {
    if (Enabled()) [[unlikely]] {
        ObserveMaxEnabled(counter, value);
    }
}

inline void Record(EventType type, u64 arg0 = 0, u64 arg1 = 0) noexcept {
    if (Enabled()) [[unlikely]] {
        RecordEnabled(type, arg0, arg1);
    }
}

inline void CountPm4Packet(Pm4Engine engine, u32 queue_id, u32 opcode, u32 depth,
                           uintptr_t address, u32 words) noexcept {
    if (Enabled()) [[unlikely]] {
        CountPm4PacketEnabled(engine, queue_id, opcode, depth, address, words);
    }
}

class ScopedDuration {
public:
    explicit ScopedDuration(Counter counter_, EventType type_ = EventType::None,
                            u64 arg0_ = 0) noexcept
        : counter{counter_}, type{type_}, arg0{arg0_}, start_ns{Enabled() ? Timestamp() : 0} {}

    ~ScopedDuration() {
        if (start_ns != 0) [[unlikely]] {
            RecordDurationEnabled(counter, type, start_ns, arg0);
        }
    }

private:
    Counter counter;
    EventType type;
    u64 arg0;
    u64 start_ns;
};

/// Writes the in-memory snapshot on a normal GUI shutdown. Returns an empty path when logging is
/// disabled or no telemetry was collected.
[[nodiscard]] std::filesystem::path Dump();

} // namespace Common::PerformanceTelemetry
