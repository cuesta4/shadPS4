// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_gpu_profiler.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/performance_telemetry.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

namespace Vulkan {

#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
namespace {

using namespace Common::PerformanceTelemetry;

constexpr u32 QuerySlotCount = 64;
constexpr u32 DefaultTimestampSamplePeriod = 4;
constexpr u32 DefaultStatisticSamplePeriod = 32;
constexpr u32 DefaultTimestampQueryBudget = 384;
constexpr u32 DefaultStatisticQueryBudget = 64;
constexpr u32 MaxTimestampQueryBudget = 2048;
constexpr u32 MaxStatisticQueryBudget = 256;

u32 EnvironmentU32(const char* name, u32 fallback, u32 minimum, u32 maximum) {
    const char* value = std::getenv(name);
    if (!value || *value == '\0') {
        return fallback;
    }
    u32 parsed{};
    const auto [end, ec] = std::from_chars(value, value + std::char_traits<char>::length(value),
                                           parsed);
    return ec == std::errc{} && *end == '\0' ? std::clamp(parsed, minimum, maximum) : fallback;
}

u64 HashText(const char* text) noexcept {
    constexpr u64 OffsetBasis = 14695981039346656037ULL;
    constexpr u64 Prime = 1099511628211ULL;
    u64 hash = OffsetBasis;
    for (; *text != '\0'; ++text) {
        hash = (hash ^ static_cast<u8>(*text)) * Prime;
    }
    return hash;
}

u64 StatisticValueBits(const vk::PipelineExecutableStatisticKHR& statistic) noexcept {
    switch (statistic.format) {
    case vk::PipelineExecutableStatisticFormatKHR::eBool32:
        return statistic.value.b32;
    case vk::PipelineExecutableStatisticFormatKHR::eInt64:
        return std::bit_cast<u64>(statistic.value.i64);
    case vk::PipelineExecutableStatisticFormatKHR::eUint64:
        return statistic.value.u64;
    case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
        return std::bit_cast<u64>(statistic.value.f64);
    default:
        return 0;
    }
}

vk::PipelineStageFlags2 BeginStage(GpuIntervalKind kind) {
    switch (kind) {
    case GpuIntervalKind::RenderingScope:
    case GpuIntervalKind::GraphicsPipelineBlock:
        return vk::PipelineStageFlagBits2::eAllGraphics;
    case GpuIntervalKind::ComputePipelineBlock:
        return vk::PipelineStageFlagBits2::eComputeShader;
    case GpuIntervalKind::Tile:
    case GpuIntervalKind::Detile:
        return vk::PipelineStageFlagBits2::eComputeShader;
    case GpuIntervalKind::Copy:
    case GpuIntervalKind::Resolve:
    case GpuIntervalKind::Clear:
        return vk::PipelineStageFlagBits2::eTransfer;
    default:
        return vk::PipelineStageFlagBits2::eAllCommands;
    }
}

vk::PipelineStageFlags2 EndStage(GpuIntervalKind kind) {
    return kind == GpuIntervalKind::CommandBuffer ? vk::PipelineStageFlagBits2::eBottomOfPipe
                                                   : BeginStage(kind);
}

constexpr vk::QueryPipelineStatisticFlags GraphicsStatisticFlags =
    vk::QueryPipelineStatisticFlagBits::eInputAssemblyVertices |
    vk::QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives |
    vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations |
    vk::QueryPipelineStatisticFlagBits::eClippingInvocations |
    vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
    vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations;
constexpr vk::QueryPipelineStatisticFlags ComputeStatisticFlags =
    vk::QueryPipelineStatisticFlagBits::eComputeShaderInvocations;
constexpr size_t GraphicsStatisticCount = 6;
constexpr size_t ComputeStatisticCount = 1;

} // namespace

struct GpuProfiler::Impl {
    struct Interval {
        GpuIntervalSample sample{};
        size_t parent_index{std::numeric_limits<size_t>::max()};
        u32 begin_query{};
        u32 end_query{};
        u32 statistic_query{};
        bool statistic_active{};
    };

    struct Slot {
        vk::UniqueQueryPool timestamps;
        vk::UniqueQueryPool graphics_statistics;
        vk::UniqueQueryPool compute_statistics;
        std::vector<Interval> intervals;
        std::vector<size_t> open_intervals;
        QueryFrameSeq query_frame_id{};
        SubmitSeq submit_seq{};
        u64 retire_tick{};
        u32 timestamp_queries{};
        u32 graphics_statistic_queries{};
        u32 compute_statistic_queries{};
        bool statistic_sample{};
        bool detailed_sample{};
        bool pending{};
        u64 command_buffer_ordinal{};
    };

    Impl(const Instance& instance_, MasterSemaphore& master_semaphore_)
        : instance{instance_}, master_semaphore{master_semaphore_}, device{instance.GetDevice()},
          timestamp_sample_period{EnvironmentU32("SHADPS4_GPU_TELEMETRY_SAMPLE_PERIOD",
                                                 DefaultTimestampSamplePeriod, 1, 1024)},
          statistic_sample_period{EnvironmentU32("SHADPS4_GPU_TELEMETRY_STATS_PERIOD",
                                                 DefaultStatisticSamplePeriod, 1, 4096)},
          timestamp_query_budget{EnvironmentU32("SHADPS4_GPU_TELEMETRY_QUERY_BUDGET",
                                                DefaultTimestampQueryBudget, 16,
                                                MaxTimestampQueryBudget) &
                                 ~u32{1}},
          statistic_query_budget{EnvironmentU32("SHADPS4_GPU_TELEMETRY_STATS_BUDGET",
                                                DefaultStatisticQueryBudget, 1,
                                                MaxStatisticQueryBudget)},
          timestamps_supported{instance.TimestampValidBits() != 0},
          statistics_supported{instance.SupportsPipelineStatistics()} {
        if (!timestamps_supported) {
            return;
        }
        for (auto& slot : slots) {
            slot.timestamps = Check<"create telemetry timestamp query pool">(
                device.createQueryPoolUnique(vk::QueryPoolCreateInfo{
                    .queryType = vk::QueryType::eTimestamp,
                    .queryCount = timestamp_query_budget,
                }));
            if (statistics_supported) {
                slot.graphics_statistics = Check<"create telemetry graphics query pool">(
                    device.createQueryPoolUnique(vk::QueryPoolCreateInfo{
                        .queryType = vk::QueryType::ePipelineStatistics,
                        .queryCount = statistic_query_budget,
                        .pipelineStatistics = GraphicsStatisticFlags,
                    }));
                slot.compute_statistics = Check<"create telemetry compute query pool">(
                    device.createQueryPoolUnique(vk::QueryPoolCreateInfo{
                        .queryType = vk::QueryType::ePipelineStatistics,
                        .queryCount = statistic_query_budget,
                        .pipelineStatistics = ComputeStatisticFlags,
                    }));
            }
            slot.intervals.reserve(timestamp_query_budget / 2);
            slot.open_intervals.reserve(8);
        }
    }

    ~Impl() {
        Collect();
        RecordHealth();
    }

    void RecordHealth() {
        RecordGpuProfilerHealth(GpuProfilerHealthSample{
            .scheduler_id = 0,
            .timestamp_sample_period = timestamp_sample_period,
            .statistic_sample_period = statistic_sample_period,
            .timestamp_query_budget = timestamp_query_budget,
            .statistic_query_budget = statistic_query_budget,
            .command_buffers_seen = command_buffers_seen,
            .command_buffers_sampled = command_buffers_sampled,
            .command_buffers_detailed = command_buffers_detailed,
            .slots_unavailable = slots_unavailable,
            .intervals_seen = intervals_seen,
            .intervals_recorded = intervals_recorded,
            .intervals_filtered = intervals_filtered,
            .intervals_budget_dropped = intervals_budget_dropped,
            .query_results_available = query_results_available,
            .query_results_not_ready = query_results_not_ready,
            .timestamp_queries_written = timestamp_queries_written,
            .statistic_queries_written = statistic_queries_written,
            .query_collect_calls = query_collect_calls,
            .query_collect_cost_samples = query_collect_cost_samples,
            .query_collect_ready_slots = query_collect_ready_slots,
            .query_collect_sampled_ns = query_collect_sampled_ns,
            .calibration_calls = calibration_calls,
            .calibration_cpu_ns = calibration_cpu_ns,
            .timestamps_supported = static_cast<u8>(timestamps_supported),
            .pipeline_statistics_supported = static_cast<u8>(statistics_supported),
            .calibrated_timestamps_supported =
                static_cast<u8>(instance.SupportsCalibratedTimestamps()),
            .pipeline_executable_supported =
                static_cast<u8>(instance.SupportsPipelineExecutableProperties()),
            .pipeline_executable_capture_enabled =
                static_cast<u8>(PipelineExecutableCaptureEnabled()),
        });
    }

    void Calibrate(QueryFrameSeq query_frame_id) {
        if (!instance.SupportsCalibratedTimestamps() || (command_buffers_sampled & 63) != 1) {
            return;
        }
        const std::array infos{
            VkCalibratedTimestampInfoKHR{
                .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
                .pNext = nullptr,
                .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR,
            },
            VkCalibratedTimestampInfoKHR{
                .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
                .pNext = nullptr,
                .timeDomain = static_cast<VkTimeDomainKHR>(instance.CalibratedHostTimeDomain()),
            },
        };
        std::array<u64, 2> values{};
        u64 max_deviation{};
        const u64 calibration_start = Timestamp();
        const VkResult result = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetCalibratedTimestampsEXT(
            static_cast<VkDevice>(device), static_cast<u32>(infos.size()), infos.data(),
            values.data(), &max_deviation);
        ++calibration_calls;
        calibration_cpu_ns += Timestamp() - calibration_start;
        RecordGpuCalibration(GpuCalibrationSample{
            .query_frame_id = query_frame_id,
            .device_timestamp = values[0],
            .host_timestamp = values[1],
            .host_steady_timestamp_ns = Timestamp(),
            .max_deviation = max_deviation,
            .timestamp_period_ns = instance.TimestampPeriodNs(),
            .host_time_domain = static_cast<u32>(instance.CalibratedHostTimeDomain()),
            .timestamp_valid_bits = instance.TimestampValidBits(),
            .success = static_cast<u8>(result == VK_SUCCESS),
        });
    }

    void BeginCommandBuffer(vk::CommandBuffer cmdbuf_, CmdBufferSeq command_buffer_seq,
                            FrameSeq frame_seq) {
        Collect();
        ++command_buffers_seen;
        cmdbuf = cmdbuf_;
        if (!timestamps_supported) {
            return;
        }

        Slot& selected = slots[next_slot++ % slots.size()];
        if (selected.pending) {
            ++slots_unavailable;
            return;
        }
        active = &selected;
        active->intervals.clear();
        active->open_intervals.clear();
        active->timestamp_queries = 0;
        active->graphics_statistic_queries = 0;
        active->compute_statistic_queries = 0;
        active->submit_seq = 0;
        active->retire_tick = 0;
        active->query_frame_id = NextQueryFrameSeq();
        active->command_buffer_ordinal = command_buffers_seen;
        active->detailed_sample = command_buffers_seen % timestamp_sample_period == 0;
        active->statistic_sample = active->detailed_sample && statistics_supported &&
                                   command_buffers_seen % statistic_sample_period == 0;
        cmdbuf.resetQueryPool(active->timestamps.get(), 0, timestamp_query_budget);
        if (active->statistic_sample) {
            cmdbuf.resetQueryPool(active->graphics_statistics.get(), 0, statistic_query_budget);
            cmdbuf.resetQueryPool(active->compute_statistics.get(), 0, statistic_query_budget);
        }
        ++command_buffers_sampled;
        command_buffers_detailed += active->detailed_sample;
        Calibrate(active->query_frame_id);
        BeginInterval(GpuIntervalKind::CommandBuffer, 0, 0, command_buffer_seq, frame_seq);
    }

    void EndCommandBuffer(SubmitSeq submit_seq, u64 retire_tick) {
        if (!active) {
            return;
        }
        ClosePipelineBlock();
        EndRendering();
        while (!active->open_intervals.empty()) {
            CloseInterval(active->open_intervals.back());
        }
        active->submit_seq = submit_seq;
        active->retire_tick = retire_tick;
        for (auto& interval : active->intervals) {
            interval.sample.submit_seq = submit_seq;
        }
        active->pending = true;
        active = nullptr;
        cmdbuf = nullptr;
    }

    u64 BeginInterval(GpuIntervalKind kind, u64 object_hash, u64 bytes,
                      CmdBufferSeq command_buffer_seq = 0, FrameSeq frame_seq = 0) {
        ++intervals_seen;
        if (active && kind != GpuIntervalKind::CommandBuffer && !active->detailed_sample) {
            ++intervals_filtered;
            return 0;
        }
        if (!active || active->timestamp_queries + 2 > timestamp_query_budget) {
            ++intervals_budget_dropped;
            return 0;
        }
        const auto context = CurrentCausalContext();
        const u64 interval_id = NextGpuIntervalSeq();
        const u32 begin_query = active->timestamp_queries++;
        const u32 end_query = active->timestamp_queries++;
        const u64 parent_id = active->open_intervals.empty()
                                  ? 0
                                  : active->intervals[active->open_intervals.back()]
                                        .sample.interval_id;
        const size_t parent_index = active->open_intervals.empty()
                                        ? std::numeric_limits<size_t>::max()
                                        : active->open_intervals.back();
        active->intervals.push_back(Interval{
            .sample = GpuIntervalSample{
                .interval_id = interval_id,
                .parent_interval_id = parent_id,
                .query_frame_id = active->query_frame_id,
                .frame_seq = frame_seq ? frame_seq : CurrentFrameSeq(),
                .command_buffer_seq =
                    command_buffer_seq ? command_buffer_seq : CurrentCmdBufferSeq(),
                .cause_id = context.cause_id,
                .candidate_id = context.candidate_id,
                .scope_id = context.scope_id,
                .object_hash = object_hash,
                .pipeline_hash = kind == GpuIntervalKind::GraphicsPipelineBlock ||
                                         kind == GpuIntervalKind::ComputePipelineBlock
                                     ? object_hash
                                     : 0,
                .attachment_hash = attachment_hash,
                .bytes = bytes,
                .kind = kind,
                .status = GpuQueryStatus::Invalid,
                .attribution = kind == GpuIntervalKind::CommandBuffer ||
                                       kind == GpuIntervalKind::RenderingScope
                                   ? EffectAttribution::Unknown
                                   : EffectAttribution::Exclusive,
            },
            .parent_index = parent_index,
            .begin_query = begin_query,
            .end_query = end_query,
        });
        const size_t index = active->intervals.size() - 1;
        active->open_intervals.push_back(index);
        cmdbuf.writeTimestamp2(kind == GpuIntervalKind::CommandBuffer
                                  ? vk::PipelineStageFlagBits2::eTopOfPipe
                                  : BeginStage(kind),
                              active->timestamps.get(), begin_query);
        timestamp_queries_written += 2;
        ++intervals_recorded;
        return interval_id;
    }

    void CloseInterval(size_t index) {
        if (!active || index >= active->intervals.size()) {
            return;
        }
        auto& interval = active->intervals[index];
        if (interval.statistic_active) {
            const bool compute = interval.sample.statistic_kind == PipelineStatisticKind::Compute;
            cmdbuf.endQuery(compute ? active->compute_statistics.get()
                                    : active->graphics_statistics.get(),
                            interval.statistic_query);
            interval.statistic_active = false;
        }
        cmdbuf.writeTimestamp2(EndStage(interval.sample.kind), active->timestamps.get(),
                              interval.end_query);
        if (!active->open_intervals.empty() && active->open_intervals.back() == index) {
            active->open_intervals.pop_back();
        }
    }

    void EndInterval(u64 token) {
        if (!active || token == 0) {
            return;
        }
        while (!active->open_intervals.empty()) {
            const size_t index = active->open_intervals.back();
            CloseInterval(index);
            if (active->intervals[index].sample.interval_id == token) {
                return;
            }
        }
    }

    void BeginRendering(u64 new_attachment_hash) {
        if (!active) {
            return;
        }
        ClosePipelineBlock();
        attachment_hash = new_attachment_hash;
        rendering_interval = BeginInterval(GpuIntervalKind::RenderingScope, new_attachment_hash, 0);
    }

    void EndRendering() {
        ClosePipelineBlock();
        if (rendering_interval != 0) {
            EndInterval(rendering_interval);
            rendering_interval = 0;
        }
        attachment_hash = 0;
    }

    void BeginPipelineBlock(GpuIntervalKind kind, u64 pipeline_hash) {
        pipeline_interval = BeginInterval(kind, pipeline_hash, 0);
        pipeline_key = pipeline_hash;
        pipeline_commands = 0;
        pipeline_block_limit = PipelineBlockSize(pipeline_hash);
        if (!active || pipeline_interval == 0 || !active->statistic_sample) {
            return;
        }
        auto& interval = active->intervals.back();
        const bool compute = kind == GpuIntervalKind::ComputePipelineBlock;
        u32& count = compute ? active->compute_statistic_queries
                             : active->graphics_statistic_queries;
        if (count >= statistic_query_budget) {
            return;
        }
        interval.statistic_query = count++;
        interval.statistic_active = true;
        interval.sample.statistic_kind =
            compute ? PipelineStatisticKind::Compute : PipelineStatisticKind::Graphics;
        cmdbuf.beginQuery(compute ? active->compute_statistics.get()
                                  : active->graphics_statistics.get(),
                          interval.statistic_query, {});
        ++statistic_queries_written;
    }

    void ClosePipelineBlock() {
        if (pipeline_interval == 0) {
            return;
        }
        if (active && !active->intervals.empty()) {
            active->intervals.back().sample.command_count = pipeline_commands;
        }
        EndInterval(pipeline_interval);
        pipeline_interval = 0;
        pipeline_key = 0;
        pipeline_commands = 0;
        pipeline_block_limit = 64;
    }

    u32 PipelineBlockSize(u64 pipeline_hash) const {
        const auto it = adaptive_block_sizes.find(pipeline_hash);
        return it == adaptive_block_sizes.end() ? 64 : it->second;
    }

    void PipelineCommands(GpuIntervalKind kind, u64 pipeline_hash, u32 command_count) {
        if (!active || !active->detailed_sample) {
            return;
        }
        if (pipeline_interval == 0 || pipeline_key != pipeline_hash ||
            active->intervals.back().sample.kind != kind) {
            ClosePipelineBlock();
            BeginPipelineBlock(kind, pipeline_hash);
        }
        pipeline_commands += command_count;
        if (pipeline_commands >= pipeline_block_limit) {
            ClosePipelineBlock();
        }
    }

    u64 TimestampDelta(u64 begin, u64 end) const {
        const u32 bits = instance.TimestampValidBits();
        if (bits >= 64) {
            return end - begin;
        }
        const u64 mask = (u64{1} << bits) - 1;
        return (end - begin) & mask;
    }

    bool ReadTimestamps(const Slot& slot, std::vector<u64>& results) {
        results.resize(static_cast<size_t>(slot.timestamp_queries) * 2);
        const VkResult result = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueryPoolResults(
            static_cast<VkDevice>(device), static_cast<VkQueryPool>(slot.timestamps.get()), 0,
            slot.timestamp_queries, results.size() * sizeof(u64), results.data(), 2 * sizeof(u64),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result != VK_SUCCESS) {
            return false;
        }
        for (u32 query = 0; query < slot.timestamp_queries; ++query) {
            if (results[query * 2 + 1] == 0) {
                return false;
            }
        }
        return true;
    }

    template <size_t CounterCount>
    bool ReadStatistics(vk::QueryPool pool, u32 query_count, std::vector<u64>& results) {
        if (query_count == 0) {
            results.clear();
            return true;
        }
        constexpr size_t Stride = CounterCount + 1;
        results.resize(static_cast<size_t>(query_count) * Stride);
        const VkResult result = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueryPoolResults(
            static_cast<VkDevice>(device), static_cast<VkQueryPool>(pool), 0, query_count,
            results.size() * sizeof(u64), results.data(), Stride * sizeof(u64),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result != VK_SUCCESS) {
            return false;
        }
        for (u32 query = 0; query < query_count; ++query) {
            if (results[static_cast<size_t>(query) * Stride + CounterCount] == 0) {
                return false;
            }
        }
        return true;
    }

    void CollectSlot(Slot& slot) {
        if (!ReadTimestamps(slot, timestamp_results) ||
            !ReadStatistics<GraphicsStatisticCount>(slot.graphics_statistics.get(),
                                                    slot.graphics_statistic_queries,
                                                    graphics_statistic_results) ||
            !ReadStatistics<ComputeStatisticCount>(slot.compute_statistics.get(),
                                                   slot.compute_statistic_queries,
                                                   compute_statistic_results)) {
            ++query_results_not_ready;
            return;
        }

        interval_durations.resize(slot.intervals.size());
        for (size_t i = 0; i < slot.intervals.size(); ++i) {
            auto& interval = slot.intervals[i];
            const u64 begin = timestamp_results[interval.begin_query * 2];
            const u64 end = timestamp_results[interval.end_query * 2];
            interval.sample.gpu_begin_tick = begin;
            interval.sample.gpu_end_tick = end;
            interval.sample.duration_ns = static_cast<u64>(std::llround(
                static_cast<double>(TimestampDelta(begin, end)) * instance.TimestampPeriodNs()));
            interval.sample.exclusive_ns = interval.sample.duration_ns;
            interval.sample.status = GpuQueryStatus::Available;
            interval_durations[i] = interval.sample.duration_ns;

            if (interval.sample.statistic_kind == PipelineStatisticKind::Graphics) {
                constexpr size_t Stride = GraphicsStatisticCount + 1;
                const size_t offset = static_cast<size_t>(interval.statistic_query) * Stride;
                interval.sample.input_assembly_vertices = graphics_statistic_results[offset + 0];
                interval.sample.input_assembly_primitives =
                    graphics_statistic_results[offset + 1];
                interval.sample.vertex_shader_invocations =
                    graphics_statistic_results[offset + 2];
                interval.sample.clipping_invocations = graphics_statistic_results[offset + 3];
                interval.sample.clipping_primitives = graphics_statistic_results[offset + 4];
                interval.sample.fragment_shader_invocations =
                    graphics_statistic_results[offset + 5];
            } else if (interval.sample.statistic_kind == PipelineStatisticKind::Compute) {
                constexpr size_t Stride = ComputeStatisticCount + 1;
                const size_t offset = static_cast<size_t>(interval.statistic_query) * Stride;
                interval.sample.compute_shader_invocations = compute_statistic_results[offset];
            }
        }

        for (size_t child = 0; child < slot.intervals.size(); ++child) {
            const size_t parent = slot.intervals[child].parent_index;
            if (parent == std::numeric_limits<size_t>::max() || parent >= slot.intervals.size()) {
                continue;
            }
            auto& parent_sample = slot.intervals[parent].sample;
            parent_sample.exclusive_ns = parent_sample.exclusive_ns > interval_durations[child]
                                             ? parent_sample.exclusive_ns -
                                                   interval_durations[child]
                                             : 0;
        }

        for (auto& interval : slot.intervals) {
            if ((interval.sample.kind == GpuIntervalKind::GraphicsPipelineBlock ||
                 interval.sample.kind == GpuIntervalKind::ComputePipelineBlock) &&
                interval.sample.pipeline_hash != 0) {
                adaptive_block_sizes[interval.sample.pipeline_hash] =
                    interval.sample.duration_ns >= 1'000'000
                        ? 8
                        : interval.sample.duration_ns >= 500'000 ? 16 : 64;
            }
            RecordGpuInterval(interval.sample);
        }

        const auto root = std::find_if(slot.intervals.begin(), slot.intervals.end(),
                                       [](const Interval& interval) {
                                           return interval.sample.kind ==
                                                  GpuIntervalKind::CommandBuffer;
                                       });
        if (root != slot.intervals.end()) {
            if (last_gpu_end_tick != 0 &&
                slot.command_buffer_ordinal == last_collected_command_buffer_ordinal + 1) {
                const u64 gap_ticks = TimestampDelta(last_gpu_end_tick, root->sample.gpu_begin_tick);
                RecordGpuInterval(GpuIntervalSample{
                    .interval_id = NextGpuIntervalSeq(),
                    .query_frame_id = slot.query_frame_id,
                    .frame_seq = root->sample.frame_seq,
                    .command_buffer_seq = root->sample.command_buffer_seq,
                    .submit_seq = slot.submit_seq,
                    .gpu_begin_tick = last_gpu_end_tick,
                    .gpu_end_tick = root->sample.gpu_begin_tick,
                    .duration_ns = static_cast<u64>(std::llround(
                        static_cast<double>(gap_ticks) * instance.TimestampPeriodNs())),
                    .exclusive_ns = static_cast<u64>(std::llround(
                        static_cast<double>(gap_ticks) * instance.TimestampPeriodNs())),
                    .kind = GpuIntervalKind::QueueGap,
                    .status = GpuQueryStatus::Available,
                    .attribution = EffectAttribution::Unknown,
                });
            }
            last_gpu_end_tick = root->sample.gpu_end_tick;
            last_collected_command_buffer_ordinal = slot.command_buffer_ordinal;
        }
        query_results_available += slot.timestamp_queries + slot.graphics_statistic_queries +
                                   slot.compute_statistic_queries;
        slot.pending = false;
    }

    void Collect() {
        ++query_collect_calls;
        const bool sample_cost = (query_collect_calls & 63) == 0;
        const u64 collect_start = sample_cost ? Timestamp() : 0;
        std::array<Slot*, QuerySlotCount> ready{};
        size_t count{};
        const u64 completed_tick = master_semaphore.KnownGpuTick();
        for (auto& slot : slots) {
            if (slot.pending && slot.retire_tick <= completed_tick) {
                ready[count++] = &slot;
            }
        }
        std::sort(ready.begin(), ready.begin() + count, [](const Slot* lhs, const Slot* rhs) {
            return lhs->query_frame_id < rhs->query_frame_id;
        });
        for (size_t i = 0; i < count; ++i) {
            CollectSlot(*ready[i]);
        }
        query_collect_ready_slots += count;
        if (sample_cost) {
            ++query_collect_cost_samples;
            query_collect_sampled_ns += Timestamp() - collect_start;
        }
        if (command_buffers_seen != 0 && command_buffers_seen % 256 == 0 &&
            last_health_at != command_buffers_seen) {
            last_health_at = command_buffers_seen;
            RecordHealth();
        }
    }

    const Instance& instance;
    MasterSemaphore& master_semaphore;
    vk::Device device;
    std::array<Slot, QuerySlotCount> slots{};
    Slot* active{};
    vk::CommandBuffer cmdbuf{};
    u32 next_slot{};
    const u32 timestamp_sample_period;
    const u32 statistic_sample_period;
    const u32 timestamp_query_budget;
    const u32 statistic_query_budget;
    const bool timestamps_supported;
    const bool statistics_supported;
    u64 rendering_interval{};
    u64 pipeline_interval{};
    u64 pipeline_key{};
    u64 attachment_hash{};
    u32 pipeline_commands{};
    u32 pipeline_block_limit{64};
    u64 last_gpu_end_tick{};
    u64 last_collected_command_buffer_ordinal{};
    u64 last_health_at{};
    std::vector<u64> timestamp_results;
    std::vector<u64> graphics_statistic_results;
    std::vector<u64> compute_statistic_results;
    std::vector<u64> interval_durations;
    std::unordered_map<u64, u32> adaptive_block_sizes;
    u64 command_buffers_seen{};
    u64 command_buffers_sampled{};
    u64 command_buffers_detailed{};
    u64 slots_unavailable{};
    u64 intervals_seen{};
    u64 intervals_recorded{};
    u64 intervals_filtered{};
    u64 intervals_budget_dropped{};
    u64 query_results_available{};
    u64 query_results_not_ready{};
    u64 timestamp_queries_written{};
    u64 statistic_queries_written{};
    u64 query_collect_calls{};
    u64 query_collect_cost_samples{};
    u64 query_collect_ready_slots{};
    u64 query_collect_sampled_ns{};
    u64 calibration_calls{};
    u64 calibration_cpu_ns{};
};

#else

struct GpuProfiler::Impl {};

#endif

bool PipelineExecutableCaptureEnabled() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_GPU_TELEMETRY_PIPELINE_EXECUTABLES");
        return value != nullptr &&
               (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
                std::strcmp(value, "on") == 0);
    }();
    return enabled;
#else
    return false;
#endif
}

void RecordPipelineExecutableStatistics(const Instance& instance, vk::Pipeline pipeline,
                                        u64 pipeline_hash, bool is_compute) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (!Common::PerformanceTelemetry::Enabled() || !PipelineExecutableCaptureEnabled() ||
        !instance.SupportsPipelineExecutableProperties() || !pipeline) {
        return;
    }
    const vk::Device device = instance.GetDevice();
    const auto [properties_result, properties] =
        device.getPipelineExecutablePropertiesKHR(vk::PipelineInfoKHR{.pipeline = pipeline});
    if (properties_result != vk::Result::eSuccess) {
        return;
    }
    for (u32 executable_index = 0; executable_index < properties.size(); ++executable_index) {
        const auto& property = properties[executable_index];
        const auto [statistics_result, statistics] = device.getPipelineExecutableStatisticsKHR(
            vk::PipelineExecutableInfoKHR{
                .pipeline = pipeline,
                .executableIndex = executable_index,
            });
        if (statistics_result != vk::Result::eSuccess) {
            continue;
        }
        if (statistics.empty()) {
            Common::PerformanceTelemetry::RecordGpuPipelineExecutable(
                Common::PerformanceTelemetry::GpuPipelineExecutableSample{
                    .pipeline_hash = pipeline_hash,
                    .executable_name_hash = HashText(property.name),
                    .stage_bits = static_cast<u64>(static_cast<VkShaderStageFlags>(property.stages)),
                    .executable_index = executable_index,
                    .subgroup_size = property.subgroupSize,
                    .is_compute = static_cast<u8>(is_compute),
                });
            continue;
        }
        for (const auto& statistic : statistics) {
            Common::PerformanceTelemetry::RecordGpuPipelineExecutable(
                Common::PerformanceTelemetry::GpuPipelineExecutableSample{
                    .pipeline_hash = pipeline_hash,
                    .executable_name_hash = HashText(property.name),
                    .statistic_name_hash = HashText(statistic.name),
                    .statistic_value = StatisticValueBits(statistic),
                    .stage_bits = static_cast<u64>(static_cast<VkShaderStageFlags>(property.stages)),
                    .executable_index = executable_index,
                    .subgroup_size = property.subgroupSize,
                    .statistic_format = static_cast<u32>(statistic.format),
                    .is_compute = static_cast<u8>(is_compute),
                });
        }
    }
#else
    static_cast<void>(instance);
    static_cast<void>(pipeline);
    static_cast<void>(pipeline_hash);
    static_cast<void>(is_compute);
#endif
}

GpuProfiler::GpuProfiler(const Instance& instance, MasterSemaphore& master_semaphore)
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    : impl{std::make_unique<Impl>(instance, master_semaphore)} {}
#else
    : impl{std::make_unique<Impl>()} {
    static_cast<void>(instance);
    static_cast<void>(master_semaphore);
}
#endif

GpuProfiler::~GpuProfiler() = default;

void GpuProfiler::BeginCommandBuffer(
    vk::CommandBuffer cmdbuf, Common::PerformanceTelemetry::CmdBufferSeq command_buffer_seq,
    Common::PerformanceTelemetry::FrameSeq frame_seq) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    impl->BeginCommandBuffer(cmdbuf, command_buffer_seq, frame_seq);
#else
    static_cast<void>(cmdbuf);
    static_cast<void>(command_buffer_seq);
    static_cast<void>(frame_seq);
#endif
}

void GpuProfiler::EndCommandBuffer(Common::PerformanceTelemetry::SubmitSeq submit_seq,
                                   u64 retire_tick) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    impl->EndCommandBuffer(submit_seq, retire_tick);
#else
    static_cast<void>(submit_seq);
    static_cast<void>(retire_tick);
#endif
}

void GpuProfiler::Collect() {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    impl->Collect();
#endif
}

void GpuProfiler::BeginRendering(u64 attachment_hash) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    impl->BeginRendering(attachment_hash);
#else
    static_cast<void>(attachment_hash);
#endif
}

void GpuProfiler::EndRendering() {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    impl->EndRendering();
#endif
}

void GpuProfiler::GraphicsDraw(u64 pipeline_hash, u32 command_count) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    impl->PipelineCommands(Common::PerformanceTelemetry::GpuIntervalKind::GraphicsPipelineBlock,
                           pipeline_hash, command_count);
#else
    static_cast<void>(pipeline_hash);
    static_cast<void>(command_count);
#endif
}

void GpuProfiler::ComputeDispatch(u64 pipeline_hash, u32 command_count) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    impl->PipelineCommands(Common::PerformanceTelemetry::GpuIntervalKind::ComputePipelineBlock,
                           pipeline_hash, command_count);
#else
    static_cast<void>(pipeline_hash);
    static_cast<void>(command_count);
#endif
}

u64 GpuProfiler::BeginInterval(Common::PerformanceTelemetry::GpuIntervalKind kind, u64 object_hash,
                               u64 bytes) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    impl->ClosePipelineBlock();
    return impl->BeginInterval(kind, object_hash, bytes);
#else
    static_cast<void>(kind);
    static_cast<void>(object_hash);
    static_cast<void>(bytes);
    return 0;
#endif
}

void GpuProfiler::EndInterval(u64 token) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    impl->EndInterval(token);
#else
    static_cast<void>(token);
#endif
}

} // namespace Vulkan
