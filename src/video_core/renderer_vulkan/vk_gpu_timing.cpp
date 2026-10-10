// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <numeric>
#include <span>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_gpu_timing.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {
namespace {

constexpr u32 BankCount = 8;
constexpr u32 QueryCount = 65536;
constexpr u32 TransferQueryCount = 4096;
constexpr u64 ReportFrames = 64;
constexpr std::array Names{"draw", "dispatch", "detile", "tile", "postfx", "postprocess",
                           "composition", "buffer_copy", "image_upload", "image_readback",
                           "image_copy", "resolve", "fill", "clear", "barrier", "render_begin",
                           "render_end", "transfer_upload", "image_blit", "buffer_fault"};
static_assert(Names.size() == u32(GpuWork::Count));

u64 Delta(u64 begin, u64 end, u32 bits) {
    return (end - begin) & (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1);
}

struct Key {
    GpuWork kind;
    std::array<u64, 5> ids;
    bool operator==(const Key&) const = default;
};

struct KeyHash {
    size_t operator()(const Key& key) const {
        size_t hash = u32(key.kind);
        for (const u64 id : key.ids) {
            hash ^= id + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct Stats {
    double ns{};
    double max_ns{};
    u64 calls{};
    u64 work{};

    void Add(double duration, u64 amount) {
        ns += duration;
        max_ns = std::max(max_ns, duration);
        ++calls;
        work += amount;
    }
};

struct QueryResult {
    u64 value;
    u64 available;
};

struct Event {
    GpuTimingLabel label;
    u64 work;
    u32 begin;
};

struct Bank {
    std::atomic<u32> state{};
    vk::UniqueQueryPool pool;
    vk::UniqueQueryPool transfer_pool;
    std::vector<Event> events;
    std::vector<Event> transfers;
    u32 queries{};
    u32 transfer_queries{};
    u32 end_query{};
    u64 frame{};
    u64 tick{};
    u64 dropped_events{};
    bool closed{};
};

} // namespace

struct GpuTiming::Impl {
    Impl(const Instance& instance, vk::Semaphore timeline_, bool presentation)
        : device{instance.GetDevice()}, timeline{timeline_}, period{instance.TimestampPeriodNs()},
          bits{instance.TimestampValidBits()}, transfer_bits{instance.TransferTimestampValidBits()},
          upload_queue{instance.HasTransferQueue() ? "transfer" : "graphics_prologue"},
          queue_name{presentation ? "present" : "guest"} {
        if (bits == 0 || period <= 0) {
            LOG_WARNING(Render_Vulkan, "GPU64 {}: GPU timestamps are unsupported", queue_name);
            return;
        }
        if (transfer_bits == 0) {
            LOG_WARNING(Render_Vulkan, "GPU64 {}: upload queue has no timestamp support", queue_name);
        }
        for (auto& bank : banks) {
            auto [result, pool] = device.createQueryPoolUnique(vk::QueryPoolCreateInfo{
                .queryType = vk::QueryType::eTimestamp, .queryCount = QueryCount});
            if (result != vk::Result::eSuccess) {
                LOG_WARNING(Render_Vulkan, "GPU64 {}: query pool creation failed: {}", queue_name,
                            vk::to_string(result));
                return;
            }
            bank.pool = std::move(pool);
            bank.events.reserve(8192);
            bank.transfers.reserve(128);
            if (transfer_bits != 0) {
                auto [transfer_result, transfer_pool] =
                    device.createQueryPoolUnique(vk::QueryPoolCreateInfo{
                        .queryType = vk::QueryType::eTimestamp,
                        .queryCount = TransferQueryCount});
                if (transfer_result != vk::Result::eSuccess) {
                    LOG_WARNING(Render_Vulkan, "GPU64 {}: transfer query pool creation failed",
                                queue_name);
                    return;
                }
                bank.transfer_pool = std::move(transfer_pool);
            }
        }
        enabled = true;
        worker = std::jthread([this](std::stop_token stop) { Run(stop); });
        LOG_INFO(Render_Vulkan,
                 "GPU64 {}: enabled; ALL_COMMANDS completion intervals, report every 64 frames; "
                 "period={} ns bits={} transfer_bits={}; enabled by SHADPS4_GPU_TIMING=1",
                 queue_name, period, bits, transfer_bits);
    }

    ~Impl() {
        if (!enabled) {
            return;
        }
        const u64 tick = last_tick;
        if (tick != 0) {
            const auto result = device.waitSemaphores(vk::SemaphoreWaitInfo{
                .semaphoreCount = 1, .pSemaphores = &timeline, .pValues = &tick}, ~u64{0});
            if (result != vk::Result::eSuccess) {
                LOG_WARNING(Render_Vulkan, "GPU64 {}: shutdown wait failed: {}", queue_name,
                            vk::to_string(result));
            }
        }
        worker.request_stop();
        cv.notify_one();
        worker.join();
    }

    void BeginCommandBuffer(vk::CommandBuffer cmdbuf) {
        if (!enabled || frame_open) {
            return;
        }
        frame_open = true;
        ++frame;
        for (auto& bank : banks) {
            if (bank.state.load(std::memory_order_acquire) != 0) {
                continue;
            }
            active = &bank;
            bank.state.store(1, std::memory_order_relaxed);
            bank.events.clear();
            bank.transfers.clear();
            bank.queries = 1;
            bank.transfer_queries = 0;
            bank.frame = frame;
            bank.closed = false;
            bank.dropped_events = 0;
            cmdbuf.resetQueryPool(*bank.pool, 0, QueryCount);
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *bank.pool, 0);
            return;
        }
    }

    void Begin(vk::CommandBuffer cmdbuf, GpuTimingLabel label, u64 work) {
        command_open = false;
        if (!active || active->closed) {
            return;
        }
        if (active->queries + 2 >= QueryCount) {
            ++active->dropped_events;
            return;
        }
        const u32 query = active->queries;
        active->queries += 2;
        active->events.push_back({label, work, query});
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *active->pool, query);
        command_open = true;
    }

    void End(vk::CommandBuffer cmdbuf) {
        if (command_open) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *active->pool,
                                  active->events.back().begin + 1);
            command_open = false;
        }
    }

    void EndFrame(vk::CommandBuffer cmdbuf) {
        if (active && !active->closed) {
            active->end_query = active->queries++;
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *active->pool,
                                  active->end_query);
            active->closed = true;
        }
        frame_closed = true;
    }

    void OnSubmit(u64 tick) {
        last_tick = tick;
        if (!frame_closed) {
            return;
        }
        if (active) {
            active->tick = tick;
            active->state.store(2, std::memory_order_release);
            active = nullptr;
        }
        published_tick.store(tick, std::memory_order_relaxed);
        published_frame.store(frame, std::memory_order_release);
        frame_closed = frame_open = false;
        cv.notify_one();
    }

    void BeginTransfer(vk::CommandBuffer cmdbuf, GpuTimingLabel label, u64 bytes) {
        transfer_open = false;
        if (!active) {
            return;
        }
        if (transfer_bits == 0 || active->transfer_queries + 2 > TransferQueryCount) {
            ++active->dropped_events;
            return;
        }
        const u32 query = active->transfer_queries;
        active->transfer_queries += 2;
        active->transfers.push_back({label, bytes, query});
        cmdbuf.resetQueryPool(*active->transfer_pool, query, 2);
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands,
                              *active->transfer_pool, query);
        transfer_open = true;
    }

    void EndTransfer(vk::CommandBuffer cmdbuf) {
        if (transfer_open) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands,
                                  *active->transfer_pool, active->transfers.back().begin + 1);
            transfer_open = false;
        }
    }

    bool Read(vk::QueryPool pool, u32 count, std::span<QueryResult> output) {
        if (count == 0) {
            return true;
        }
        const auto result = device.getQueryPoolResults(
            pool, 0, count, count * sizeof(QueryResult), output.data(), sizeof(QueryResult),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        return result == vk::Result::eSuccess &&
               std::all_of(output.begin(), output.begin() + count,
                           [](const auto& query) { return query.available != 0; });
    }

    void Accumulate(std::span<const Event> events, u32 valid_bits,
                    std::span<const QueryResult> output) {
        for (const auto& event : events) {
            const double ns = Delta(output[event.begin].value,
                                    output[event.begin + 1].value, valid_bits) * period;
            const auto& label = event.label;
            categories[u32(label.kind)].Add(ns, event.work);
            resources[Key{label.kind, {label.pipeline, label.shader0, label.shader1,
                                       label.resource0, label.resource1}}].Add(ns, event.work);
        }
    }

    bool Collect(Bank& bank) {
        if (!Read(*bank.pool, bank.queries, results) ||
            (!bank.transfers.empty() &&
             !Read(*bank.transfer_pool, bank.transfer_queries, transfer_results))) {
            return false;
        }
        const u64 next_window = (bank.frame - 1) / ReportFrames;
        while (window < next_window) {
            Report(false, (window + 1) * ReportFrames);
        }
        frame_span.Add(Delta(results[0].value, results[bank.end_query].value, bits) * period, 0);
        Accumulate(bank.events, bits, results);
        Accumulate(bank.transfers, transfer_bits, transfer_results);
        ++samples;
        dropped_events += bank.dropped_events;
        return true;
    }

    void Report(bool partial, u64 last) {
        const u64 first = window * ReportFrames + 1;
        const u64 divisor = std::max(u64{1}, samples);
        const u64 dropped_frames = last - first + 1 - samples;
        double accounted{};
        for (u32 i = 0; i < u32(GpuWork::Count); ++i) {
            if (i != u32(GpuWork::TransferUpload)) {
                accounted += categories[i].ns;
            }
        }
        LOG_INFO(Render_Vulkan,
                 "GPU64 {} frames={}-{} samples={} partial={} span_ms/sample={:.4f} "
                 "unattributed_ms/sample={:.4f} scopes={} dropped_frames={} dropped_events={} "
                 "resources={} upload_queue={}; uploads are separate and may overlap; "
                 "span includes queue gaps/waits; intervals measure completion contributions",
                 queue_name, first, last, samples, partial, frame_span.ns / divisor / 1e6,
                 std::max(0.0, frame_span.ns - accounted) / divisor / 1e6,
                 std::accumulate(categories.begin(), categories.end(), u64{0},
                                 [](u64 sum, const auto& stats) { return sum + stats.calls; }),
                 dropped_frames, dropped_events, resources.size(), upload_queue);
        const auto row = [&](std::string_view name, const Key& key, const Stats& stats) {
            file << fmt::format("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:x}\t{:x}\t{:x}\t{:x}\t{:x}"
                                "\t{}\t{:.6f}\t{:.6f}\t{:.3f}\t{}\n",
                                first, last, queue_name, samples, partial, dropped_frames,
                                dropped_events, name, key.ids[0],
                                key.ids[1], key.ids[2], key.ids[3], key.ids[4], stats.calls,
                                stats.ns / 1e6, stats.ns / divisor / 1e6, stats.max_ns / 1e3,
                                stats.work);
        };
        row("frame_span", {}, frame_span);
        for (u32 i = 0; i < u32(GpuWork::Count); ++i) {
            const auto& stats = categories[i];
            if (stats.calls == 0) {
                continue;
            }
            LOG_INFO(Render_Vulkan, "GPU64 {} category={} ms/sample={:.4f} calls={} max_us={:.3f}",
                     queue_name, Names[i], stats.ns / divisor / 1e6, stats.calls, stats.max_ns / 1e3);
            row(fmt::format("total_{}", Names[i]), {}, stats);
        }
        using Ranked = std::pair<const Key, Stats>;
        std::vector<const Ranked*> ranked;
        ranked.reserve(resources.size());
        for (const auto& resource : resources) {
            ranked.push_back(&resource);
        }
        std::ranges::sort(ranked, [](const auto* a, const auto* b) {
            return a->second.ns > b->second.ns;
        });
        for (size_t i = 0; i < ranked.size(); ++i) {
            const auto& [key, stats] = *ranked[i];
            row(Names[u32(key.kind)], key, stats);
            if (i < 20) {
                LOG_INFO(Render_Vulkan,
                         "GPU64 {} rank={} kind={} ms/sample={:.4f} calls={} pipeline={:#x} "
                         "shader0={:#x} shader1={:#x} resource0={:#x} resource1={:#x}",
                         queue_name, i + 1, Names[u32(key.kind)], stats.ns / divisor / 1e6,
                         stats.calls, key.ids[0], key.ids[1], key.ids[2], key.ids[3], key.ids[4]);
            }
        }
        file.flush();
        resources.clear();
        categories = {};
        frame_span = {};
        samples = dropped_events = 0;
        ++window;
    }

    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:GpuTiming");
        results.resize(QueryCount);
        transfer_results.resize(TransferQueryCount);
        const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) /
                          fmt::format("gpu_timings_{}.tsv", queue_name);
        file.open(path, std::ios::out | std::ios::trunc);
        file << "first_frame\tlast_frame\tqueue\tsamples\tpartial\tdropped_frames\tdropped_events"
                "\tkind\tpipeline\tshader0"
                "\tshader1\tresource0\tresource1\tcalls\tgpu_ms\tms_per_sample\tmax_us\twork_hint\n";
        LOG_INFO(Render_Vulkan, "GPU64 {}: full resource timings: {}", queue_name,
                 Common::FS::PathToUTF8String(path));
        if (!file) {
            LOG_ERROR(Render_Vulkan, "GPU64 {}: cannot open timing TSV", queue_name);
        }
        while (true) {
            const u64 published = published_frame.load(std::memory_order_acquire);
            const u64 published_submit = published_tick.load(std::memory_order_relaxed);
            const auto [result, completed] = device.getSemaphoreCounterValue(timeline);
            if (result != vk::Result::eSuccess) {
                LOG_WARNING(Render_Vulkan, "GPU64 {}: timeline query failed: {}", queue_name,
                            vk::to_string(result));
                break;
            }
            std::array<Bank*, BankCount> pending{};
            size_t count{};
            for (auto& bank : banks) {
                if (bank.state.load(std::memory_order_acquire) == 2 && bank.frame <= published) {
                    pending[count++] = &bank;
                }
            }
            std::sort(pending.begin(), pending.begin() + count,
                      [](const auto* a, const auto* b) { return a->frame < b->frame; });
            for (size_t i = 0; i < count; ++i) {
                auto& bank = *pending[i];
                if (bank.tick > completed || !Collect(bank)) {
                    break;
                }
                bank.state.store(0, std::memory_order_release);
            }
            if (published_submit <= completed) {
                while ((window + 1) * ReportFrames <= published) {
                    const u64 boundary = (window + 1) * ReportFrames;
                    const bool uncollected = std::ranges::any_of(banks, [boundary](const Bank& bank) {
                        return bank.state.load(std::memory_order_acquire) == 2 &&
                               bank.frame <= boundary;
                    });
                    if (uncollected) {
                        break;
                    }
                    Report(false, boundary);
                }
            }
            if (stop.stop_requested()) {
                break;
            }
            std::unique_lock lock{mutex};
            cv.wait_for(lock, std::chrono::milliseconds(5));
        }
        const u64 published = published_frame.load(std::memory_order_acquire);
        if (published > window * ReportFrames) {
            Report(true, published);
        }
    }

    vk::Device device;
    vk::Semaphore timeline;
    double period;
    u32 bits;
    u32 transfer_bits;
    const char* upload_queue;
    const char* queue_name;
    std::array<Bank, BankCount> banks;
    Bank* active{};
    u64 frame{};
    u64 last_tick{};
    bool enabled{};
    bool frame_open{};
    bool frame_closed{};
    bool command_open{};
    bool transfer_open{};
    std::atomic<u64> published_frame{};
    std::atomic<u64> published_tick{};
    std::jthread worker;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<QueryResult> results;
    std::vector<QueryResult> transfer_results;
    std::ofstream file;
    std::unordered_map<Key, Stats, KeyHash> resources;
    std::array<Stats, u32(GpuWork::Count)> categories{};
    Stats frame_span;
    u64 window{};
    u64 samples{};
    u64 dropped_events{};
};

bool GpuTiming::Requested() {
    const char* env = std::getenv("SHADPS4_GPU_TIMING");
    return env != nullptr && env[0] == '1';
}

GpuTiming::GpuTiming(const Instance& instance, vk::Semaphore timeline, bool presentation)
    : impl{std::make_unique<Impl>(instance, timeline, presentation)} {}
GpuTiming::~GpuTiming() = default;
bool GpuTiming::IsEnabled() const { return impl->enabled; }
void GpuTiming::BeginCommandBuffer(vk::CommandBuffer cmdbuf) { impl->BeginCommandBuffer(cmdbuf); }
void GpuTiming::Begin(vk::CommandBuffer cmdbuf, GpuTimingLabel label, u64 work) {
    impl->Begin(cmdbuf, label, work);
}
void GpuTiming::End(vk::CommandBuffer cmdbuf) { impl->End(cmdbuf); }
void GpuTiming::EndFrame(vk::CommandBuffer cmdbuf) { impl->EndFrame(cmdbuf); }
void GpuTiming::OnSubmit(u64 tick) { impl->OnSubmit(tick); }
void GpuTiming::BeginTransfer(vk::CommandBuffer cmdbuf, GpuTimingLabel label, u64 bytes) {
    impl->BeginTransfer(cmdbuf, label, bytes);
}
void GpuTiming::EndTransfer(vk::CommandBuffer cmdbuf) { impl->EndTransfer(cmdbuf); }

GpuTimingScope::GpuTimingScope(Scheduler& scheduler_, GpuWork kind, u64 resource0,
                               u64 resource1, u64 work) {
    if (scheduler_.BeginGpuTiming(kind, resource0, resource1, work)) {
        scheduler = &scheduler_;
    }
}
GpuTimingScope::~GpuTimingScope() {
    if (scheduler) {
        scheduler->EndGpuTiming();
    }
}
GpuTimingContext::GpuTimingContext(Scheduler& scheduler_, GpuTimingLabel label) {
    if (scheduler_.HasGpuTiming()) {
        scheduler = &scheduler_;
        previous = scheduler->ExchangeGpuTimingLabel(label);
    }
}
GpuTimingContext::~GpuTimingContext() {
    if (scheduler) {
        scheduler->ExchangeGpuTimingLabel(previous);
    }
}

} // namespace Vulkan
