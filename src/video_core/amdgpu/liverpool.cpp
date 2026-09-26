// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/preprocessor/stringize.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <utility>

#include "common/assert.h"
#include "common/debug.h"
#include "common/hash.h"
#include "common/performance_telemetry.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/flush_epoch.h"
#include "video_core/gpu_authority_tracker.h"
#include "video_core/guest_copy_engine.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace AmdGpu {

namespace {

/// The command processor is about to write guest memory that deferred copies may still read.
inline void PrepareGuestWrite(VAddr address, u64 size) {
    VideoCore::GuestCopyEngine::Instance().WaitForGuestWrite(address, size);
}

inline void PrepareGuestWrite(const void* address, u64 size) {
    PrepareGuestWrite(std::bit_cast<VAddr>(address), size);
}

/// Everything parsed before a completion signal must have finished reading guest memory before
/// the guest can observe the signal and recycle that memory.
inline void CompleteGuestReads(u64 guest_copy_seq) {
    VideoCore::GuestCopyEngine::Instance().WaitCompleted(guest_copy_seq);
}

inline void CompleteGuestReads() {
    VideoCore::GuestCopyEngine::Instance().Drain();
}

[[nodiscard]] inline u64 GuestCopySeq() {
    return VideoCore::GuestCopyEngine::Instance().SubmittedSeq();
}

constexpr bool IsSyncPm4Opcode(PM4ItOpcode op) {
    switch (op) {
    case PM4ItOpcode::EventWrite:
    case PM4ItOpcode::EventWriteEos:
    case PM4ItOpcode::EventWriteEop:
    case PM4ItOpcode::ReleaseMem:
    case PM4ItOpcode::WaitRegMem:
    case PM4ItOpcode::WriteData:
    case PM4ItOpcode::DmaData:
    case PM4ItOpcode::MemSemaphore:
    case PM4ItOpcode::AcquireMem:
    case PM4ItOpcode::Rewind:
        return true;
    default:
        return false;
    }
}

struct GraphicsRegisterRange {
    u32 first;
    u32 last;
};

#define GRAPHICS_REG_RANGE(field)                                                                  \
    GraphicsRegisterRange {                                                                        \
        static_cast<u32>(offsetof(Regs, field) / sizeof(u32)),                                     \
            static_cast<u32>((offsetof(Regs, field) + sizeof(((Regs*)nullptr)->field)) /           \
                             sizeof(u32))                                                          \
    }

constexpr std::array GraphicsPipelineRegisterRanges = {
    GraphicsRegisterRange{
        static_cast<u32>(offsetof(Regs, ps_program) / sizeof(u32)),
        static_cast<u32>((offsetof(Regs, ps_program) + offsetof(ShaderProgram, user_data)) /
                         sizeof(u32))},
    GraphicsRegisterRange{
        static_cast<u32>(offsetof(Regs, vs_program) / sizeof(u32)),
        static_cast<u32>((offsetof(Regs, vs_program) + offsetof(ShaderProgram, user_data)) /
                         sizeof(u32))},
    GraphicsRegisterRange{
        static_cast<u32>(offsetof(Regs, gs_program) / sizeof(u32)),
        static_cast<u32>((offsetof(Regs, gs_program) + offsetof(ShaderProgram, user_data)) /
                         sizeof(u32))},
    GraphicsRegisterRange{
        static_cast<u32>(offsetof(Regs, es_program) / sizeof(u32)),
        static_cast<u32>((offsetof(Regs, es_program) + offsetof(ShaderProgram, user_data)) /
                         sizeof(u32))},
    GraphicsRegisterRange{
        static_cast<u32>(offsetof(Regs, hs_program) / sizeof(u32)),
        static_cast<u32>((offsetof(Regs, hs_program) + offsetof(ShaderProgram, user_data)) /
                         sizeof(u32))},
    GraphicsRegisterRange{
        static_cast<u32>(offsetof(Regs, ls_program) / sizeof(u32)),
        static_cast<u32>((offsetof(Regs, ls_program) + offsetof(ShaderProgram, user_data)) /
                         sizeof(u32))},
    GRAPHICS_REG_RANGE(depth_buffer),
    GRAPHICS_REG_RANGE(depth_render_override),
    GRAPHICS_REG_RANGE(clipper_control),
    GRAPHICS_REG_RANGE(polygon_control),
    GRAPHICS_REG_RANGE(color_control),
    GRAPHICS_REG_RANGE(color_shader_mask),
    GRAPHICS_REG_RANGE(color_target_mask),
    GRAPHICS_REG_RANGE(color_export_format),
    GRAPHICS_REG_RANGE(color_buffers),
    GRAPHICS_REG_RANGE(blend_control),
    GRAPHICS_REG_RANGE(stage_enable),
    GRAPHICS_REG_RANGE(vs_output_control),
    GRAPHICS_REG_RANGE(vs_output_config),
    GRAPHICS_REG_RANGE(shader_pos_format),
    GRAPHICS_REG_RANGE(vgt_instance_step_rate_0),
    GRAPHICS_REG_RANGE(vgt_instance_step_rate_1),
    GRAPHICS_REG_RANGE(vgt_esgs_ring_itemsize),
    GRAPHICS_REG_RANGE(vgt_gsvs_ring_itemsize),
    GRAPHICS_REG_RANGE(ls_hs_config),
    GRAPHICS_REG_RANGE(tess_config),
    GraphicsRegisterRange{static_cast<u32>(offsetof(Regs, stage_enable) / sizeof(u32) - 7),
                          static_cast<u32>(offsetof(Regs, stage_enable) / sizeof(u32) - 6)},
    GRAPHICS_REG_RANGE(vgt_gs_instance_cnt),
    GRAPHICS_REG_RANGE(vgt_gs_out_prim_type),
    GRAPHICS_REG_RANGE(vgt_gs_vert_itemsize),
    GRAPHICS_REG_RANGE(vgt_gs_mode),
    GRAPHICS_REG_RANGE(vgt_strmout_config),
    GRAPHICS_REG_RANGE(ps_input_ena),
    GraphicsRegisterRange{static_cast<u32>(offsetof(Regs, ps_input_addr) / sizeof(u32)),
                          static_cast<u32>((offsetof(Regs, z_export_format) +
                                            sizeof(((Regs*)nullptr)->z_export_format)) /
                                           sizeof(u32))},
    GRAPHICS_REG_RANGE(depth_shader_control),
    GRAPHICS_REG_RANGE(ps_inputs),
    GRAPHICS_REG_RANGE(primitive_type),
};

#undef GRAPHICS_REG_RANGE

consteval auto BuildGraphicsPipelineRegisterMask() {
    std::array<u64, (Regs::NumRegs + 63) / 64> mask{};
    for (const auto& range : GraphicsPipelineRegisterRanges) {
        for (u32 index = range.first; index < range.last; ++index) {
            mask[index / 64] |= 1ULL << (index % 64);
        }
    }
    return mask;
}

constexpr auto GraphicsPipelineRegisterMask = BuildGraphicsPipelineRegisterMask();
constexpr auto MemoryWaitFallbackInterval = std::chrono::microseconds{250};

template <typename Condvar, typename Lock, typename Rep, typename Period, typename Pred>
#if defined(__clang__) || defined(__GNUC__)
[[gnu::cold]]
#endif
SHAD_NO_INLINE bool WaitMemoryFallback(
    Condvar& cv, std::unique_lock<Lock>& lk, std::stop_token stoken,
    const std::chrono::duration<Rep, Period>& timeout, Pred&& pred) {
    return cv.wait_for(lk, stoken, timeout, std::forward<Pred>(pred));
}

template <u32 WordCount>
[[nodiscard]] inline u32 PipelineRegisterBits(u32 first_register) {
    static_assert(WordCount <= 32);
    const u32 bit_offset = first_register & 63U;
    const u32 mask_index = first_register / 64;
    u64 bits = GraphicsPipelineRegisterMask[mask_index] >> bit_offset;
    if (bit_offset > 64 - WordCount) [[unlikely]] {
        bits |= GraphicsPipelineRegisterMask[mask_index + 1] << (64 - bit_offset);
    }
    return static_cast<u32>(bits);
}

[[nodiscard]] u32 NextReadyQueue(u64 ready_mask, s32 current_queue) {
    ASSERT(ready_mask != 0);
    if (std::has_single_bit(ready_mask)) [[likely]] {
        return std::countr_zero(ready_mask);
    }
    const u32 first_queue = static_cast<u32>(current_queue + 1);
    const u64 queues_after_current = ready_mask & (~0ULL << first_queue);
    return std::countr_zero(queues_after_current != 0 ? queues_after_current : ready_mask);
}

static SHAD_NO_INLINE bool InvalidGraphicsRegisterRange(u32 first_register, u32 word_count) {
    ASSERT(first_register <= Regs::NumRegs && word_count <= Regs::NumRegs - first_register);
    return false;
}

static SHAD_NO_INLINE bool WriteComputeProgramRegisters(ComputeProgram& program, u32 register_offset,
                                                        const u32* payload, u32 word_count,
                                                        bool telemetry_enabled) {
    const size_t byte_count = static_cast<size_t>(word_count) * sizeof(u32);
    ASSERT(byte_count <= sizeof(ComputeProgram));
    auto* destination = reinterpret_cast<u32*>(&program) + (register_offset - 0x200);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    const bool changed = telemetry_enabled && std::memcmp(destination, payload, byte_count) != 0;
#else
    static_cast<void>(telemetry_enabled);
    constexpr bool changed = false;
#endif
    std::memcpy(destination, payload, byte_count);
    return changed;
}

static SHAD_NO_INLINE void GraphicsPacketAssertionFailed() {
    ASSERT(false);
}

[[noreturn]] static SHAD_NO_INLINE void InvalidDmaData(const PM4DmaData& packet) {
    UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(packet.src_sel.Value()),
                    u32(packet.dst_sel.Value()));
}

[[noreturn]] static SHAD_NO_INLINE void UnsupportedWriteDataAddressMode() {
    UNREACHABLE();
}

static SHAD_NO_INLINE void WarnStrmoutBufferUpdate(const PM4CmdStrmoutBufferUpdate& packet) {
    LOG_WARNING(Render_Vulkan,
                "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                "source_select = {}, buffer_select = {}",
                packet.update_memory.Value(), magic_enum::enum_name(packet.source_select.Value()),
                packet.buffer_select.Value());
}

static SHAD_NO_INLINE void WarnGetLodStats() {
    LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
}

static SHAD_NO_INLINE void WarnReservedCondExec() {
    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
}

static SHAD_NO_INLINE void WarnSetPredication() {
    LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
}

static SHAD_NO_INLINE void WarnCopyData(const PM4CmdCopyData& packet) {
    LOG_WARNING(Render,
                "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                u32(packet.src_sel.Value()), u32(packet.dst_sel.Value()),
                packet.count_sel.Value(), packet.wr_confirm.Value(),
                u32(packet.engine_sel.Value()));
}

static SHAD_NO_INLINE void BeginHostMarker(Vulkan::Rasterizer& rasterizer,
                                           const void* command_address,
                                           std::string_view command_name) {
    rasterizer.ScopeMarkerBegin(fmt::format("gfx:{}:{}", command_address, command_name));
}

[[noreturn]] static SHAD_NO_INLINE void UnknownType3Opcode(PM4ItOpcode opcode, u32 count) {
    UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}", static_cast<u32>(opcode),
                    count);
}

[[nodiscard]] inline bool TestWaitValue(u32 value, PM4CmdWaitRegMem::Function function, u32 mask,
                                        u32 reference) {
    if (mask == 0xffffffffU) [[likely]] {
        if (function == PM4CmdWaitRegMem::Function::NotEqual) [[likely]] {
            return value != reference;
        }
        if (function == PM4CmdWaitRegMem::Function::Equal) {
            return value == reference;
        }
    }
    return PM4CmdWaitRegMem::TestValue(value, function, mask, reference);
}

void RecordPendingCompletionHazard(Common::PerformanceTelemetry::CausalTraceToken& trace,
                                   u64 completed_stages, u64 completed_writes,
                                   u64 requirement_bits, u8 confidence) {
    if (!Common::PerformanceTelemetry::HeavyEnabled()) {
        return;
    }
    if (trace.hazard_id == 0) {
        trace.hazard_id = Common::PerformanceTelemetry::NextHazardSeq();
    }
    Common::PerformanceTelemetry::RecordHazardResolution(
        Common::PerformanceTelemetry::HazardResolutionSample{
            .hazard_id = trace.hazard_id,
            .cause_id = trace.cause_id,
            .candidate_id = trace.candidate_id,
            .src_stage = completed_stages,
            .src_access = completed_writes,
            .sync_requirement_bits = requirement_bits,
            .resolution = Common::PerformanceTelemetry::HazardResolutionKind::Pending,
            .avoidability = Common::PerformanceTelemetry::Avoidability::ConservativeFallback,
            .confidence = confidence,
        });
}

Common::PerformanceTelemetry::CausalTraceToken RecordWaitScope(
    Common::PerformanceTelemetry::Pm4Engine engine, u32 queue_id, VAddr wait_address,
    u64 reference, Common::PerformanceTelemetry::WaitSeq wait_seq,
    const Common::PerformanceTelemetry::CausalTraceToken& producer_trace,
    bool producer_correlated) {
    using namespace Common::PerformanceTelemetry;
    if (!Enabled()) {
        return {};
    }
    CausalTraceToken trace{
        .candidate_id = producer_trace.candidate_id,
        .scope_id = NextScopeSeq(),
        .cause_id = producer_trace.cause_id ? producer_trace.cause_id : NextCauseSeq(),
        .signal_id = producer_trace.signal_id,
        .hazard_id = producer_trace.hazard_id,
    };
    const auto packet_seq = CurrentPacketSeq();
    RecordCompletionScope(CompletionScopeSample{
        .scope_id = trace.scope_id,
        .cause_id = trace.cause_id,
        .signal_id = trace.signal_id,
        .frame_seq = CurrentFrameSeq(),
        .command_buffer_seq = CurrentCmdBufferSeq(),
        .first_packet_seq = packet_seq,
        .last_packet_seq = packet_seq,
        .completed_stage_bits = static_cast<u64>(FenceStageScope::All),
        .visible_access_bits = reference,
        .label_addr = wait_address,
        .label_value = reference,
        .pm4_digest = HashCombine(wait_address, reference),
        .queue_id = queue_id,
        .engine = static_cast<u16>(engine),
        .kind = CompletionScopeKind::WaitRegMem,
        .confidence = static_cast<u8>(producer_correlated ? 255 : 96),
    });
    if (trace.signal_id != 0) {
        RecordLogicalSignal(LogicalSignalSample{
            .signal_id = trace.signal_id,
            .candidate_id = trace.candidate_id,
            .scope_id = trace.scope_id,
            .cause_id = trace.cause_id,
            .wait_seq = wait_seq,
            .packet_seq = packet_seq,
            .label_addr = wait_address,
            .value = reference,
            .phase = LogicalSignalPhase::WaitMatched,
            .action = SignalAction::VirtualGpuWait,
        });
    }
    RecordPendingCompletionHazard(
        trace, static_cast<u64>(FenceStageScope::All), 0,
        static_cast<u64>(SyncRequirement::ExecutionOrder) |
            static_cast<u64>(SyncRequirement::MemoryVisibility),
        static_cast<u8>(producer_correlated ? 255 : 96));
    return trace;
}

void RecordWaitEffect(const Common::PerformanceTelemetry::CausalTraceToken& trace,
                      Common::PerformanceTelemetry::WaitSeq wait_seq, u64 duration_ns,
                      bool virtualized, bool producer_correlated) {
    using namespace Common::PerformanceTelemetry;
    if (!Enabled() || trace.scope_id == 0) {
        return;
    }
    RecordHazardResolution(HazardResolutionSample{
        .hazard_id = trace.hazard_id,
        .cause_id = trace.cause_id,
        .candidate_id = trace.candidate_id,
        .sync_requirement_bits = static_cast<u64>(SyncRequirement::ExecutionOrder) |
                                 static_cast<u64>(SyncRequirement::MemoryVisibility),
        .resolution = virtualized ? HazardResolutionKind::ImplicitDependency
                                  : HazardResolutionKind::LegacyFallback,
        .avoidability = virtualized ? Avoidability::ProvenEliminable
                                    : Avoidability::ConservativeFallback,
        .confidence = static_cast<u8>(producer_correlated ? 255 : 96),
    });
    RecordCausalEffect(CausalEffectSample{
        .effect_id = NextEffectSeq(),
        .cause_id = trace.cause_id,
        .candidate_id = trace.candidate_id,
        .scope_id = trace.scope_id,
        .hazard_id = trace.hazard_id,
        .object_id = wait_seq,
        .command_buffer_seq = CurrentCmdBufferSeq(),
        .duration_ns = duration_ns,
        .kind = CausalEffectKind::HostWait,
        .attribution = EffectAttribution::Exclusive,
        .avoidability = virtualized ? Avoidability::ProvenEliminable
                                    : Avoidability::ConservativeFallback,
        .confidence = static_cast<u8>(producer_correlated ? 255 : 96),
    });
}

} // namespace

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
#define COUNT_FAILED_MEMORY_WAIT() ++failed_tests
#else
#define COUNT_FAILED_MEMORY_WAIT() static_cast<void>(0)
#endif

#define WAIT_MEMORY(queue_id, address, condition, yield_command)                                  \
    do {                                                                                           \
        while (!(condition)) {                                                                     \
            COUNT_FAILED_MEMORY_WAIT();                                                            \
            const bool memory_watch_armed =                                                       \
                ArmMemoryWait(queue_id, reinterpret_cast<VAddr>(address));                         \
            if (memory_watch_armed && (condition)) {                                               \
                CancelMemoryWait(queue_id);                                                        \
                break;                                                                             \
            }                                                                                      \
            yield_command                                                                          \
            if (memory_watch_armed) {                                                              \
                CancelMemoryWait(queue_id);                                                        \
            }                                                                                      \
        }                                                                                          \
    } while (false)

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

static SHAD_NO_INLINE std::span<const u32> InvalidNextPacket(std::span<const u32> span,
                                                             size_t offset) {
    LOG_ERROR(Lib_GnmDriver,
              ": packet length exceeds remaining submission size. Packet dword count={}, "
              "remaining submission dwords={}",
              offset, span.size());
    return {};
}

static inline std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset <= span.size()) [[likely]] {
        return span.subspan(offset);
    }
    return InvalidNextPacket(span, offset);
}

static SHAD_NO_INLINE std::span<const u32> NextNonType3Packet(std::span<const u32> dcb, u32 type) {
    const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
    switch (type) {
    case 0:
        UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                        header->type0.base.Value(), header->type0.NumWords());
    case 2:
        // Type-2 packets are used for padding purposes
        if (Common::PerformanceTelemetry::Enabled()) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::Pm4Type2Packets, 1);
        }
        return NextPacket(dcb, 1);
    default:
        UNREACHABLE_MSG("Wrong PM4 type {}", type);
    }
}

Liverpool::Liverpool() {
    for (u32 queue_id = 0; queue_id < NumTotalQueues; ++queue_id) {
        memory_waits[queue_id].owner = this;
        memory_waits[queue_id].queue_id = queue_id;
    }
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    submit_cv.notify_all();
    process_thread.join();
    for (u32 queue_id = 0; queue_id < NumTotalQueues; ++queue_id) {
        CancelMemoryWait(queue_id);
    }
}

bool Liverpool::ArmMemoryWait(u32 queue_id, VAddr address) {
    if (rasterizer == nullptr) [[unlikely]] {
        return false;
    }

    auto& wait = memory_waits[queue_id];
    ASSERT(wait.id == 0);
    const u64 queue_bit = 1ULL << queue_id;
    blocked_queue_mask.fetch_or(queue_bit, std::memory_order_acq_rel);
    ready_queue_mask.fetch_and(~queue_bit, std::memory_order_acq_rel);

    const auto watch = rasterizer->ArmMemoryWriteWatch(
        address,
        [](void* user_data, VAddr, u64, VideoCore::MemoryWriteSource) noexcept {
            auto& context = *static_cast<MemoryWaitContext*>(user_data);
            context.owner->WakeMemoryWait(context.queue_id);
        },
        &wait);
    if (!watch) [[unlikely]] {
        WakeMemoryWait(queue_id);
        return false;
    }
    wait.page = watch.page;
    wait.id = watch.id;
    wait.epoch = watch.epoch;
    return true;
}

void Liverpool::CancelMemoryWait(u32 queue_id) {
    auto& wait = memory_waits[queue_id];
    if (wait.id == 0) {
        return;
    }

    const VideoCore::MemoryWriteWatch watch{
        .page = wait.page,
        .id = wait.id,
        .epoch = wait.epoch,
    };
    wait.id = 0;
    if (rasterizer != nullptr) {
        rasterizer->CancelMemoryWriteWatch(watch);
    }
    WakeMemoryWait(queue_id);
}

void Liverpool::WakeMemoryWait(u32 queue_id) noexcept {
    const u64 queue_bit = 1ULL << queue_id;
    blocked_queue_mask.fetch_and(~queue_bit, std::memory_order_acq_rel);
    ready_queue_mask.fetch_or(queue_bit, std::memory_order_release);
    submit_cv.notify_one();
}

void Liverpool::ReleaseMemoryWaitFallbacks(bool telemetry_enabled) noexcept {
    const u64 blocked = blocked_queue_mask.exchange(0, std::memory_order_acq_rel);
    if (blocked == 0) {
        return;
    }
    ready_queue_mask.fetch_or(blocked, std::memory_order_release);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::MemoryWatchFallbacks,
            std::popcount(blocked));
    }
#else
    static_cast<void>(telemetry_enabled);
#endif
}

void Liverpool::ProcessCommands() {
    // Process incoming commands with high priority
    while (num_commands.load(std::memory_order_acquire) != 0) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();
    VideoCore::GuestCopyEngine::Instance().SetProducerThread();
    curr_qid = -1;

    while (!stoken.stop_requested()) {
        bool memory_wait_fallback{};
        {
            Common::PerformanceTelemetry::ScopedDuration blocked{
                Common::PerformanceTelemetry::Counter::GcpBlockedNs,
                Common::PerformanceTelemetry::EventType::GcpBlocked};
            std::unique_lock lk{submit_mutex};
            const auto has_ready_work = [this] {
                return num_commands.load(std::memory_order_acquire) != 0 ||
                       ready_queue_mask.load(std::memory_order_acquire) != 0 ||
                       (submit_done.load(std::memory_order_acquire) &&
                        num_submits.load(std::memory_order_acquire) == 0);
            };
            if (num_submits.load(std::memory_order_acquire) != 0 &&
                ready_queue_mask.load(std::memory_order_acquire) == 0) {
                memory_wait_fallback = !WaitMemoryFallback(
                    submit_cv, lk, stoken, MemoryWaitFallbackInterval, has_ready_work);
            } else {
                Common::CondvarWait(submit_cv, lk, stoken, has_ready_work);
            }
        }
        if (stoken.stop_requested()) {
            break;
        }

        const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
        [[maybe_unused]] const bool telemetry_detail =
            telemetry_enabled && Common::PerformanceTelemetry::HeavyEnabled();
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::GcpWakes, 1);
        }
        if (memory_wait_fallback) {
            ReleaseMemoryWaitFallbacks(telemetry_enabled);
        }
        Common::PerformanceTelemetry::ScopedDuration active{
            telemetry_enabled, Common::PerformanceTelemetry::Counter::GcpActiveNs,
            Common::PerformanceTelemetry::EventType::GcpActive};
        VideoCore::StartCapture();

        for (;;) {
            if (num_commands.load(std::memory_order_acquire) != 0) [[unlikely]] {
                ProcessCommands();
            }

            const u64 ready_queues = ready_queue_mask.load(std::memory_order_acquire);
            if (ready_queues == 0) {
                break;
            }
            curr_qid = static_cast<s32>(NextReadyQueue(ready_queues, curr_qid));
            if (telemetry_enabled) {
                Common::PerformanceTelemetry::AddEnabled(
                    Common::PerformanceTelemetry::Counter::QueueScans, 1);
            }

            Task::Handle task = active_tasks[curr_qid];
            if (!task) [[unlikely]] {
                auto& queue = mapped_queues[curr_qid];
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    ready_queue_mask.fetch_and(~(1ULL << curr_qid), std::memory_order_acq_rel);
                    continue;
                }
                task = queue.submits.front();
                active_tasks[curr_qid] = task;
                if (telemetry_enabled) {
                    Common::PerformanceTelemetry::AddEnabled(
                        Common::PerformanceTelemetry::Counter::QueueFrontLoads, 1);
                }
            }
            if (telemetry_enabled) {
                auto& ready_since_ns = task.promise().telemetry_ready_since_ns;
                if (ready_since_ns != 0) {
                    Common::PerformanceTelemetry::RecordDurationEnabled(
                        Common::PerformanceTelemetry::Counter::QueueReadyNs,
                        Common::PerformanceTelemetry::EventType::None, ready_since_ns,
                        static_cast<u64>(curr_qid));
                }
                ready_since_ns = 0;
            }
            {
                Common::PerformanceTelemetry::ScopedDuration resume{
                    telemetry_enabled, Common::PerformanceTelemetry::Counter::QueueResumeNs};
                if (telemetry_enabled) {
                    Common::PerformanceTelemetry::AddEnabled(
                        Common::PerformanceTelemetry::Counter::QueueResumes, 1);
                }
                task.resume();
            }

            if (task.done()) [[unlikely]] {
                active_tasks[curr_qid] = {};
                task.destroy();

                {
                    auto& queue = mapped_queues[curr_qid];
                    std::scoped_lock lock{queue.m_access};
                    queue.submits.pop();
                    const u64 queue_bit = 1ULL << curr_qid;
                    if (queue.submits.empty()) {
                        ready_queue_mask.fetch_and(~queue_bit, std::memory_order_acq_rel);
                    } else {
                        ready_queue_mask.fetch_or(queue_bit, std::memory_order_release);
                    }
                }

                // WaitGpuIdle and IsGpuIdle let the guest treat a retired submit as consumed.
                CompleteGuestReads();
                {
                    std::scoped_lock lock{submit_mutex};
                    --num_submits;
                    submit_cv.notify_all();
                }
            }
        }

        if (rasterizer && queued_signal_tick != 0 &&
            queued_signal_tick >= rasterizer->CurrentTick()) {
            // Signals queued behind readback commits wait for this tick; with the command
            // processor idle nothing else would submit it.
            rasterizer->Flush(Common::PerformanceTelemetry::SubmitReason::WaitProgress);
        }
        if (num_submits.load(std::memory_order_acquire) == 0) {
            if (submit_done) {
                VideoCore::EndCapture();
                if (rasterizer) {
                    rasterizer->OnSubmit();
                    rasterizer->Flush(
                        Common::PerformanceTelemetry::SubmitReason::GuestSubmit);
                }
                submit_done = false;
            }
            CompleteGuestReads();
            Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
        }
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb, u32 ib_depth) {
    FIBER_ENTER(ccb_task_name);
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    [[maybe_unused]] const bool telemetry_detail =
        telemetry_enabled && Common::PerformanceTelemetry::HeavyEnabled();

    while (!ccb.empty()) {
        if (num_commands.load(std::memory_order_acquire) != 0) [[unlikely]] {
            ProcessCommands();
        }

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::CountPm4PacketEnabled(
                Common::PerformanceTelemetry::Pm4Engine::Constant, GfxQueueId,
                static_cast<u32>(opcode), ib_depth, reinterpret_cast<uintptr_t>(header),
                header->type3.NumWords() + 1, header->raw);
        }
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
            if (telemetry_detail) {
                Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                    Common::PerformanceTelemetry::Pm4Engine::Constant, GfxQueueId,
                    static_cast<u32>(opcode), ib_depth,
                    header->type3.NumWords() != 0 ? it_body[0] : 0, 0);
            }
#endif
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            PrepareGuestWrite(dump_const->Address<void*>(), dump_const->Size());
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            if (rasterizer) {
                rasterizer->NotifyMemoryWrite(std::bit_cast<VAddr>(dump_const->Address<void*>()),
                                              dump_const->Size(),
                                              VideoCore::MemoryWriteSource::CommandProcessor);
            }
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size},
                                ib_depth + 1);
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}

#ifdef _MSC_VER
#define SHAD_LOCAL_FORCE_INLINE __forceinline
#else
#define SHAD_LOCAL_FORCE_INLINE __attribute__((always_inline)) inline
#endif

template <u32 NumWords>
SHAD_LOCAL_FORCE_INLINE bool Liverpool::WriteGraphicsRegistersSmall(u32 first_register,
                                                                     const u32* payload) {
    static_assert(NumWords == 1 || NumWords == 2);
    if (first_register > Regs::NumRegs - NumWords) [[unlikely]] {
        return InvalidGraphicsRegisterRange(first_register, NumWords);
    }

    u32* const destination = &regs.reg_array[first_register];
    if constexpr (NumWords == 1) {
        const u32 new_value = *payload;
        if (*destination == new_value) [[likely]] {
            return false;
        }
        const bool pipeline_state_changed = PipelineRegisterBits<1>(first_register) != 0;
        *destination = new_value;
        ++graphics_state_generation;
        graphics_pipeline_generation += pipeline_state_changed;
    } else {
        u64 old_values{};
        u64 new_values{};
        std::memcpy(&old_values, destination, sizeof(old_values));
        std::memcpy(&new_values, payload, sizeof(new_values));
        const u64 different = old_values ^ new_values;
        if (different == 0) [[likely]] {
            return false;
        }
        const u32 changed_registers = static_cast<u32>(static_cast<u32>(different) != 0) |
                                      (static_cast<u32>(different >> 32) != 0) << 1;
        const bool pipeline_state_changed =
            (changed_registers & PipelineRegisterBits<2>(first_register)) != 0;
        std::memcpy(destination, &new_values, sizeof(new_values));
        ++graphics_state_generation;
        graphics_pipeline_generation += pipeline_state_changed;
    }
    return true;
}

SHAD_LOCAL_FORCE_INLINE bool Liverpool::WriteGraphicsRegisters(u32 first_register,
                                                                const u32* payload,
                                                                u32 word_count) {
    if (word_count == 1) [[likely]] {
        return WriteGraphicsRegistersSmall<1>(first_register, payload);
    }
    if (word_count == 2) {
        return WriteGraphicsRegistersSmall<2>(first_register, payload);
    }

    if (first_register > Regs::NumRegs || word_count > Regs::NumRegs - first_register) [[unlikely]] {
        return InvalidGraphicsRegisterRange(first_register, word_count);
    }
    if (word_count == 0) {
        return false;
    }
    return WriteGraphicsRegistersSlow(first_register, payload, word_count);
}

SHAD_LOCAL_FORCE_INLINE bool Liverpool::WriteGraphicsRegisters4(u32 first_register,
                                                                 const u32* payload) {
    if (first_register > Regs::NumRegs - 4) [[unlikely]] {
        return InvalidGraphicsRegisterRange(first_register, 4);
    }

    u32* const destination = &regs.reg_array[first_register];
    const auto old_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(destination));
    const auto new_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(payload));
    const u32 equal_registers = static_cast<u32>(
        _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(old_values, new_values))));
    const u32 changed_registers = (~equal_registers) & 0xf;
    if (changed_registers == 0) [[likely]] {
        return false;
    }

    const bool pipeline_state_changed =
        (changed_registers & PipelineRegisterBits<4>(first_register)) != 0;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(destination), new_values);
    ++graphics_state_generation;
    graphics_pipeline_generation += pipeline_state_changed;
    return true;
}

SHAD_NO_INLINE bool Liverpool::WriteGraphicsRegisters8(u32 first_register, const u32* payload) {
    if (first_register > Regs::NumRegs - 8) [[unlikely]] {
        return InvalidGraphicsRegisterRange(first_register, 8);
    }

    u32* const destination = &regs.reg_array[first_register];
    const auto old_values =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(destination));
    const auto new_values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(payload));
    const u32 equal_registers = static_cast<u32>(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(old_values, new_values))));
    const u32 changed_registers = (~equal_registers) & 0xff;
    if (changed_registers == 0) [[likely]] {
        return false;
    }

    const bool pipeline_state_changed =
        (changed_registers & PipelineRegisterBits<8>(first_register)) != 0;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination), new_values);
    ++graphics_state_generation;
    graphics_pipeline_generation += pipeline_state_changed;
    return true;
}

#undef SHAD_LOCAL_FORCE_INLINE

SHAD_NO_INLINE bool Liverpool::WriteGraphicsRegistersSlow(u32 first_register, const u32* payload,
                                                           u32 word_count) {
    u32* const destination = &regs.reg_array[first_register];
    const size_t byte_count = static_cast<size_t>(word_count) * sizeof(u32);
    if (std::memcmp(destination, payload, byte_count) == 0) {
        return false;
    }

    bool pipeline_state_changed = false;
    u32 processed = 0;
    while (processed < word_count && !pipeline_state_changed) {
        const u32 register_index = first_register + processed;
        const u32 bit_offset = register_index & 63U;
        const u32 chunk_size = std::min<u32>(word_count - processed, 64U - bit_offset);
        u64 relevant = GraphicsPipelineRegisterMask[register_index / 64] >> bit_offset;
        if (chunk_size != 64) {
            relevant &= (1ULL << chunk_size) - 1;
        }
        while (relevant != 0) {
            const u32 offset = processed + std::countr_zero(relevant);
            if (destination[offset] != payload[offset]) {
                pipeline_state_changed = true;
                break;
            }
            relevant &= relevant - 1;
        }
        processed += chunk_size;
    }

    std::memcpy(destination, payload, byte_count);
    ++graphics_state_generation;
    if (pipeline_state_changed) {
        ++graphics_pipeline_generation;
    }
    return true;
}

SHAD_NO_INLINE void Liverpool::HandleContextRegisterHint(u32 register_address, u32 packet_count,
                                                          const u32* payload) {
    switch (register_address) {
    case ContextRegs::CbColor0Base:
    case ContextRegs::CbColor1Base:
    case ContextRegs::CbColor2Base:
    case ContextRegs::CbColor3Base:
    case ContextRegs::CbColor4Base:
    case ContextRegs::CbColor5Base:
    case ContextRegs::CbColor6Base:
    case ContextRegs::CbColor7Base: {
        const auto col_buf_id = (register_address - ContextRegs::CbColor0Base) /
                                (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
        ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

        if (packet_count == 0x0e || packet_count == 0x0d || packet_count == 0x0b) {
            ASSERT_MSG(payload[packet_count] == 0xc0001000,
                       "NOP hint is missing in CB setup sequence");
            last_cb_extent[col_buf_id].raw = payload[packet_count + 1];
        } else {
            last_cb_extent[col_buf_id].raw = 0;
        }
        break;
    }
    case ContextRegs::CbColor0Cmask:
    case ContextRegs::CbColor1Cmask:
    case ContextRegs::CbColor2Cmask:
    case ContextRegs::CbColor3Cmask:
    case ContextRegs::CbColor4Cmask:
    case ContextRegs::CbColor5Cmask:
    case ContextRegs::CbColor6Cmask:
    case ContextRegs::CbColor7Cmask: {
        const auto col_buf_id = (register_address - ContextRegs::CbColor0Cmask) /
                                (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
        ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

        if (packet_count == 0x04) {
            ASSERT_MSG(payload[packet_count] == 0xc0001000,
                       "NOP hint is missing in CB setup sequence");
            last_cb_extent[col_buf_id].raw = payload[packet_count + 1];
        }
        break;
    }
    case ContextRegs::DbZInfo: {
        if (packet_count == 8) {
            ASSERT_MSG(payload[20] == 0xc0001000, "NOP hint is missing in DB setup sequence");
            last_db_extent.raw = payload[21];
        } else {
            last_db_extent.raw = 0;
        }
        break;
    }
    default:
        break;
    }
}

namespace {

SHAD_NO_INLINE void WriteFenceMemory(Vulkan::Rasterizer* rasterizer, void* address, u64 data,
                                     u32 num_bytes) {
    PrepareGuestWrite(address, num_bytes);
    auto* memory = Core::Memory::Instance();
    if (!memory->TryWriteBacking(address, &data, num_bytes)) {
        memcpy(address, &data, num_bytes);
        if (rasterizer) {
            rasterizer->NotifyMemoryWrite(std::bit_cast<VAddr>(address), num_bytes,
                                          VideoCore::MemoryWriteSource::CommandProcessor);
        }
    }
}

SHAD_NO_INLINE void SignalEventWriteEos(const PM4CmdEventWriteEos& packet,
                                        Vulkan::Rasterizer* rasterizer,
                                        const Common::PerformanceTelemetry::FenceTraceToken& fence_token = {},
                                        const Common::PerformanceTelemetry::CausalTraceToken& causal = {}) {
    {
        Common::PerformanceTelemetry::ScopedMemoryWriteOrigin origin{
            Common::PerformanceTelemetry::MemoryWriteOrigin::FenceSignal,
            fence_token.fence_seq};
        packet.SignalFence([rasterizer](void* address, u64 data, u32 num_bytes) {
            WriteFenceMemory(rasterizer, address, data, num_bytes);
        });
    }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Common::PerformanceTelemetry::HeavyEnabled()) {
        Common::PerformanceTelemetry::RecordFenceSignal(Common::PerformanceTelemetry::FenceSignalSample{
            .fence_seq = fence_token.fence_seq,
            .generation = fence_token.generation,
            .label_addr = fence_token.label_addr,
            .label_value = fence_token.label_val,
            .ready_tick = rasterizer ? rasterizer->CurrentTick() : 0,
            .signal_ns = Common::PerformanceTelemetry::Timestamp(),
            .writeback_count = 0,
            .writeback_bytes = 0,
            .gpu_ready_to_signal_ns = 0,
            .irq = 0,
            .signal_origin = Common::PerformanceTelemetry::MemoryWriteOrigin::FenceSignal,
        });
        if (causal.signal_id != 0) {
            Common::PerformanceTelemetry::RecordLogicalSignal(
            Common::PerformanceTelemetry::LogicalSignalSample{
                .signal_id = causal.signal_id,
                .candidate_id = causal.candidate_id,
                .scope_id = causal.scope_id,
                .cause_id = causal.cause_id,
                .packet_seq = fence_token.packet_seq,
                .label_addr = fence_token.label_addr,
                .label_generation = fence_token.generation,
                .value = fence_token.label_val,
                .producer_tick = rasterizer ? rasterizer->CurrentTick() : 0,
                .phase = Common::PerformanceTelemetry::LogicalSignalPhase::Published,
                .action =
                    Common::PerformanceTelemetry::SignalAction::PublishAfterPhysicalTick,
                .producer_submitted = 1,
                .producer_completed = 1,
            });
        }
    }
#endif
}

SHAD_NO_INLINE void SignalEventWriteEop(const PM4CmdEventWriteEop& packet,
                                        Vulkan::Rasterizer* rasterizer,
                                        const Common::PerformanceTelemetry::FenceTraceToken& fence_token = {},
                                        const Common::PerformanceTelemetry::CausalTraceToken& causal = {},
                                        bool write_label = true) {
    {
        Common::PerformanceTelemetry::ScopedMemoryWriteOrigin origin{
            Common::PerformanceTelemetry::MemoryWriteOrigin::FenceSignal,
            fence_token.fence_seq};
        packet.SignalFence(
            [rasterizer, write_label](void* address, u64 data, u32 num_bytes) {
                if (write_label) {
                    WriteFenceMemory(rasterizer, address, data, num_bytes);
                }
            },
            [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
    }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Common::PerformanceTelemetry::HeavyEnabled()) {
        Common::PerformanceTelemetry::RecordFenceSignal(Common::PerformanceTelemetry::FenceSignalSample{
            .fence_seq = fence_token.fence_seq,
            .generation = fence_token.generation,
            .label_addr = fence_token.label_addr,
            .label_value = fence_token.label_val,
            .ready_tick = rasterizer ? rasterizer->CurrentTick() : 0,
            .signal_ns = Common::PerformanceTelemetry::Timestamp(),
            .writeback_count = 0,
            .writeback_bytes = 0,
            .gpu_ready_to_signal_ns = 0,
            .irq = 1,
            .signal_origin = Common::PerformanceTelemetry::MemoryWriteOrigin::FenceSignal,
        });
        if (causal.signal_id != 0) {
            Common::PerformanceTelemetry::RecordLogicalSignal(
            Common::PerformanceTelemetry::LogicalSignalSample{
                .signal_id = causal.signal_id,
                .candidate_id = causal.candidate_id,
                .scope_id = causal.scope_id,
                .cause_id = causal.cause_id,
                .packet_seq = fence_token.packet_seq,
                .label_addr = fence_token.label_addr,
                .label_generation = fence_token.generation,
                .value = fence_token.label_val,
                .producer_tick = rasterizer ? rasterizer->CurrentTick() : 0,
                .phase = Common::PerformanceTelemetry::LogicalSignalPhase::Published,
                .action =
                    Common::PerformanceTelemetry::SignalAction::PublishAfterPhysicalTick,
                .irq = static_cast<u8>(packet.int_sel.Value() != InterruptSelect::None),
                .producer_submitted = 1,
                .producer_completed = 1,
            });
        }
    }
#endif
}

SHAD_NO_INLINE void SignalReleaseMem(const PM4CmdReleaseMem& release_packet,
                                     Vulkan::Rasterizer* rasterizer, u32 pipe_id,
                                     const Common::PerformanceTelemetry::FenceTraceToken& fence_token = {},
                                     const Common::PerformanceTelemetry::CausalTraceToken& causal = {},
                                     bool write_label = true) {
    PM4CmdReleaseMem packet = release_packet;
    if (!write_label) {
        // The label lies in memory that was unmapped; only the interrupt remains.
        packet.data_sel.Assign(DataSelect::None);
    }
    {
        Common::PerformanceTelemetry::ScopedMemoryWriteOrigin origin{
            Common::PerformanceTelemetry::MemoryWriteOrigin::FenceSignal,
            fence_token.fence_seq};
        packet.SignalFence(
            [pipe_id] {
                Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
            },
            [rasterizer](VAddr dst, u16 gds_index, u16 num_dwords) {
                rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
            });
    }
    const auto data_sel = packet.data_sel.Value();
    if (rasterizer && data_sel != DataSelect::None && data_sel != DataSelect::GdsMemStore) {
        const u64 write_size = data_sel == DataSelect::Data32Low ? sizeof(u32) : sizeof(u64);
        rasterizer->NotifyMemoryWrite(packet.Address<VAddr>(), write_size,
                                      VideoCore::MemoryWriteSource::CommandProcessor);
    }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Common::PerformanceTelemetry::HeavyEnabled()) {
        Common::PerformanceTelemetry::RecordFenceSignal(Common::PerformanceTelemetry::FenceSignalSample{
            .fence_seq = fence_token.fence_seq,
            .generation = fence_token.generation,
            .label_addr = fence_token.label_addr,
            .label_value = fence_token.label_val,
            .ready_tick = rasterizer ? rasterizer->CurrentTick() : 0,
            .signal_ns = Common::PerformanceTelemetry::Timestamp(),
            .writeback_count = 0,
            .writeback_bytes = 0,
            .gpu_ready_to_signal_ns = 0,
            .irq = 1,
            .signal_origin = Common::PerformanceTelemetry::MemoryWriteOrigin::FenceSignal,
        });
        if (causal.signal_id != 0) {
            Common::PerformanceTelemetry::RecordLogicalSignal(
            Common::PerformanceTelemetry::LogicalSignalSample{
                .signal_id = causal.signal_id,
                .candidate_id = causal.candidate_id,
                .scope_id = causal.scope_id,
                .cause_id = causal.cause_id,
                .packet_seq = fence_token.packet_seq,
                .label_addr = fence_token.label_addr,
                .label_generation = fence_token.generation,
                .value = fence_token.label_val,
                .producer_tick = rasterizer ? rasterizer->CurrentTick() : 0,
                .phase = Common::PerformanceTelemetry::LogicalSignalPhase::Published,
                .action =
                    Common::PerformanceTelemetry::SignalAction::PublishAfterPhysicalTick,
                .irq = static_cast<u8>(packet.int_sel.Value() != InterruptSelect::None),
                .producer_submitted = 1,
                .producer_completed = 1,
            });
        }
    }
#endif
}

} // namespace

bool Liverpool::PublishCompletionSignal(VAddr label_addr, u64 label_size,
                                        Common::UniqueFunction<void, bool>&& publish) {
    auto& authority_tracker = VideoCore::GpuAuthorityTracker::Instance();
    const bool queued = authority_tracker.PublishSignal(VideoCore::SignalPublication{
        .label_addr = label_addr,
        .label_size = label_size,
        .publish = std::move(publish),
    });
    if (!queued) {
        return false;
    }
    // The signal follows readbacks committed when the GPU reaches the current tick at the
    // latest. The tick has to be submitted for that, even if the command processor goes idle.
    rasterizer->DeferGpuCompletion([] { VideoCore::GpuAuthorityTracker::Instance().PublishReady(); });
    queued_signal_tick = rasterizer->CurrentTick();
    Common::PerformanceTelemetry::Add(
        Common::PerformanceTelemetry::Counter::WritebackFenceDeferrals);
    return true;
}

SHAD_NO_INLINE void Liverpool::ProcessEventWriteEos(const PM4CmdEventWriteEos& packet) {
    Common::PerformanceTelemetry::ScopedFastDuration sync_time{
        Common::PerformanceTelemetry::Enabled(),
        Common::PerformanceTelemetry::Counter::GcpSyncPacketNs};
    Common::PerformanceTelemetry::CausalTraceToken completion_trace{};
    if (Common::PerformanceTelemetry::HeavyEnabled()) {
        completion_trace.scope_id = Common::PerformanceTelemetry::NextScopeSeq();
        completion_trace.cause_id = Common::PerformanceTelemetry::NextCauseSeq();
        if (packet.Address<VAddr>() != 0) {
            completion_trace.signal_id = Common::PerformanceTelemetry::NextSignalSeq();
        }
        const auto packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
        Common::PerformanceTelemetry::RecordCompletionScope(
            Common::PerformanceTelemetry::CompletionScopeSample{
                .scope_id = completion_trace.scope_id,
                .cause_id = completion_trace.cause_id,
                .signal_id = completion_trace.signal_id,
                .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                .command_buffer_seq = Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                .first_packet_seq = packet_seq,
                .last_packet_seq = packet_seq,
                .completed_stage_bits = packet.event_control,
                .completed_write_bits = packet.cmd_info,
                .visible_access_bits = packet.event_type.Value(),
                .cache_action_bits = packet.event_control,
                .label_addr = packet.Address<VAddr>(),
                .label_value = packet.DataDWord(),
                .pm4_digest = HashCombine(packet.event_control, packet.cmd_info),
                .queue_id = static_cast<u32>(GfxQueueId),
                .engine = static_cast<u16>(Common::PerformanceTelemetry::Pm4Engine::Graphics),
                .kind = Common::PerformanceTelemetry::CompletionScopeKind::EventWriteEos,
                .confidence = 255,
            });
        if (completion_trace.signal_id != 0) {
            Common::PerformanceTelemetry::RecordLogicalSignal(
                Common::PerformanceTelemetry::LogicalSignalSample{
                    .signal_id = completion_trace.signal_id,
                    .scope_id = completion_trace.scope_id,
                    .cause_id = completion_trace.cause_id,
                    .packet_seq = packet_seq,
                    .label_addr = packet.Address<VAddr>(),
                    .value = packet.DataDWord(),
                    .producer_tick = rasterizer ? rasterizer->CurrentTick() : 0,
                    .phase = Common::PerformanceTelemetry::LogicalSignalPhase::Created,
                    .action =
                        Common::PerformanceTelemetry::SignalAction::PublishAfterPhysicalTick,
                });
        }
    }
    RecordPendingCompletionHazard(
        completion_trace, packet.event_control, packet.cmd_info,
        static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::ExecutionOrder) |
            static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::MemoryVisibility) |
            (completion_trace.signal_id
                 ? static_cast<u64>(
                       Common::PerformanceTelemetry::SyncRequirement::HostSignalVisibility)
                 : 0),
        255);
    Common::PerformanceTelemetry::ScopedCausalContext completion_context{completion_trace};
    Common::PerformanceTelemetry::FenceTraceToken fence_token{};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Common::PerformanceTelemetry::HeavyEnabled() && packet.command == PM4CmdEventWriteEos::Command::SignalFence) {
        const auto fence_seq = Common::PerformanceTelemetry::NextFenceSeq();
        const auto gen = Common::PerformanceTelemetry::NextFenceGen();
        const auto pkt_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
        fence_token = {
            .fence_seq = fence_seq,
            .generation = gen,
            .packet_seq = pkt_seq,
            .label_addr = packet.Address<VAddr>(),
            .label_val = packet.data,
        };
        last_sync_packet = {
            .fence_seq = fence_seq,
            .generation = gen,
            .packet_seq = pkt_seq,
            .candidate_id = completion_trace.candidate_id,
            .scope_id = completion_trace.scope_id,
            .cause_id = completion_trace.cause_id,
            .signal_id = completion_trace.signal_id,
            .hazard_id = completion_trace.hazard_id,
            .label_addr = packet.Address<VAddr>(),
            .label_value = packet.data,
        };
        Common::PerformanceTelemetry::RecordFenceCreate(Common::PerformanceTelemetry::FenceCreateSample{
            .fence_seq = fence_seq,
            .generation = gen,
            .packet_seq = pkt_seq,
            .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
            .fence_kind = Common::PerformanceTelemetry::FenceKind::EventWriteEos,
            .raw_event_type = static_cast<u32>(packet.event_type.Value()),
            .decoded_event_type = static_cast<u32>(packet.event_type.Value()),
            .command = static_cast<u32>(packet.command.Value()),
            .label_addr = packet.Address<VAddr>(),
            .label_value = packet.data,
            .label_size = sizeof(u32),
            .queue_id = static_cast<u32>(GfxQueueId),
            .stage_scope = Common::PerformanceTelemetry::FenceStageScope::Ps,
            .interrupt_select = 0,
            .interrupt_id = 0,
            .data_select = static_cast<u32>(packet.command.Value()),
            .producer_begin_seq = Common::PerformanceTelemetry::CurrentProducerSeq(),
            .classification_initial = Common::PerformanceTelemetry::FenceClassification::Ambiguous,
        });
        if (packet.Address<VAddr>() != 0) {
            Common::PerformanceTelemetry::ArmReadWatchInterest(Common::PerformanceTelemetry::ReadWatchInterest{
                .fence_seq = fence_seq,
                .generation = gen,
                .readback_seq = 0,
                .resource_id = 0,
                .resource_version = 0,
                .guest_addr = packet.Address<VAddr>(),
                .size = sizeof(u32),
                .kind = Common::PerformanceTelemetry::ReadWatchKind::Label,
                .fence_packet = pkt_seq,
                .arm_timestamp_ns = Common::PerformanceTelemetry::Timestamp(),
            });
            if (rasterizer && Common::PerformanceTelemetry::IsSyncSemanticProfile()) {
                rasterizer->ArmSemanticReadWatch(packet.Address<VAddr>(), sizeof(u32));
            }
        }
    }
#endif
    VideoCore::FlushEpoch::AdvanceSync();
    VideoCore::TextureCache::DownloadDrain drain{};
    if (rasterizer) {
        Common::PerformanceTelemetry::ScopedFenceCause cause{fence_token};
        drain = rasterizer->ProcessDownloadImages(VideoCore::TextureCache::DownloadContext{
            .scope_id = completion_trace.scope_id,
            .cause_id = completion_trace.cause_id,
            .signal_id = completion_trace.signal_id,
            .hazard_id = completion_trace.hazard_id,
            .trigger = VideoCore::TextureCache::DownloadTrigger::EventWriteEos,
            .fence_seq = fence_token.fence_seq,
            .trigger_control = packet.event_control,
            .trigger_data_control = packet.cmd_info,
        });
    }
    const u64 guest_copy_seq = GuestCopySeq();
    if (packet.command == PM4CmdEventWriteEos::Command::GdsStore) {
        if (packet.size != 1) [[unlikely]] {
            GraphicsPacketAssertionFailed();
        }
        CompleteGuestReads(guest_copy_seq);
        if (rasterizer) {
            // The value lands when the GPU gets there; signals after it wait for it.
            rasterizer->StoreGdsAsync(packet.Address<VAddr>(), packet.gds_index);
            queued_signal_tick = rasterizer->CurrentTick();
        }
        return;
    }
    if (rasterizer && packet.command == PM4CmdEventWriteEos::Command::SignalFence &&
        (drain.drained != 0 ||
         VideoCore::GpuAuthorityTracker::Instance().MustOrderSignals())) {
        auto* completion_rasterizer = rasterizer;
        PublishCompletionSignal(
            packet.Address<VAddr>(), sizeof(u32),
            [packet, completion_rasterizer, fence_token, completion_trace,
             guest_copy_seq](bool write_label) {
                CompleteGuestReads(guest_copy_seq);
                if (write_label) {
                    SignalEventWriteEos(packet, completion_rasterizer, fence_token,
                                        completion_trace);
                }
            });
        return;
    }
    CompleteGuestReads(guest_copy_seq);
    SignalEventWriteEos(packet, rasterizer, fence_token, completion_trace);
}

SHAD_NO_INLINE void Liverpool::ProcessEventWriteEop(const PM4CmdEventWriteEop& packet) {
    Common::PerformanceTelemetry::ScopedFastDuration sync_time{
        Common::PerformanceTelemetry::Enabled(),
        Common::PerformanceTelemetry::Counter::GcpSyncPacketNs};
    Common::PerformanceTelemetry::CausalTraceToken completion_trace{};
    if (Common::PerformanceTelemetry::HeavyEnabled()) {
        completion_trace.scope_id = Common::PerformanceTelemetry::NextScopeSeq();
        completion_trace.cause_id = Common::PerformanceTelemetry::NextCauseSeq();
        if (packet.Address<void>() != nullptr) {
            completion_trace.signal_id = Common::PerformanceTelemetry::NextSignalSeq();
        }
        const auto packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
        const u64 label_value = packet.data_lo | (static_cast<u64>(packet.data_hi) << 32);
        const bool irq = packet.int_sel.Value() != InterruptSelect::None;
        Common::PerformanceTelemetry::RecordCompletionScope(
            Common::PerformanceTelemetry::CompletionScopeSample{
                .scope_id = completion_trace.scope_id,
                .cause_id = completion_trace.cause_id,
                .signal_id = completion_trace.signal_id,
                .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                .command_buffer_seq = Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                .first_packet_seq = packet_seq,
                .last_packet_seq = packet_seq,
                .completed_stage_bits = packet.event_control,
                .completed_write_bits = packet.data_control,
                .visible_access_bits = packet.event_type.Value(),
                .cache_action_bits = packet.event_control,
                .label_addr = reinterpret_cast<VAddr>(packet.Address<void>()),
                .label_value = label_value,
                .pm4_digest = HashCombine(packet.event_control, packet.data_control),
                .queue_id = static_cast<u32>(GfxQueueId),
                .engine = static_cast<u16>(Common::PerformanceTelemetry::Pm4Engine::Graphics),
                .kind = Common::PerformanceTelemetry::CompletionScopeKind::EventWriteEop,
                .irq_bits = static_cast<u8>(irq),
                .confidence = 255,
            });
        if (completion_trace.signal_id != 0) {
            Common::PerformanceTelemetry::RecordLogicalSignal(
                Common::PerformanceTelemetry::LogicalSignalSample{
                    .signal_id = completion_trace.signal_id,
                    .scope_id = completion_trace.scope_id,
                    .cause_id = completion_trace.cause_id,
                    .packet_seq = packet_seq,
                    .label_addr = reinterpret_cast<VAddr>(packet.Address<void>()),
                    .value = label_value,
                    .producer_tick = rasterizer ? rasterizer->CurrentTick() : 0,
                    .phase = Common::PerformanceTelemetry::LogicalSignalPhase::Created,
                    .action =
                        Common::PerformanceTelemetry::SignalAction::PublishAfterPhysicalTick,
                    .irq = static_cast<u8>(irq),
                });
        }
    }
    RecordPendingCompletionHazard(
        completion_trace, packet.event_control, packet.data_control,
        static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::ExecutionOrder) |
            static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::MemoryVisibility) |
            (completion_trace.signal_id
                 ? static_cast<u64>(
                       Common::PerformanceTelemetry::SyncRequirement::HostSignalVisibility)
                 : 0) |
            (packet.int_sel.Value() != InterruptSelect::None
                 ? static_cast<u64>(
                       Common::PerformanceTelemetry::SyncRequirement::InterruptPublication)
                 : 0),
        255);
    Common::PerformanceTelemetry::ScopedCausalContext completion_context{completion_trace};
    Common::PerformanceTelemetry::FenceTraceToken fence_token{};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Common::PerformanceTelemetry::HeavyEnabled()) {
        const auto fence_seq = Common::PerformanceTelemetry::NextFenceSeq();
        const auto gen = Common::PerformanceTelemetry::NextFenceGen();
        const auto pkt_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
        const bool irq = (packet.int_sel.Value() != InterruptSelect::None);
        const u64 label_val = packet.data_lo | (static_cast<u64>(packet.data_hi) << 32);
        fence_token = {
            .fence_seq = fence_seq,
            .generation = gen,
            .packet_seq = pkt_seq,
            .label_addr = reinterpret_cast<VAddr>(packet.Address<void>()),
            .label_val = label_val,
        };
        last_sync_packet = {
            .fence_seq = fence_seq,
            .generation = gen,
            .packet_seq = pkt_seq,
            .candidate_id = completion_trace.candidate_id,
            .scope_id = completion_trace.scope_id,
            .cause_id = completion_trace.cause_id,
            .signal_id = completion_trace.signal_id,
            .hazard_id = completion_trace.hazard_id,
            .label_addr = reinterpret_cast<VAddr>(packet.Address<void>()),
            .label_value = label_val,
        };
        Common::PerformanceTelemetry::RecordFenceCreate(Common::PerformanceTelemetry::FenceCreateSample{
            .fence_seq = fence_seq,
            .generation = gen,
            .packet_seq = pkt_seq,
            .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
            .fence_kind = Common::PerformanceTelemetry::FenceKind::EventWriteEop,
            .raw_event_type = static_cast<u32>(packet.event_type.Value()),
            .decoded_event_type = static_cast<u32>(packet.event_type.Value()),
            .command = 0,
            .label_addr = reinterpret_cast<VAddr>(packet.Address<void>()),
            .label_value = label_val,
            .label_size = static_cast<u32>((packet.data_sel.Value() == DataSelect::Data32Low) ? sizeof(u32) : sizeof(u64)),
            .queue_id = static_cast<u32>(GfxQueueId),
            .stage_scope = Common::PerformanceTelemetry::FenceStageScope::Pipe,
            .interrupt_select = static_cast<u32>(irq ? 1 : 0),
            .interrupt_id = static_cast<u32>(Platform::InterruptId::GfxEop),
            .data_select = static_cast<u32>(packet.data_sel.Value()),
            .producer_begin_seq = Common::PerformanceTelemetry::CurrentProducerSeq(),
            .classification_initial = irq ? Common::PerformanceTelemetry::FenceClassification::CpuVisible : Common::PerformanceTelemetry::FenceClassification::Ambiguous,
            .classification_reason_bits = irq ? static_cast<u32>(Common::PerformanceTelemetry::FenceEvidence::InterruptRequested) : 0,
        });
        if (packet.Address<void>() != nullptr) {
            const u64 label_size = static_cast<u64>((packet.data_sel.Value() == DataSelect::Data32Low) ? sizeof(u32) : sizeof(u64));
            Common::PerformanceTelemetry::ArmReadWatchInterest(Common::PerformanceTelemetry::ReadWatchInterest{
                .fence_seq = fence_seq,
                .generation = gen,
                .readback_seq = 0,
                .resource_id = 0,
                .resource_version = 0,
                .guest_addr = reinterpret_cast<VAddr>(packet.Address<void>()),
                .size = label_size,
                .kind = Common::PerformanceTelemetry::ReadWatchKind::Label,
                .fence_packet = pkt_seq,
                .arm_timestamp_ns = Common::PerformanceTelemetry::Timestamp(),
            });
            if (rasterizer && Common::PerformanceTelemetry::IsSyncSemanticProfile()) {
                rasterizer->ArmSemanticReadWatch(reinterpret_cast<VAddr>(packet.Address<void>()), label_size);
            }
        }
    }
#endif
    VideoCore::FlushEpoch::AdvanceSync();
    VideoCore::TextureCache::DownloadDrain drain{};
    if (rasterizer) {
        Common::PerformanceTelemetry::ScopedFenceCause cause{fence_token};
        drain = rasterizer->ProcessDownloadImages(VideoCore::TextureCache::DownloadContext{
            .scope_id = completion_trace.scope_id,
            .cause_id = completion_trace.cause_id,
            .signal_id = completion_trace.signal_id,
            .hazard_id = completion_trace.hazard_id,
            .trigger = VideoCore::TextureCache::DownloadTrigger::EventWriteEop,
            .fence_seq = fence_token.fence_seq,
            .trigger_control = packet.event_control,
            .trigger_data_control = packet.data_control,
        });
    }
    const u64 guest_copy_seq = GuestCopySeq();
    if (rasterizer &&
        (drain.drained != 0 || VideoCore::GpuAuthorityTracker::Instance().MustOrderSignals())) {
        const auto data_sel = packet.data_sel.Value();
        const u64 label_size = data_sel == DataSelect::None        ? 0
                               : data_sel == DataSelect::Data32Low ? sizeof(u32)
                                                                   : sizeof(u64);
        auto* completion_rasterizer = rasterizer;
        PublishCompletionSignal(
            reinterpret_cast<VAddr>(packet.Address<void>()), label_size,
            [packet, completion_rasterizer, fence_token, completion_trace,
             guest_copy_seq](bool write_label) {
                CompleteGuestReads(guest_copy_seq);
                SignalEventWriteEop(packet, completion_rasterizer, fence_token, completion_trace,
                                    write_label);
            });
        return;
    }
    CompleteGuestReads(guest_copy_seq);
    SignalEventWriteEop(packet, rasterizer, fence_token, completion_trace);
}

SHAD_NO_INLINE void Liverpool::ProcessGraphicsEventWrite(
    const PM4Header* header, [[maybe_unused]] u32 count, [[maybe_unused]] u32 ib_depth,
    bool telemetry_enabled, [[maybe_unused]] bool telemetry_detail) {
    const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
    Common::PerformanceTelemetry::CausalTraceToken event_trace{};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (telemetry_detail) {
        event_trace.scope_id = Common::PerformanceTelemetry::NextScopeSeq();
        event_trace.cause_id = Common::PerformanceTelemetry::NextCauseSeq();
        const auto packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
        Common::PerformanceTelemetry::RecordCompletionScope(
            Common::PerformanceTelemetry::CompletionScopeSample{
                .scope_id = event_trace.scope_id,
                .cause_id = event_trace.cause_id,
                .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                .command_buffer_seq =
                    Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                .first_packet_seq = packet_seq,
                .last_packet_seq = packet_seq,
                .completed_stage_bits = event->event_control,
                .completed_write_bits = event->inv_l2,
                .visible_access_bits = static_cast<u64>(event->event_type.Value()),
                .cache_action_bits = event->inv_l2,
                .pm4_digest = HashCombine(event->event_control,
                                         header->type3.NumWords()),
                .queue_id = static_cast<u32>(GfxQueueId),
                .engine = static_cast<u16>(
                    Common::PerformanceTelemetry::Pm4Engine::Graphics),
                .kind = Common::PerformanceTelemetry::CompletionScopeKind::EventWrite,
                .confidence = 160,
            });
        RecordPendingCompletionHazard(
            event_trace, event->event_control, event->inv_l2,
            static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::
                                 ExecutionOrder) |
                (event->inv_l2
                     ? static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::
                                            MemoryVisibility)
                     : 0),
            160);
        Common::PerformanceTelemetry::RecordPm4ControlEnabled(
            Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
            static_cast<u32>(PM4ItOpcode::EventWrite), ib_depth, event->event_control, 0);
        if (count >= 3) {
            Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                static_cast<u32>(PM4ItOpcode::EventWrite), ib_depth, event->address[0],
                event->address[1], 1);
        }
    }
#endif
    LOG_TRACE(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
              magic_enum::enum_name(event->event_type.Value()),
              magic_enum::enum_name(event->event_index.Value()));
    if (event->event_index.Value() == EventIndex::ZpassDone &&
        event->event_type.Value() == EventType::PixelPipeStatDump) {
        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
        const VAddr result_address = static_cast<VAddr>(event->address[0]) |
                                     static_cast<VAddr>(event->address[1]) << 32;
        u64* results = std::bit_cast<u64*>(result_address);
        const s32 counter_pairs = num_counter_pairs;
        PrepareGuestWrite(result_address,
                          static_cast<u64>(counter_pairs) * 2 * sizeof(u64));
        const u64 counter_value = pixel_counter | OcclusionCounterValidMask;
        {
            Common::PerformanceTelemetry::SampledDuration<
                Common::PerformanceTelemetry::TimerSite::EventQueryStores>
                store_duration{telemetry_enabled};
            for (s32 i = 0; i < counter_pairs; ++i, results += 2) {
                *results = counter_value;
            }
        }
        VideoCore::MemoryWriteNotifyResult notify_result{};
        if (rasterizer) {
            Common::PerformanceTelemetry::SampledDuration<
                Common::PerformanceTelemetry::TimerSite::EventQueryNotify>
                notify_duration{telemetry_enabled};
            notify_result = rasterizer->NotifyMemoryWrite(
                result_address, counter_pairs * 2 * sizeof(u64),
                VideoCore::MemoryWriteSource::CommandProcessor);
        }
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::RecordEventQueryEnabled(
                static_cast<u32>(counter_pairs), notify_result.had_active_watches,
                notify_result.matched_pages, notify_result.callbacks);
        }
        pixel_counter += OcclusionCounterStep;
    } else if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
        // TODO: handle proper synchronization, for now signal that update is done
        // immediately
        regs.cp_strmout_cntl.offset_update_done = 1;
    } else if (rasterizer) {
        Common::PerformanceTelemetry::ScopedCausalContext event_context{event_trace};
        rasterizer->FlushCaches(event->event_type.Value());
    }
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb,
                                            u32 ib_depth) {
    FIBER_ENTER(dcb_task_name);
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    [[maybe_unused]] const bool telemetry_detail =
        telemetry_enabled && Common::PerformanceTelemetry::HeavyEnabled();

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb, ib_depth);
        RESUME_GFX(ce_task);
    }
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();
    const bool guest_markers_enabled = rasterizer && EmulatorSettings.IsVkGuestMarkersEnabled();

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    while (!dcb.empty()) {
        if (num_commands.load(std::memory_order_acquire) != 0) [[unlikely]] {
            ProcessCommands();
        }

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 header_raw = header->raw;
        const u32 type = header_raw >> 30;

        if (type != 3) [[unlikely]] {
            dcb = NextNonType3Packet(dcb, type);
            continue;
        }

        {
            const u32 count = ((header_raw >> 16) + 1) & 0x3fff;
            u32 packet_words = count + 1;
            const auto opcode = static_cast<PM4ItOpcode>((header_raw >> 8) & 0xff);
            if (telemetry_enabled) {
                Common::PerformanceTelemetry::CountPm4PacketEnabled(
                    Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                    static_cast<u32>(opcode), ib_depth, reinterpret_cast<uintptr_t>(header),
                    packet_words, header_raw);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                const auto packet_seq = Common::PerformanceTelemetry::NextPacketSeq();
                if (IsSyncPm4Opcode(opcode)) {
                    Common::PerformanceTelemetry::RecordSyncPm4Packet(Common::PerformanceTelemetry::SyncPm4PacketSample{
                        .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                        .cmd_buffer_seq = Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                        .packet_seq = packet_seq,
                        .queue_id = static_cast<u32>(GfxQueueId),
                        .engine = Common::PerformanceTelemetry::Pm4Engine::Graphics,
                        .ib_depth = static_cast<u32>(ib_depth),
                        .opcode = static_cast<u32>(opcode),
                        .packet_addr = static_cast<uintptr_t>(reinterpret_cast<uintptr_t>(header)),
                    });
                }
#endif
            }
            const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth,
                        nop->header.NumWords() != 0 ? nop->data_block[0] : 0, 0);
                }
#endif
                if (nop->header.count.Value() == 0) {
                    break;
                }

                const u32 payload = nop->data_block[0];
                if ((payload & 0xffff0000u) != 0x68750000u) [[likely]] {
                    break;
                }

                switch (payload) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    CompleteGuestReads();
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    const auto* context = reinterpret_cast<const PM4CmdContextControl*>(header);
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, context->load_control.raw,
                        context->shadow_enable.raw);
                }
#endif
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                ++graphics_pipeline_generation;
                ++graphics_state_generation;
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                [[maybe_unused]] const bool changed =
                    WriteGraphicsRegisters(reg_addr, payload, count - 1);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    Common::PerformanceTelemetry::RecordPm4RegisterEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics,
                        static_cast<u32>(opcode), set_data->reg_offset, count - 1, changed);
                }
#endif
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                const u32 word_count = count - 1;
                [[maybe_unused]] bool changed;
                if (word_count == 1) [[likely]] {
                    changed = WriteGraphicsRegistersSmall<1>(reg_addr, payload);
                } else if (word_count == 2) {
                    changed = WriteGraphicsRegistersSmall<2>(reg_addr, payload);
                } else if (word_count == 8) {
                    changed = WriteGraphicsRegisters8(reg_addr, payload);
                } else if (reg_addr > Regs::NumRegs ||
                           word_count > Regs::NumRegs - reg_addr) [[unlikely]] {
                    changed = InvalidGraphicsRegisterRange(reg_addr, word_count);
                } else {
                    changed = WriteGraphicsRegistersSlow(reg_addr, payload, word_count);
                }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    Common::PerformanceTelemetry::RecordPm4RegisterEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics,
                        static_cast<u32>(opcode), set_data->reg_offset, count - 1, changed);
                }
#endif

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                constexpr u32 ColorHintSpan =
                    ContextRegs::CbColor7Cmask - ContextRegs::CbColor0Base;
                if (reg_addr - ContextRegs::CbColor0Base <= ColorHintSpan ||
                    reg_addr == ContextRegs::DbZInfo) [[unlikely]] {
                    HandleContextRegisterHint(reg_addr, header->type3.count, payload);
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                const u32 word_count = count - 1;
                [[maybe_unused]] bool changed{};

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    changed = WriteComputeProgramRegisters(mapped_queues[GfxQueueId].cs_state,
                                                           set_data->reg_offset, payload, word_count,
                                                           telemetry_enabled);
                } else {
                    const u32 reg_addr = Regs::ShRegWordOffset + set_data->reg_offset;
                    if (word_count == 2) [[likely]] {
                        changed = WriteGraphicsRegistersSmall<2>(reg_addr, payload);
                    } else if (word_count == 1) {
                        changed = WriteGraphicsRegistersSmall<1>(reg_addr, payload);
                    } else if (word_count == 4) {
                        changed = WriteGraphicsRegisters4(reg_addr, payload);
                    } else if (word_count == 8) {
                        changed = WriteGraphicsRegisters8(reg_addr, payload);
                    } else if (reg_addr > Regs::NumRegs ||
                               word_count > Regs::NumRegs - reg_addr) [[unlikely]] {
                        changed = InvalidGraphicsRegisterRange(reg_addr, word_count);
                    } else {
                        changed = WriteGraphicsRegistersSlow(reg_addr, payload, word_count);
                    }
                }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    Common::PerformanceTelemetry::RecordPm4RegisterEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics,
                        static_cast<u32>(opcode), set_data->reg_offset, count - 1, changed);
                }
#endif
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                [[maybe_unused]] const bool changed = WriteGraphicsRegisters(
                    Regs::UconfigRegWordOffset + set_data->reg_offset,
                    reinterpret_cast<const u32*>(header + 2), count - 1);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    Common::PerformanceTelemetry::RecordPm4RegisterEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics,
                        static_cast<u32>(opcode), set_data->reg_offset, count - 1, changed);
                }
#endif
                break;
            }
            case PM4ItOpcode::SetPredication: {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, count > 0 ? it_body[0] : 0,
                        count > 1 ? it_body[1] : 0);
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, count > 2 ? it_body[2] : 0,
                        count > 3 ? it_body[3] : 0, 1);
                }
#endif
                if (!warned_set_predication) {
                    warned_set_predication = true;
                    WarnSetPredication();
                }
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, index_type->raw, 0);
                }
#endif
                regs.index_buffer_type.raw = index_type->raw;
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) [[unlikely]] {
                        BeginHostMarker(*rasterizer, cmd_address, "DrawIndex2");
                        rasterizer->Draw(true);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) [[unlikely]] {
                        BeginHostMarker(*rasterizer, cmd_address, "DrawIndexOffset2");
                        rasterizer->Draw(true, draw_index_off->index_offset);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true, draw_index_off->index_offset);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) [[unlikely]] {
                        BeginHostMarker(*rasterizer, cmd_address, "DrawIndexAuto");
                        rasterizer->Draw(false);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(false);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) [[unlikely]] {
                        BeginHostMarker(*rasterizer, cmd_address, "DrawIndirect");
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) [[unlikely]] {
                        BeginHostMarker(*rasterizer, cmd_address, "DrawIndirectMulti");
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) [[unlikely]] {
                        BeginHostMarker(*rasterizer, cmd_address, "DrawIndexIndirect");
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) [[unlikely]] {
                        BeginHostMarker(*rasterizer, cmd_address, "DrawIndexIndirectMulti");
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) [[unlikely]] {
                        BeginHostMarker(*rasterizer, cmd_address,
                                        "DrawIndexIndirectCountMulti");
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) [[unlikely]] {
                        BeginHostMarker(*rasterizer, cmd_address, "DispatchDirect");
                        rasterizer->DispatchDirect();
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchDirect();
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) [[unlikely]] {
                        BeginHostMarker(*rasterizer, cmd_address, "DispatchIndirect");
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                    }
                }
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                if (set_base->base_index != PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable)
                    [[unlikely]] {
                    GraphicsPacketAssertionFailed();
                }
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                ProcessGraphicsEventWrite(header, count, ib_depth, telemetry_enabled,
                                          telemetry_detail);
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, event_eos->event_control,
                        event_eos->cmd_info);
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, event_eos->address_lo,
                        event_eos->data, 1);
                }
#endif
                ProcessEventWriteEos(*event_eos);
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, event_eop->event_control,
                        event_eop->data_control);
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, event_eop->address_lo,
                        event_eop->data_lo, 1);
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, event_eop->data_hi, 0, 2);
                }
#endif
                ProcessEventWriteEop(*event_eop);
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                           dma_data->data, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                           dma_data->NumBytes(), true, false);
                } else if (dma_data->src_sel == DmaDataSrc::Data &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                           dma_data->data, false);
                } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                           dma_data->NumBytes(), false, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                           dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                           false, false);
                } else {
                    InvalidDmaData(*dma_data);
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                if (write_data->dst_sel.Value() != 2 && write_data->dst_sel.Value() != 5)
                    [[unlikely]] {
                    GraphicsPacketAssertionFailed();
                }
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                if (!write_data->wr_one_addr.Value()) {
                    PrepareGuestWrite(address, data_size);
                    std::memcpy(address, write_data->data, data_size);
                    if (rasterizer) {
                        rasterizer->NotifyMemoryWrite(
                            std::bit_cast<VAddr>(address), data_size,
                            VideoCore::MemoryWriteSource::CommandProcessor);
                    }
                } else {
                    UnsupportedWriteDataAddressMode();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                WarnCopyData(*copy_data);
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                PrepareGuestWrite(mem_semaphore->Address<VAddr>(), sizeof(u64));
                if (mem_semaphore->IsSignaling()) {
                    mem_semaphore->Signal();
                } else {
                    while (!mem_semaphore->Signaled()) {
                        YIELD_GFX();
                    }
                    mem_semaphore->Decrement();
                }
                if (rasterizer) {
                    rasterizer->NotifyMemoryWrite(
                        mem_semaphore->Address<VAddr>(), sizeof(u64),
                        VideoCore::MemoryWriteSource::CommandProcessor);
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                const auto* acquire_mem = reinterpret_cast<const PM4CmdAcquireMem*>(header);
                const VAddr base_address =
                    (static_cast<VAddr>(acquire_mem->cp_coher_base_lo) |
                     (static_cast<VAddr>(acquire_mem->cp_coher_base_hi) << 32)) << 8;
                const u64 size =
                    (static_cast<u64>(acquire_mem->cp_coher_size_lo) |
                     (static_cast<u64>(acquire_mem->cp_coher_size_hi) << 32)) << 8;
                Common::PerformanceTelemetry::CausalTraceToken acquire_trace{};
                if (telemetry_enabled) {
                    acquire_trace.candidate_id = last_sync_packet.candidate_id;
                    acquire_trace.scope_id = Common::PerformanceTelemetry::NextScopeSeq();
                    acquire_trace.cause_id = last_sync_packet.cause_id
                                                 ? last_sync_packet.cause_id
                                                 : Common::PerformanceTelemetry::NextCauseSeq();
                    acquire_trace.signal_id = last_sync_packet.signal_id;
                    acquire_trace.hazard_id = last_sync_packet.hazard_id;
                    const auto packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
                    Common::PerformanceTelemetry::RecordCompletionScope(
                        Common::PerformanceTelemetry::CompletionScopeSample{
                            .scope_id = acquire_trace.scope_id,
                            .cause_id = acquire_trace.cause_id,
                            .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                            .command_buffer_seq =
                                Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                            .first_packet_seq = packet_seq,
                            .last_packet_seq = packet_seq,
                            .completed_stage_bits = acquire_mem->cp_coher_cntl,
                            .completed_write_bits = acquire_mem->cp_coher_cntl,
                            .visible_access_bits = acquire_mem->cp_coher_cntl,
                            .cache_action_bits = acquire_mem->cp_coher_cntl,
                            .guest_begin = base_address,
                            .guest_end = base_address + size,
                            .pm4_digest = HashCombine(acquire_mem->cp_coher_cntl,
                                                             acquire_mem->poll_interval),
                            .queue_id = static_cast<u32>(GfxQueueId),
                            .engine = static_cast<u16>(
                                Common::PerformanceTelemetry::Pm4Engine::Graphics),
                            .kind =
                                Common::PerformanceTelemetry::CompletionScopeKind::AcquireMem,
                            .confidence = 255,
                        });
                    RecordPendingCompletionHazard(
                        acquire_trace, acquire_mem->cp_coher_cntl,
                        acquire_mem->cp_coher_cntl,
                        static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::
                                             ExecutionOrder) |
                            static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::
                                                 MemoryVisibility),
                        255);
                }
                Common::PerformanceTelemetry::ScopedCausalContext acquire_context{acquire_trace};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, acquire_mem->cp_coher_cntl,
                        acquire_mem->poll_interval);
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, acquire_mem->cp_coher_size_lo,
                        acquire_mem->cp_coher_size_hi, 1);
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId,
                        static_cast<u32>(opcode), ib_depth, acquire_mem->cp_coher_base_lo,
                        acquire_mem->cp_coher_base_hi, 2);
                    const auto pkt_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
                    Common::PerformanceTelemetry::RecordAcquireMem(Common::PerformanceTelemetry::AcquireMemSample{
                        .packet_seq = pkt_seq,
                        .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                        .queue_id = static_cast<u8>(GfxQueueId),
                        .engine = Common::PerformanceTelemetry::Pm4Engine::Graphics,
                        .cp_coher_cntl = acquire_mem->cp_coher_cntl,
                        .cp_coher_size_lo = acquire_mem->cp_coher_size_lo,
                        .cp_coher_size_hi = acquire_mem->cp_coher_size_hi,
                        .cp_coher_base_lo = acquire_mem->cp_coher_base_lo,
                        .cp_coher_base_hi = acquire_mem->cp_coher_base_hi,
                        .poll_interval = acquire_mem->poll_interval,
                        .decoded_base_addr = base_address,
                        .decoded_size = size,
                        .decoded_cache_flags = acquire_mem->cp_coher_cntl,
                        .previous_wait_seq = last_wait_seq,
                        .shadow_fence_seq = last_sync_packet.fence_seq,
                    });
                }
#endif
                if (rasterizer) {
                    rasterizer->AcquireMemory(acquire_mem->cp_coher_cntl, base_address, size);
                }
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                const auto function = wait_reg_mem->function.Value();
                const u32 mask = wait_reg_mem->mask;
                const u32 reference = wait_reg_mem->ref;
                const auto test_value = [function, mask, reference](u32 value) {
                    return TestWaitValue(value, function, mask, reference);
                };
                Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::WaitRegMemCalls);
                const u64 wait_spin_start = Common::PerformanceTelemetry::Timestamp();
                u64 failed_tests{};
                Common::PerformanceTelemetry::WaitSeq wait_seq{};
                Common::PerformanceTelemetry::FenceSeq matched_fence_seq{};
                Common::PerformanceTelemetry::FenceSeq shadow_fence_seq{};
                Common::PerformanceTelemetry::CausalTraceToken wait_trace{};
                bool wait_producer_correlated{};
                bool wait_required_host_block{};
                u32 val_begin{};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                u64 wait_location{};
                bool used_vo_sleep{};
                Common::PerformanceTelemetry::ShadowBasis shadow_basis{Common::PerformanceTelemetry::ShadowBasis::None};
                if (telemetry_detail) {
                    wait_seq = Common::PerformanceTelemetry::NextWaitSeq();
                    const VAddr wait_addr = (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory)
                                                ? reinterpret_cast<VAddr>(wait_reg_mem->Address<const u32*>())
                                                : static_cast<VAddr>(wait_reg_mem->Reg());
                    if (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
                        val_begin = *wait_reg_mem->Address<const u32*>();
                    }
                    matched_fence_seq = Common::PerformanceTelemetry::MatchFenceForWait(
                        wait_seq, Common::PerformanceTelemetry::CurrentPacketSeq(),
                        Common::PerformanceTelemetry::CurrentFrameSeq(), static_cast<u8>(GfxQueueId),
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, wait_addr, reference, mask,
                        static_cast<u32>(function), 0, 0);

                    if (matched_fence_seq == 0 && last_sync_packet.label_addr == wait_addr && last_sync_packet.fence_seq != 0) {
                        shadow_fence_seq = last_sync_packet.fence_seq;
                        shadow_basis = Common::PerformanceTelemetry::ShadowBasis::AdjacentSameLabel;
                    }

                    Common::PerformanceTelemetry::RecordWaitCreate(Common::PerformanceTelemetry::WaitCreateSample{
                        .wait_seq = wait_seq,
                        .packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq(),
                        .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                        .queue_id = static_cast<u8>(GfxQueueId),
                        .engine = Common::PerformanceTelemetry::Pm4Engine::Graphics,
                        .wait_addr = wait_addr,
                        .ref = reference,
                        .mask = mask,
                        .function = static_cast<u32>(function),
                        .matched_fence_seq = matched_fence_seq,
                        .shadow_fence_seq = shadow_fence_seq,
                        .shadow_basis = shadow_basis,
                        .matcher_status = matched_fence_seq != 0 ? Common::PerformanceTelemetry::CorrelationStatus::Complete : Common::PerformanceTelemetry::CorrelationStatus::Partial,
                    });
                    wait_producer_correlated =
                        last_sync_packet.fence_seq != 0 &&
                        (matched_fence_seq == last_sync_packet.fence_seq ||
                         shadow_fence_seq == last_sync_packet.fence_seq);
                    const Common::PerformanceTelemetry::CausalTraceToken producer_trace{
                        .candidate_id = wait_producer_correlated
                                            ? last_sync_packet.candidate_id
                                            : 0,
                        .scope_id = wait_producer_correlated ? last_sync_packet.scope_id : 0,
                        .cause_id = wait_producer_correlated ? last_sync_packet.cause_id : 0,
                        .signal_id = wait_producer_correlated ? last_sync_packet.signal_id : 0,
                        .hazard_id = wait_producer_correlated ? last_sync_packet.hazard_id : 0,
                    };
                    wait_trace = RecordWaitScope(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId, wait_addr,
                        reference, wait_seq, producer_trace, wait_producer_correlated);
                    last_wait_seq = wait_seq;
                }
#endif
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                VideoCore::FlushEpoch::AdvanceSync();
                if (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
                    const u32* poll_address = wait_reg_mem->Address<const u32*>();
                    [[maybe_unused]] const VAddr wait_addr = reinterpret_cast<VAddr>(poll_address);
                    Common::PerformanceTelemetry::ScopedSemanticReadOrigin wait_origin{
                        Common::PerformanceTelemetry::SemanticReadOrigin::WaitRegMemPoll,
                        matched_fence_seq, 0, 0, 0};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                    wait_location = reinterpret_cast<u64>(poll_address);
#endif
                    bool already_satisfied = test_value(*poll_address);
                    if (!already_satisfied &&
                        VideoCore::GpuAuthorityTracker::Instance().TryPublishReady(
                            Common::PerformanceTelemetry::Counter::SignalsPublishedAtWait)) {
                        // Signals queued behind readbacks that committed meanwhile. On the
                        // console the wait passes once the label is in memory, which is now.
                        already_satisfied = test_value(*poll_address);
                    }
                    bool progress_submit_done = false;
                    if (!already_satisfied) {
                        wait_required_host_block = true;
                        {
                            Common::PerformanceTelemetry::ScopedCausalContext wait_context{
                                wait_trace};
                            if (rasterizer) {
                                rasterizer->Flush(
                                    Common::PerformanceTelemetry::SubmitReason::WaitProgress);
                                progress_submit_done = true;
                            }
                        }
                    }
                    if (!already_satisfied &&
                        vo_port->IsVoLabel(reinterpret_cast<const u64*>(poll_address)) &&
                        num_submits == mapped_queues[GfxQueueId].submits.size()) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                        used_vo_sleep = true;
                        vo_port->WaitVoLabel([&] {
                            Common::PerformanceTelemetry::ScopedSemanticReadOrigin vo_origin{
                                Common::PerformanceTelemetry::SemanticReadOrigin::WaitRegMemPoll,
                                matched_fence_seq, 0, 0, 0};
                            const bool passed = test_value(*poll_address);
                            failed_tests += !passed;
                            return passed;
                        });
#else
                        vo_port->WaitVoLabel([&] { return test_value(*poll_address); });
#endif
                    } else if (!already_satisfied) {
                        WAIT_MEMORY(GfxQueueId, poll_address, test_value(*poll_address),
                                    YIELD_GFX());
                    }
                    if (!already_satisfied) {
                        Common::PerformanceTelemetry::Add(
                            Common::PerformanceTelemetry::Counter::WaitRegMemSpins, failed_tests);
                        Common::PerformanceTelemetry::Add(
                            Common::PerformanceTelemetry::Counter::WaitRegMemSpinNs,
                            Common::PerformanceTelemetry::Timestamp() - wait_spin_start);
                    }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                    if (telemetry_detail) {
                        const u64 wait_end_ts = Common::PerformanceTelemetry::Timestamp();
                        Common::PerformanceTelemetry::RecordCpuToGpuLabelWait(
                            Common::PerformanceTelemetry::CpuToGpuLabelWaitSample{
                                .wait_seq = wait_seq,
                                .frame_id = Common::PerformanceTelemetry::CurrentFrameSeq(),
                                .label_addr = wait_addr,
                                .ref = reference,
                                .mask = mask,
                                .wait_begin = wait_spin_start,
                                .wait_end = wait_end_ts,
                                .duration_ns = wait_end_ts - wait_spin_start,
                                .last_guest_write_ts = 0,
                                .last_guest_write_value = 0,
                                .writer_thread_id = 0,
                                .delta_write_to_wait_complete_ns = 0,
                                .yield_count = static_cast<u32>(failed_tests),
                                .wait_progress_submit_count = progress_submit_done ? 1u : 0u,
                                .gpu_idle_overlap_ns = 0,
                            });
                    }
#endif
                } else {
                    const u32 register_index = wait_reg_mem->Reg();
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                    wait_location = (u64{1} << 63) | register_index;
#endif
                    if (!test_value(regs.reg_array[register_index])) {
                        wait_required_host_block = true;
                    }
                    while (!test_value(regs.reg_array[register_index])) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                        ++failed_tests;
#endif
                        YIELD_GFX();
                    }
                }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (telemetry_detail) {
                    u32 val_end = 0;
                    if (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
                        Common::PerformanceTelemetry::ScopedSemanticReadOrigin end_origin{
                            Common::PerformanceTelemetry::SemanticReadOrigin::WaitRegMemPoll,
                            matched_fence_seq, 0, 0, 0};
                        val_end = *wait_reg_mem->Address<const u32*>();
                    }
                    const u64 wait_end_ts = Common::PerformanceTelemetry::Timestamp();
                    RecordWaitEffect(wait_trace, wait_seq,
                                     wait_required_host_block ? wait_end_ts - wait_spin_start : 0,
                                     !wait_required_host_block, wait_producer_correlated);
                    Common::PerformanceTelemetry::RecordWaitComplete(Common::PerformanceTelemetry::WaitCompleteSample{
                        .wait_seq = wait_seq,
                        .fence_seq = matched_fence_seq,
                        .start_timestamp = wait_spin_start,
                        .end_timestamp = wait_end_ts,
                        .duration_ns = wait_end_ts - wait_spin_start,
                        .spin_iterations = static_cast<u32>(failed_tests),
                        .yield_count = 0,
                        .exit_reason = used_vo_sleep ? 2u : 1u,
                        .value_at_begin = val_begin,
                        .value_at_end = val_end,
                        .gpu_completed_tick_begin = 0,
                        .gpu_completed_tick_end = 0,
                        .shadow_fence_seq = shadow_fence_seq,
                    });
                    Common::PerformanceTelemetry::RecordPm4WaitEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Graphics, GfxQueueId, ib_depth,
                        wait_reg_mem->raw, wait_location, reference, mask,
                        wait_reg_mem->poll_interval, failed_tests, used_vo_sleep);
                    // Disarm semantic read watch on the label address now that the wait is complete.
                    // This prevents stale read-watches from causing page-fault livelocks when the
                    // guest later writes to the same 4KB page.
                    if (wait_location != 0 && matched_fence_seq != 0 &&
                        wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory &&
                        Common::PerformanceTelemetry::IsSyncSemanticProfile()) {
                        Common::PerformanceTelemetry::DisarmLabelReadWatch(
                            matched_fence_seq, wait_location, sizeof(u32),
                            Common::PerformanceTelemetry::SemanticWatchCancelReason::WaitCompleted,
                            [this](VAddr a, u64 s) {
                                if (rasterizer) {
                                    rasterizer->DisarmSemanticReadWatch(a, s);
                                }
                            });
                    }
                }
#endif
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {},
                    ib_depth + 1);
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                if (rasterizer) {
                    rasterizer->CpSync();
                }
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                WarnStrmoutBufferUpdate(*strmout);
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                WarnGetLodStats();
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    WarnReservedCondExec();
                }
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    packet_words += cond_exec->exec_count.Value();
                }
                break;
            }
            default:
                UnknownType3Opcode(opcode, count);
            }
            dcb = NextPacket(dcb, packet_words);
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    FIBER_EXIT;
}

SHAD_NO_INLINE void Liverpool::ProcessComputeReleaseMem(
    const PM4CmdReleaseMem& release_packet, u32 vqid, const u32* queue_pipe_id,
    [[maybe_unused]] u32 ib_depth,
    bool telemetry_enabled, [[maybe_unused]] bool telemetry_detail) {
    Common::PerformanceTelemetry::ScopedFastDuration sync_time{
        telemetry_enabled, Common::PerformanceTelemetry::Counter::GcpSyncPacketNs};
    const auto* release_mem = &release_packet;
    Common::PerformanceTelemetry::CausalTraceToken completion_trace{};
    if (telemetry_enabled) {
        completion_trace.scope_id = Common::PerformanceTelemetry::NextScopeSeq();
        completion_trace.cause_id = Common::PerformanceTelemetry::NextCauseSeq();
        if (release_mem->Address<VAddr>() != 0) {
            completion_trace.signal_id = Common::PerformanceTelemetry::NextSignalSeq();
        }
        const auto packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
        const u64 label_value =
            release_mem->data_lo | (static_cast<u64>(release_mem->data_hi) << 32);
        const bool irq = release_mem->int_sel.Value() != InterruptSelect::None;
        Common::PerformanceTelemetry::RecordCompletionScope(
            Common::PerformanceTelemetry::CompletionScopeSample{
                .scope_id = completion_trace.scope_id,
                .cause_id = completion_trace.cause_id,
                .signal_id = completion_trace.signal_id,
                .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                .command_buffer_seq = Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                .first_packet_seq = packet_seq,
                .last_packet_seq = packet_seq,
                .completed_stage_bits = release_mem->dw1,
                .completed_write_bits = release_mem->dw2,
                .visible_access_bits = release_mem->event_type.Value(),
                .cache_action_bits = release_mem->dw1,
                .label_addr = release_mem->Address<VAddr>(),
                .label_value = label_value,
                .pm4_digest = HashCombine(release_mem->dw1, release_mem->dw2),
                .queue_id = static_cast<u32>(vqid + 1),
                .engine = static_cast<u16>(
                    Common::PerformanceTelemetry::Pm4Engine::Compute),
                .kind = Common::PerformanceTelemetry::CompletionScopeKind::ReleaseMem,
                .irq_bits = static_cast<u8>(irq),
                .confidence = 255,
            });
        if (completion_trace.signal_id != 0) {
            Common::PerformanceTelemetry::RecordLogicalSignal(
                Common::PerformanceTelemetry::LogicalSignalSample{
                    .signal_id = completion_trace.signal_id,
                    .scope_id = completion_trace.scope_id,
                    .cause_id = completion_trace.cause_id,
                    .packet_seq = packet_seq,
                    .label_addr = release_mem->Address<VAddr>(),
                    .value = label_value,
                    .producer_tick = rasterizer ? rasterizer->CurrentTick() : 0,
                    .phase = Common::PerformanceTelemetry::LogicalSignalPhase::Created,
                    .action = Common::PerformanceTelemetry::SignalAction::
                        PublishAfterPhysicalTick,
                    .irq = static_cast<u8>(irq),
                });
        }
    }
    RecordPendingCompletionHazard(
        completion_trace, release_mem->dw1, release_mem->dw2,
        static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::ExecutionOrder) |
            static_cast<u64>(
                Common::PerformanceTelemetry::SyncRequirement::MemoryVisibility) |
            (completion_trace.signal_id
                 ? static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::
                                        HostSignalVisibility)
                 : 0) |
            (release_mem->int_sel.Value() != InterruptSelect::None
                 ? static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::
                                        InterruptPublication)
                 : 0),
        255);
    Common::PerformanceTelemetry::ScopedCausalContext completion_context{
        completion_trace};
    Common::PerformanceTelemetry::FenceTraceToken fence_token{};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (telemetry_detail) {
        Common::PerformanceTelemetry::RecordPm4ControlEnabled(
            Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1,
            static_cast<u32>(PM4ItOpcode::ReleaseMem), ib_depth, release_mem->dw1, release_mem->dw2);
        Common::PerformanceTelemetry::RecordPm4ControlEnabled(
            Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1,
            static_cast<u32>(PM4ItOpcode::ReleaseMem), ib_depth, release_mem->address_lo,
            release_mem->address_hi, 1);
        Common::PerformanceTelemetry::RecordPm4ControlEnabled(
            Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1,
            static_cast<u32>(PM4ItOpcode::ReleaseMem), ib_depth, release_mem->data_lo,
            release_mem->data_hi, 2);

        const auto fence_seq = Common::PerformanceTelemetry::NextFenceSeq();
        const auto gen = Common::PerformanceTelemetry::NextFenceGen();
        const auto pkt_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
        const bool irq = (release_mem->int_sel.Value() != InterruptSelect::None);
        const u64 label_val = release_mem->data_lo | (static_cast<u64>(release_mem->data_hi) << 32);
        fence_token = {
            .fence_seq = fence_seq,
            .generation = gen,
            .packet_seq = pkt_seq,
            .label_addr = release_mem->Address<VAddr>(),
            .label_val = label_val,
        };
        last_sync_packet = {
            .fence_seq = fence_seq,
            .generation = gen,
            .packet_seq = pkt_seq,
            .candidate_id = completion_trace.candidate_id,
            .scope_id = completion_trace.scope_id,
            .cause_id = completion_trace.cause_id,
            .signal_id = completion_trace.signal_id,
            .hazard_id = completion_trace.hazard_id,
            .label_addr = release_mem->Address<VAddr>(),
            .label_value = label_val,
        };
        Common::PerformanceTelemetry::RecordFenceCreate(Common::PerformanceTelemetry::FenceCreateSample{
            .fence_seq = fence_seq,
            .generation = gen,
            .packet_seq = pkt_seq,
            .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
            .fence_kind = Common::PerformanceTelemetry::FenceKind::ReleaseMem,
            .raw_event_type = static_cast<u32>(release_mem->event_type.Value()),
            .decoded_event_type = static_cast<u32>(release_mem->event_type.Value()),
            .command = 0,
            .label_addr = release_mem->Address<VAddr>(),
            .label_value = label_val,
            .label_size = static_cast<u32>((release_mem->data_sel.Value() == DataSelect::Data32Low) ? sizeof(u32) : sizeof(u64)),
            .queue_id = static_cast<u32>(vqid + 1),
            .stage_scope = Common::PerformanceTelemetry::FenceStageScope::Pipe,
            .interrupt_select = static_cast<u32>(irq ? 1 : 0),
            .interrupt_id = static_cast<u32>(*queue_pipe_id),
            .data_select = static_cast<u32>(release_mem->data_sel.Value()),
            .producer_begin_seq = Common::PerformanceTelemetry::CurrentProducerSeq(),
            .classification_initial = irq ? Common::PerformanceTelemetry::FenceClassification::CpuVisible : Common::PerformanceTelemetry::FenceClassification::Ambiguous,
            .classification_reason_bits = irq ? static_cast<u32>(Common::PerformanceTelemetry::FenceEvidence::InterruptRequested) : 0,
        });
        if (release_mem->Address<VAddr>() != 0) {
            const u64 label_size = static_cast<u64>((release_mem->data_sel.Value() == DataSelect::Data32Low) ? sizeof(u32) : sizeof(u64));
            Common::PerformanceTelemetry::ArmReadWatchInterest(Common::PerformanceTelemetry::ReadWatchInterest{
                .fence_seq = fence_seq,
                .generation = gen,
                .readback_seq = 0,
                .resource_id = 0,
                .resource_version = 0,
                .guest_addr = release_mem->Address<VAddr>(),
                .size = label_size,
                .kind = Common::PerformanceTelemetry::ReadWatchKind::Label,
                .fence_packet = pkt_seq,
                .arm_timestamp_ns = Common::PerformanceTelemetry::Timestamp(),
            });
            if (rasterizer && Common::PerformanceTelemetry::IsSyncSemanticProfile()) {
                rasterizer->ArmSemanticReadWatch(release_mem->Address<VAddr>(), label_size);
            }
        }
    }
#endif
    const auto data_sel = release_mem->data_sel.Value();
    VideoCore::FlushEpoch::AdvanceSync();
    VideoCore::TextureCache::DownloadDrain drain{};
    if (rasterizer) {
        Common::PerformanceTelemetry::ScopedFenceCause cause{fence_token};
        drain = rasterizer->ProcessDownloadImages(VideoCore::TextureCache::DownloadContext{
            .scope_id = completion_trace.scope_id,
            .cause_id = completion_trace.cause_id,
            .signal_id = completion_trace.signal_id,
            .hazard_id = completion_trace.hazard_id,
            .trigger = VideoCore::TextureCache::DownloadTrigger::ReleaseMem,
            .fence_seq = fence_token.fence_seq,
            .trigger_control = release_mem->dw1,
            .trigger_data_control = release_mem->dw2,
        });
    }
    const u64 guest_copy_seq = GuestCopySeq();
    // A GDS store records a GPU copy, which only the command processor can do.
    if (rasterizer && data_sel != DataSelect::GdsMemStore &&
        (drain.drained != 0 || VideoCore::GpuAuthorityTracker::Instance().MustOrderSignals())) {
        const PM4CmdReleaseMem packet = *release_mem;
        auto* completion_rasterizer = rasterizer;
        const u32 pipe_id = *queue_pipe_id;
        const u64 label_size = data_sel == DataSelect::None        ? 0
                               : data_sel == DataSelect::Data32Low ? sizeof(u32)
                                                                   : sizeof(u64);
        const bool queued = PublishCompletionSignal(
            packet.Address<VAddr>(), label_size,
            [packet, completion_rasterizer, pipe_id, fence_token, completion_trace,
             guest_copy_seq](bool write_label) {
                CompleteGuestReads(guest_copy_seq);
                SignalReleaseMem(packet, completion_rasterizer, pipe_id, fence_token,
                                 completion_trace, write_label);
            });
        if (queued) {
            // Compute queues often wait for their own release right after it.
            Common::PerformanceTelemetry::Add(
                Common::PerformanceTelemetry::Counter::WritebackFlushes);
            rasterizer->Flush(Common::PerformanceTelemetry::SubmitReason::WritebackReleaseMem);
        }
    } else {
        CompleteGuestReads(guest_copy_seq);
        SignalReleaseMem(*release_mem, rasterizer, *queue_pipe_id, fence_token,
                         completion_trace);
    }
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid, u32 ib_depth) {
    FIBER_ENTER(acb_task_name[vqid]);
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    [[maybe_unused]] const bool telemetry_detail =
        telemetry_enabled && Common::PerformanceTelemetry::HeavyEnabled();
    auto& queue = asc_queues[{vqid}];
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        if (num_commands.load(std::memory_order_acquire) != 0) [[unlikely]] {
            ProcessCommands();
        }

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                PrepareGuestWrite(acb.data(), acb.size_bytes());
                PrepareGuestWrite(queue.read_addr, sizeof(u32));
                *queue.read_addr += acb.size();
                *queue.read_addr &= queue.ring_size_dw - 1;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            if (telemetry_enabled) {
                Common::PerformanceTelemetry::AddEnabled(
                    Common::PerformanceTelemetry::Counter::Pm4Type2Packets, 1);
            }
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                PrepareGuestWrite(queue.read_addr, sizeof(u32));
                *queue.read_addr += next_dw_off;
                *queue.read_addr &= queue.ring_size_dw - 1;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::CountPm4PacketEnabled(
                Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1,
                static_cast<u32>(opcode), ib_depth, reinterpret_cast<uintptr_t>(header),
                next_dw_off, header->raw);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
            const auto packet_seq = Common::PerformanceTelemetry::NextPacketSeq();
            if (IsSyncPm4Opcode(opcode)) {
                Common::PerformanceTelemetry::RecordSyncPm4Packet(Common::PerformanceTelemetry::SyncPm4PacketSample{
                    .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                    .cmd_buffer_seq = Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                    .packet_seq = packet_seq,
                    .queue_id = static_cast<u32>(vqid + 1),
                    .engine = Common::PerformanceTelemetry::Pm4Engine::Compute,
                    .ib_depth = static_cast<u32>(ib_depth),
                    .opcode = static_cast<u32>(opcode),
                    .packet_addr = static_cast<uintptr_t>(reinterpret_cast<uintptr_t>(header)),
                });
            }
#endif
        }

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
            if (telemetry_detail) {
                Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                    Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1,
                    static_cast<u32>(opcode), ib_depth,
                    header->type3.NumWords() != 0 ? it_body[0] : 0, 0);
            }
#endif
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid,
                ib_depth + 1);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            const auto* acquire_mem = reinterpret_cast<const PM4CmdAcquireMem*>(header);
            const VAddr base_address =
                (static_cast<VAddr>(acquire_mem->cp_coher_base_lo) |
                 (static_cast<VAddr>(acquire_mem->cp_coher_base_hi) << 32))
                << 8;
            const u64 size =
                (static_cast<u64>(acquire_mem->cp_coher_size_lo) |
                 (static_cast<u64>(acquire_mem->cp_coher_size_hi) << 32))
                << 8;
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
            if (telemetry_detail) {
                Common::PerformanceTelemetry::CausalTraceToken acquire_trace{
                    .candidate_id = last_sync_packet.candidate_id,
                    .scope_id = Common::PerformanceTelemetry::NextScopeSeq(),
                    .cause_id = last_sync_packet.cause_id
                                    ? last_sync_packet.cause_id
                                    : Common::PerformanceTelemetry::NextCauseSeq(),
                    .signal_id = last_sync_packet.signal_id,
                    .hazard_id = last_sync_packet.hazard_id,
                };
                Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                    Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1,
                    static_cast<u32>(opcode), ib_depth, acquire_mem->cp_coher_cntl,
                    acquire_mem->poll_interval);
                Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                    Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1,
                    static_cast<u32>(opcode), ib_depth, acquire_mem->cp_coher_size_lo,
                    acquire_mem->cp_coher_size_hi, 1);
                Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                    Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1,
                    static_cast<u32>(opcode), ib_depth, acquire_mem->cp_coher_base_lo,
                    acquire_mem->cp_coher_base_hi, 2);
                const auto pkt_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
                Common::PerformanceTelemetry::RecordCompletionScope(
                    Common::PerformanceTelemetry::CompletionScopeSample{
                        .scope_id = acquire_trace.scope_id,
                        .cause_id = acquire_trace.cause_id,
                        .signal_id = acquire_trace.signal_id,
                        .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                        .command_buffer_seq = Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                        .first_packet_seq = pkt_seq,
                        .last_packet_seq = pkt_seq,
                        .completed_stage_bits = acquire_mem->cp_coher_cntl,
                        .completed_write_bits = acquire_mem->cp_coher_cntl,
                        .visible_access_bits = acquire_mem->cp_coher_cntl,
                        .cache_action_bits = acquire_mem->cp_coher_cntl,
                        .guest_begin = base_address,
                        .guest_end = base_address + size,
                        .pm4_digest = HashCombine(acquire_mem->cp_coher_cntl,
                                                         acquire_mem->poll_interval),
                        .queue_id = static_cast<u32>(vqid + 1),
                        .engine = static_cast<u16>(
                            Common::PerformanceTelemetry::Pm4Engine::Compute),
                        .kind = Common::PerformanceTelemetry::CompletionScopeKind::AcquireMem,
                        .confidence = 192,
                    });
                Common::PerformanceTelemetry::RecordAcquireMem(Common::PerformanceTelemetry::AcquireMemSample{
                    .packet_seq = pkt_seq,
                    .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                    .queue_id = static_cast<u8>(vqid + 1),
                    .engine = Common::PerformanceTelemetry::Pm4Engine::Compute,
                    .cp_coher_cntl = acquire_mem->cp_coher_cntl,
                    .cp_coher_size_lo = acquire_mem->cp_coher_size_lo,
                    .cp_coher_size_hi = acquire_mem->cp_coher_size_hi,
                    .cp_coher_base_lo = acquire_mem->cp_coher_base_lo,
                    .cp_coher_base_hi = acquire_mem->cp_coher_base_hi,
                    .poll_interval = acquire_mem->poll_interval,
                    .decoded_base_addr = base_address,
                    .decoded_size = size,
                    .decoded_cache_flags = acquire_mem->cp_coher_cntl,
                    .previous_wait_seq = last_wait_seq,
                    .shadow_fence_seq = last_sync_packet.fence_seq,
                });
                RecordPendingCompletionHazard(
                    acquire_trace, acquire_mem->cp_coher_cntl, acquire_mem->cp_coher_cntl,
                    static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::
                                         ExecutionOrder) |
                        static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::
                                             MemoryVisibility),
                    192);
                Common::PerformanceTelemetry::RecordHazardResolution(
                    Common::PerformanceTelemetry::HazardResolutionSample{
                        .hazard_id = acquire_trace.hazard_id,
                        .cause_id = acquire_trace.cause_id,
                        .candidate_id = acquire_trace.candidate_id,
                        .guest_begin = base_address,
                        .guest_end = base_address + size,
                        .src_stage = static_cast<u64>(vk::PipelineStageFlagBits2::eComputeShader),
                        .src_access = static_cast<u64>(vk::AccessFlagBits2::eShaderWrite),
                        .dst_stage = static_cast<u64>(vk::PipelineStageFlagBits2::eComputeShader),
                        .dst_access = static_cast<u64>(vk::AccessFlagBits2::eShaderRead) |
                                      static_cast<u64>(vk::AccessFlagBits2::eShaderWrite),
                        .sync_requirement_bits = static_cast<u64>(
                            Common::PerformanceTelemetry::SyncRequirement::ExecutionOrder),
                        .resolution = Common::PerformanceTelemetry::HazardResolutionKind::
                            ImplicitDependency,
                        .avoidability = Common::PerformanceTelemetry::Avoidability::ProvenRequired,
                        .confidence = 192,
                    });
            }
#endif
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto* payload = reinterpret_cast<const u32*>(header + 2);
            const u32 word_count = header->type3.NumWords() - 1;
            [[maybe_unused]] bool changed{};

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                changed = WriteComputeProgramRegisters(mapped_queues[vqid + 1].cs_state,
                                                       set_data->reg_offset, payload, word_count,
                                                       telemetry_enabled);
            } else {
                changed = WriteGraphicsRegisters(Regs::ShRegWordOffset + set_data->reg_offset,
                                                 payload, word_count);
            }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
            if (telemetry_detail) {
                Common::PerformanceTelemetry::RecordPm4RegisterEnabled(
                    Common::PerformanceTelemetry::Pm4Engine::Compute,
                    static_cast<u32>(opcode), set_data->reg_offset,
                    header->type3.NumWords() - 1, changed);
            }
#endif
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) [[unlikely]] {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchDirect", vqid, cmd_address));
                    rasterizer->DispatchDirect();
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchDirect();
                }
            }
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) [[unlikely]] {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchIndirect", vqid, cmd_address));
                    rasterizer->DispatchIndirect(ib_address, 0, size);
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchIndirect(ib_address, 0, size);
                }
            }
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (!write_data->wr_one_addr.Value()) {
                PrepareGuestWrite(write_data->Address<void*>(), data_size);
                std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
                if (rasterizer) {
                    rasterizer->NotifyMemoryWrite(
                        std::bit_cast<VAddr>(write_data->Address<void*>()), data_size,
                        VideoCore::MemoryWriteSource::CommandProcessor);
                }
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            PrepareGuestWrite(mem_semaphore->Address<VAddr>(), sizeof(u64));
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
            } else {
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                mem_semaphore->Decrement();
            }
            if (rasterizer) {
                rasterizer->NotifyMemoryWrite(
                    mem_semaphore->Address<VAddr>(), sizeof(u64),
                    VideoCore::MemoryWriteSource::CommandProcessor);
            }
            break;
        }
            case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            const auto function = wait_reg_mem->function.Value();
            const u32 mask = wait_reg_mem->mask;
            const u32 reference = wait_reg_mem->ref;
            const auto test_value = [function, mask, reference](u32 value) {
                return TestWaitValue(value, function, mask, reference);
            };
            const u64 wait_spin_start = Common::PerformanceTelemetry::Timestamp();
            u64 failed_tests{};
            Common::PerformanceTelemetry::WaitSeq wait_seq{};
            Common::PerformanceTelemetry::FenceSeq matched_fence_seq{};
            Common::PerformanceTelemetry::FenceSeq shadow_fence_seq{};
            Common::PerformanceTelemetry::CausalTraceToken wait_trace{};
            bool wait_producer_correlated{};
            bool wait_required_host_block{};
            u32 val_begin{};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
            u64 wait_location{};
            Common::PerformanceTelemetry::ShadowBasis shadow_basis{Common::PerformanceTelemetry::ShadowBasis::None};
            if (telemetry_detail) {
                wait_seq = Common::PerformanceTelemetry::NextWaitSeq();
                const VAddr wait_addr = (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory)
                                            ? reinterpret_cast<VAddr>(wait_reg_mem->Address<const u32*>())
                                            : static_cast<VAddr>(wait_reg_mem->Reg());
                if (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
                    val_begin = *wait_reg_mem->Address<const u32*>();
                }
                matched_fence_seq = Common::PerformanceTelemetry::MatchFenceForWait(
                    wait_seq, Common::PerformanceTelemetry::CurrentPacketSeq(),
                    Common::PerformanceTelemetry::CurrentFrameSeq(), static_cast<u8>(vqid + 1),
                    Common::PerformanceTelemetry::Pm4Engine::Compute, wait_addr, reference, mask,
                    static_cast<u32>(function), 0, 0);

                if (matched_fence_seq == 0 && last_sync_packet.label_addr == wait_addr && last_sync_packet.fence_seq != 0) {
                    shadow_fence_seq = last_sync_packet.fence_seq;
                    shadow_basis = Common::PerformanceTelemetry::ShadowBasis::AdjacentSameLabel;
                }

                Common::PerformanceTelemetry::RecordWaitCreate(Common::PerformanceTelemetry::WaitCreateSample{
                    .wait_seq = wait_seq,
                    .packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq(),
                    .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                    .queue_id = static_cast<u8>(vqid + 1),
                    .engine = Common::PerformanceTelemetry::Pm4Engine::Compute,
                    .wait_addr = wait_addr,
                    .ref = reference,
                    .mask = mask,
                    .function = static_cast<u32>(function),
                    .matched_fence_seq = matched_fence_seq,
                    .shadow_fence_seq = shadow_fence_seq,
                    .shadow_basis = shadow_basis,
                    .matcher_status = matched_fence_seq != 0 ? Common::PerformanceTelemetry::CorrelationStatus::Complete : Common::PerformanceTelemetry::CorrelationStatus::Partial,
                });
                wait_producer_correlated =
                    last_sync_packet.fence_seq != 0 &&
                    (matched_fence_seq == last_sync_packet.fence_seq ||
                     shadow_fence_seq == last_sync_packet.fence_seq);
                const Common::PerformanceTelemetry::CausalTraceToken producer_trace{
                    .candidate_id = wait_producer_correlated ? last_sync_packet.candidate_id : 0,
                    .scope_id = wait_producer_correlated ? last_sync_packet.scope_id : 0,
                    .cause_id = wait_producer_correlated ? last_sync_packet.cause_id : 0,
                    .signal_id = wait_producer_correlated ? last_sync_packet.signal_id : 0,
                    .hazard_id = wait_producer_correlated ? last_sync_packet.hazard_id : 0,
                };
                wait_trace = RecordWaitScope(
                    Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1, wait_addr,
                    reference, wait_seq, producer_trace, wait_producer_correlated);
                last_wait_seq = wait_seq;
            }
#endif
            VideoCore::FlushEpoch::AdvanceSync();
            if (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
                Common::PerformanceTelemetry::ScopedSemanticReadOrigin wait_origin{
                    Common::PerformanceTelemetry::SemanticReadOrigin::WaitRegMemPoll,
                    matched_fence_seq, 0, 0, 0};
                const u32* poll_address = wait_reg_mem->Address<const u32*>();
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                wait_location = reinterpret_cast<u64>(poll_address);
#endif
                bool already_satisfied = test_value(*poll_address);
                if (!already_satisfied &&
                    VideoCore::GpuAuthorityTracker::Instance().TryPublishReady(
                        Common::PerformanceTelemetry::Counter::SignalsPublishedAtWait)) {
                    already_satisfied = test_value(*poll_address);
                }
                if (!already_satisfied) {
                    wait_required_host_block = true;
                    {
                        Common::PerformanceTelemetry::ScopedCausalContext wait_context{wait_trace};
                        if (rasterizer) {
                            rasterizer->Flush(
                                Common::PerformanceTelemetry::SubmitReason::WaitProgress);
                        }
                    }
                    WAIT_MEMORY(vqid + 1, poll_address, test_value(*poll_address),
                                YIELD_ASC(vqid));
                }
            } else {
                const u32 register_index = wait_reg_mem->Reg();
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                wait_location = (u64{1} << 63) | register_index;
#endif
                if (!test_value(regs.reg_array[register_index])) {
                    wait_required_host_block = true;
                }
                while (!test_value(regs.reg_array[register_index])) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                    ++failed_tests;
#endif
                    YIELD_ASC(vqid);
                }
            }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
            if (telemetry_detail) {
                u32 val_end = 0;
                if (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
                    Common::PerformanceTelemetry::ScopedSemanticReadOrigin end_origin{
                        Common::PerformanceTelemetry::SemanticReadOrigin::WaitRegMemPoll,
                        matched_fence_seq, 0, 0, 0};
                    val_end = *wait_reg_mem->Address<const u32*>();
                }
                const u64 wait_end_ts = Common::PerformanceTelemetry::Timestamp();
                RecordWaitEffect(wait_trace, wait_seq,
                                 wait_required_host_block ? wait_end_ts - wait_spin_start : 0,
                                 !wait_required_host_block, wait_producer_correlated);
                Common::PerformanceTelemetry::RecordWaitComplete(Common::PerformanceTelemetry::WaitCompleteSample{
                    .wait_seq = wait_seq,
                    .fence_seq = matched_fence_seq,
                    .start_timestamp = wait_spin_start,
                    .end_timestamp = wait_end_ts,
                    .duration_ns = wait_end_ts - wait_spin_start,
                    .spin_iterations = static_cast<u32>(failed_tests),
                    .yield_count = 0,
                    .exit_reason = 1u,
                    .value_at_begin = val_begin,
                    .value_at_end = val_end,
                    .gpu_completed_tick_begin = 0,
                    .gpu_completed_tick_end = 0,
                    .shadow_fence_seq = shadow_fence_seq,
                });
                Common::PerformanceTelemetry::RecordPm4WaitEnabled(
                    Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1, ib_depth,
                    wait_reg_mem->raw, wait_location, reference, mask,
                    wait_reg_mem->poll_interval, failed_tests, false);
                if (wait_location != 0 && matched_fence_seq != 0 &&
                    wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory &&
                    Common::PerformanceTelemetry::IsSyncSemanticProfile()) {
                    Common::PerformanceTelemetry::DisarmLabelReadWatch(
                        matched_fence_seq, wait_location, sizeof(u32),
                        Common::PerformanceTelemetry::SemanticWatchCancelReason::WaitCompleted,
                        [this](VAddr a, u64 s) {
                            if (rasterizer) {
                                rasterizer->DisarmSemanticReadWatch(a, s);
                            }
                        });
                }
            }
#endif
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            ProcessComputeReleaseMem(*reinterpret_cast<const PM4CmdReleaseMem*>(header),
                                     vqid, &queue.pipe_id, ib_depth, telemetry_enabled,
                                     telemetry_detail);
            break;
        }
        case PM4ItOpcode::EventWrite: {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
            if (telemetry_detail) {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                Common::PerformanceTelemetry::CausalTraceToken event_trace{
                    .scope_id = Common::PerformanceTelemetry::NextScopeSeq(),
                    .cause_id = Common::PerformanceTelemetry::NextCauseSeq(),
                };
                const auto packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq();
                Common::PerformanceTelemetry::RecordCompletionScope(
                    Common::PerformanceTelemetry::CompletionScopeSample{
                        .scope_id = event_trace.scope_id,
                        .cause_id = event_trace.cause_id,
                        .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                        .command_buffer_seq =
                            Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                        .first_packet_seq = packet_seq,
                        .last_packet_seq = packet_seq,
                        .completed_stage_bits = event->event_control,
                        .completed_write_bits = event->inv_l2,
                        .visible_access_bits = static_cast<u64>(event->event_type.Value()),
                        .cache_action_bits = event->inv_l2,
                        .pm4_digest = HashCombine(event->event_control,
                                                 header->type3.NumWords()),
                        .queue_id = static_cast<u32>(vqid + 1),
                        .engine = static_cast<u16>(
                            Common::PerformanceTelemetry::Pm4Engine::Compute),
                        .kind = Common::PerformanceTelemetry::CompletionScopeKind::EventWrite,
                        .confidence = 96,
                    });
                RecordPendingCompletionHazard(
                    event_trace, event->event_control, event->inv_l2,
                    static_cast<u64>(
                        Common::PerformanceTelemetry::SyncRequirement::ExecutionOrder) |
                        (event->inv_l2
                             ? static_cast<u64>(Common::PerformanceTelemetry::SyncRequirement::
                                                    MemoryVisibility)
                             : 0),
                    96);
                Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                    Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1,
                    static_cast<u32>(opcode), ib_depth, event->event_control, 0);
                if (header->type3.NumWords() >= 3) {
                    Common::PerformanceTelemetry::RecordPm4ControlEnabled(
                        Common::PerformanceTelemetry::Pm4Engine::Compute, vqid + 1,
                        static_cast<u32>(opcode), ib_depth, event->address[0],
                        event->address[1], 1);
                }
            }
#endif
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        if constexpr (!is_indirect) {
            PrepareGuestWrite(acb.data(), static_cast<u64>(next_dw_off) * sizeof(u32));
            PrepareGuestWrite(queue.read_addr, sizeof(u32));
        }
        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr &= queue.ring_size_dw - 1;
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    [[maybe_unused]] const bool telemetry_detail =
        telemetry_enabled && Common::PerformanceTelemetry::HeavyEnabled();
    auto task = ProcessGraphics(dcb, ccb);
    task.handle.promise().telemetry_ready_since_ns =
        telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    const u32 queue_depth = ++num_submits;
    constexpr u64 QueueBit = 1ULL << GfxQueueId;
    if ((blocked_queue_mask.load(std::memory_order_acquire) & QueueBit) == 0) {
        ready_queue_mask.fetch_or(QueueBit, std::memory_order_release);
    }
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::GfxSubmits, 1);
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::DcbBytes, dcb.size_bytes());
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::CcbBytes, ccb.size_bytes());
        Common::PerformanceTelemetry::ObserveMaxEnabled(
            Common::PerformanceTelemetry::Counter::SubmitQueueDepthMax, queue_depth);
    }
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    [[maybe_unused]] const bool telemetry_detail =
        telemetry_enabled && Common::PerformanceTelemetry::HeavyEnabled();
    const auto& task = ProcessCompute(acb, vqid);
    task.handle.promise().telemetry_ready_since_ns =
        telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    const u32 queue_depth = ++num_submits;
    const u64 queue_bit = 1ULL << gnm_vqid;
    if ((blocked_queue_mask.load(std::memory_order_acquire) & queue_bit) == 0) {
        ready_queue_mask.fetch_or(queue_bit, std::memory_order_release);
    }
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::AscSubmits, 1);
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::AcbBytes, acb.size_bytes());
        Common::PerformanceTelemetry::ObserveMaxEnabled(
            Common::PerformanceTelemetry::Counter::SubmitQueueDepthMax, queue_depth);
    }
    submit_cv.notify_one();
}

} // namespace AmdGpu
