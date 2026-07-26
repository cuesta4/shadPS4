// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <semaphore>
#include <span>
#include <thread>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/gpu_queue.h"
#include "video_core/amdgpu/register_file.h"
#include "video_core/gpu_commands/commands.h"

namespace VideoCore {
class GpuCommandSink;
}

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

/**
 * Serial authority for GPU submissions and PM4 interpretation.
 *
 * Liverpool owns this service and exposes only the public device API. Keeping the authoritative
 * register file, queues, interpreters, and arbitration together preserves PM4 ordering while
 * removing parsing and thread-loop responsibilities from the public device facade.
 */
class GpuCommandProcessor {
public:
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;
    static constexpr u32 NumComputePipes = 7u;
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u);

    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    GpuCommandProcessor();
    ~GpuCommandProcessor();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);
    void SubmitDone() noexcept;
    void WaitGpuIdle() noexcept;
    [[nodiscard]] bool IsGpuIdle() const noexcept;
    void ReserveCopyBufferSpace();
    void Enqueue(Common::UniqueFunction<void> command, bool wait);

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) noexcept {
        vo_port = port;
    }

    void BindCommandSink(VideoCore::GpuCommandSink* sink) noexcept {
        command_sink = sink;
    }

    [[nodiscard]] Common::SlotVector<AscQueueInfo>& AscQueues() noexcept {
        return asc_queues;
    }

private:
    using Task = GpuTask;
    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;

    [[nodiscard]] VideoCore::CommandProvenance CaptureProvenance(
        VAddr packet_address) noexcept {
        return {
            .sequence = ++command_sequence,
            .submission = current_submission,
            .packet_address = packet_address,
            .queue_id = static_cast<u32>(curr_qid),
            .packet_offset_dw = current_packet_offset_dw,
            .opcode = current_packet_opcode,
        };
    }

    [[nodiscard]] VideoCore::CapturedGraphicsState CaptureGraphicsState() const noexcept {
        return register_file.CaptureGraphics();
    }

    [[nodiscard]] VideoCore::CapturedComputeState CaptureComputeState() const noexcept {
        auto state = register_file.CaptureCompute(mapped_queues[curr_qid].compute_state);
        state.dispatch = {
            .dim_x = state.program.dim_x,
            .dim_y = state.program.dim_y,
            .dim_z = state.program.dim_z,
            .shared_memory_size = state.program.SharedMemSize(),
        };
        return state;
    }

    [[nodiscard]] ComputeProgram& GetCsRegs() noexcept {
        return mapped_queues[curr_qid].compute_state;
    }

    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessCeUpdate(std::span<const u32> ccb);
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid);
    void ProcessCommands();
    void Process(std::stop_token stoken);

    GpuRegisterFile register_file{};
    Regs& regs{register_file.Raw()};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_extent{
        register_file.ExtentHints().color};
    CbDbExtent& last_db_extent{register_file.ExtentHints().depth};
    GpuQueueManager<NumTotalQueues> mapped_queues{};
    u32 num_mapped_queues{1u};
    Common::SlotVector<AscQueueInfo> asc_queues{};

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};

    struct ConstantEngine {
        void Reset() {
            ce_count = 0;
            de_count = 0;
            ce_compare_count = 0;
        }

        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_compare_count{};
        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    VideoCore::GpuCommandSink* command_sink{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    std::jthread process_thread{};
    std::atomic<u32> num_submits{};
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::thread::id gpu_id;
    s32 curr_qid{-1};
    u64 command_sequence{};
    u64 current_submission{};
    u32 current_packet_offset_dw{};
    u32 current_packet_opcode{};
    std::atomic<u64> submission_sequence{};
};

} // namespace AmdGpu
