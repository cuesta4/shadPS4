// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/amdgpu/gpu_command_processor.h"
#include "video_core/amdgpu/liverpool.h"

namespace AmdGpu {

Liverpool::Liverpool()
    : processor{std::make_unique<GpuCommandProcessor>()},
      asc_queues{processor->AscQueues()} {}

Liverpool::~Liverpool() = default;

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    processor->SubmitGfx(dcb, ccb);
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    processor->SubmitAsc(gnm_vqid, acb);
}

void Liverpool::SubmitDone() noexcept {
    processor->SubmitDone();
}

void Liverpool::WaitGpuIdle() noexcept {
    processor->WaitGpuIdle();
}

bool Liverpool::IsGpuIdle() const {
    return processor->IsGpuIdle();
}

void Liverpool::SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
    processor->SetVoPort(port);
}

void Liverpool::BindCommandSink(VideoCore::GpuCommandSink* command_sink) {
    processor->BindCommandSink(command_sink);
}

void Liverpool::ReserveCopyBufferSpace() {
    processor->ReserveCopyBufferSpace();
}

void Liverpool::ExecuteOnGpuThread(Common::UniqueFunction<void> command) {
    processor->Enqueue(std::move(command), true);
}

void Liverpool::Enqueue(Common::UniqueFunction<void> command, bool wait) {
    processor->Enqueue(std::move(command), wait);
}

} // namespace AmdGpu
