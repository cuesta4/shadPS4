// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/gpu_commands/captured_state.h"

namespace VideoCore {

struct DrawCommand {
    CommandProvenance provenance{};
    CapturedGraphicsState state{};
    bool is_indexed{};
    u32 index_offset{};
};

struct DrawIndirectCommand {
    CommandProvenance provenance{};
    CapturedGraphicsState state{};
    bool is_indexed{};
    VAddr argument_address{};
    u32 offset{};
    u32 stride{};
    u32 max_count{};
    VAddr count_address{};
};

struct DispatchDirectCommand {
    CommandProvenance provenance{};
    CapturedComputeState state{};
};

struct DispatchIndirectCommand {
    CommandProvenance provenance{};
    CapturedComputeState state{};
    VAddr argument_address{};
    u32 offset{};
    u32 size{};
};

struct FillBufferCommand {
    CommandProvenance provenance{};
    VAddr address{};
    u32 size{};
    u32 value{};
    bool is_gds{};
};

struct CopyBufferCommand {
    CommandProvenance provenance{};
    VAddr destination{};
    VAddr source{};
    u32 size{};
    bool destination_is_gds{};
    bool source_is_gds{};
};

} // namespace VideoCore
