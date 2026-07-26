// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

#include "common/types.h"
#include "video_core/gpu_commands/commands.h"

namespace VideoCore {

class GpuCommandSink {
public:
    virtual ~GpuCommandSink() = default;

    virtual void Execute(const DrawCommand& command) = 0;
    virtual void Execute(const DrawIndirectCommand& command) = 0;
    virtual void Execute(const DispatchDirectCommand& command) = 0;
    virtual void Execute(const DispatchIndirectCommand& command) = 0;
    virtual void Execute(const FillBufferCommand& command) = 0;
    virtual void Execute(const CopyBufferCommand& command) = 0;

    virtual u32 ReadDataFromGds(u32 offset) = 0;
    virtual void ProcessDownloadImages() = 0;
    virtual void CpSync() = 0;
    virtual u64 Flush() = 0;
    virtual void Finish() = 0;
    virtual void OnSubmit() = 0;

    virtual void ScopeMarkerBegin(std::string_view label, bool from_guest = false) = 0;
    virtual void ScopeMarkerEnd(bool from_guest = false) = 0;
    virtual void ScopedMarkerInsertColor(std::string_view label, u32 color,
                                         bool from_guest = false) = 0;
};

class GpuCommandSinkBinder {
public:
    virtual ~GpuCommandSinkBinder() = default;
    virtual void BindCommandSink(GpuCommandSink* command_sink) = 0;
};

} // namespace VideoCore
