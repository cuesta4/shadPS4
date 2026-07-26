// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/gpu_commands/captured_state.h"

namespace Vulkan {

enum class GraphicsOperation {
    Draw,
    EliminateFastClear,
    FmaskDecompress,
    Resolve,
    SkipPrimitiveNone,
    DepthStencilCopy,
};

struct GraphicsOperationClassification {
    GraphicsOperation operation{GraphicsOperation::Draw};
    bool copy_depth{};
    bool copy_stencil{};
};

[[nodiscard]] GraphicsOperationClassification ClassifyGraphicsOperation(
    const VideoCore::CapturedGraphicsState& state) noexcept;

} // namespace Vulkan
