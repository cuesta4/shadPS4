// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace AmdGpu {
struct ComputeProgram;
union Regs;
} // namespace AmdGpu

namespace Shader {
struct Info;
struct ShaderInvocationData;
}

namespace Vulkan {

class VulkanCommandRecorder;
}

namespace VideoCore {
class BufferCache;
}

namespace Vulkan {

struct ShaderHleServices {
    VideoCore::BufferCache& buffers;
    VulkanCommandRecorder& recorder;
};

/// Attempts to execute a shader using HLE if possible.
bool ExecuteShaderHLE(const Shader::Info& info, const Shader::ShaderInvocationData& invocation,
                      const AmdGpu::ComputeProgram& cs_program,
                      const ShaderHleServices& services);

} // namespace Vulkan
