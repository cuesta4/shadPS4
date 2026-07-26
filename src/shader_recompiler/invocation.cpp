// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/invocation.h"

#include <algorithm>

#include "shader_recompiler/info.h"

namespace Shader {

ShaderInvocationData ShaderInvocationData::FromParams(const Info& analysis,
                                                      const ShaderParams& params) {
    ShaderInvocationData invocation{};
    ASSERT(params.user_data.size() <= invocation.user_data.size());
    std::copy(params.user_data.begin(), params.user_data.end(), invocation.user_data.begin());
    invocation.pgm_base = params.Base();
    invocation.flattened_ud_buf.resize(analysis.srt_info.flattened_bufsize_dw);
    if (!invocation.flattened_ud_buf.empty()) {
        const size_t copy_dwords =
            std::min(invocation.user_data.size(), invocation.flattened_ud_buf.size());
        std::copy_n(invocation.user_data.begin(), copy_dwords,
                    invocation.flattened_ud_buf.begin());
        if (analysis.srt_info.walker_func) {
            analysis.srt_info.walker_func(invocation.user_data.data(),
                                          invocation.flattened_ud_buf.data());
        }
    }
    return invocation;
}

ShaderInvocationData ShaderInvocationData::FromCompilerInfo(const Info& info) {
    ShaderInvocationData invocation{};
    ASSERT(info.user_data.size() <= invocation.user_data.size());
    std::copy(info.user_data.begin(), info.user_data.end(), invocation.user_data.begin());
    invocation.flattened_ud_buf.assign(info.flattened_ud_buf.begin(),
                                       info.flattened_ud_buf.end());
    invocation.pgm_base = info.pgm_base;
    invocation.compiler_resources = std::make_unique<CompilerResolvedResources>();
    auto& resources = *invocation.compiler_resources;
    resources.buffers.assign(info.resolved_buffers.begin(), info.resolved_buffers.end());
    resources.images.assign(info.resolved_images.begin(), info.resolved_images.end());
    resources.samplers.assign(info.resolved_samplers.begin(), info.resolved_samplers.end());
    resources.fmasks.assign(info.resolved_fmasks.begin(), info.resolved_fmasks.end());
    resources.vertex_buffers.assign(info.resolved_vertex_buffers.begin(),
                                    info.resolved_vertex_buffers.end());
    return invocation;
}

void ShaderInvocationData::PushUd(const Info& analysis, Backend::Bindings& bindings,
                                  PushData& push) const {
    u32 mask = analysis.ud_mask.mask;
    while (mask) {
        const u32 index = std::countr_zero(mask);
        ASSERT(bindings.user_data < NUM_USER_DATA_REGS && index < NUM_USER_DATA_REGS);
        mask &= ~(1U << index);
        push.ud_regs[bindings.user_data++] = user_data[index];
    }
}

void ShaderInvocationData::ReadTessConstantBuffer(
    const Info& analysis, TessellationDataConstantBuffer& tess_constants) const {
    ASSERT(analysis.tess_consts_dword_offset >= 0);
    const auto buffer = ReadUdReg<AmdGpu::Buffer>(
        static_cast<u32>(analysis.tess_consts_ptr_base),
        static_cast<u32>(analysis.tess_consts_dword_offset));
    const VAddr address = buffer.base_address;
    std::memcpy(&tess_constants,
                reinterpret_cast<const TessellationDataConstantBuffer*>(address),
                sizeof(tess_constants));
}

} // namespace Shader
