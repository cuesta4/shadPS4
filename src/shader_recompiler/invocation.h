// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cstring>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include "common/assert.h"
#include "common/types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/frontend/tessellation.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/params.h"
#include "shader_recompiler/resource.h"

namespace Shader {

struct Info;

struct CompilerResolvedResources {
    std::vector<AmdGpu::Buffer> buffers;
    std::vector<AmdGpu::Image> images;
    std::vector<AmdGpu::Sampler> samplers;
    std::vector<AmdGpu::Image> fmasks;
    std::vector<AmdGpu::Buffer> vertex_buffers;
};

struct ShaderInvocationData {
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    std::vector<u32> flattened_ud_buf;
    VAddr pgm_base{};

    // Runtime invocations do not carry vector headers for compiler-only resolved descriptors.
    std::unique_ptr<CompilerResolvedResources> compiler_resources;

    static ShaderInvocationData FromParams(const Info& analysis, const ShaderParams& params);
    static ShaderInvocationData FromCompilerInfo(const Info& info);

    [[nodiscard]] std::span<const AmdGpu::Buffer> ResolvedBuffers() const noexcept {
        return compiler_resources ? std::span{compiler_resources->buffers}
                                  : std::span<const AmdGpu::Buffer>{};
    }

    [[nodiscard]] std::span<const AmdGpu::Image> ResolvedImages() const noexcept {
        return compiler_resources ? std::span{compiler_resources->images}
                                  : std::span<const AmdGpu::Image>{};
    }

    [[nodiscard]] std::span<const AmdGpu::Sampler> ResolvedSamplers() const noexcept {
        return compiler_resources ? std::span{compiler_resources->samplers}
                                  : std::span<const AmdGpu::Sampler>{};
    }

    [[nodiscard]] std::span<const AmdGpu::Image> ResolvedFmasks() const noexcept {
        return compiler_resources ? std::span{compiler_resources->fmasks}
                                  : std::span<const AmdGpu::Image>{};
    }

    [[nodiscard]] std::span<const AmdGpu::Buffer> ResolvedVertexBuffers() const noexcept {
        return compiler_resources ? std::span{compiler_resources->vertex_buffers}
                                  : std::span<const AmdGpu::Buffer>{};
    }

    template <typename T>
    T ReadUdSharp(u32 sharp_idx) const noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        ASSERT(sharp_idx * sizeof(u32) + sizeof(T) <=
               flattened_ud_buf.size() * sizeof(u32));
        T value{};
        std::memcpy(&value, flattened_ud_buf.data() + sharp_idx, sizeof(T));
        return value;
    }

    template <typename T>
    T ReadUdReg(u32 ptr_index, u32 dword_offset) const noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        T data{};
        const u32* base = user_data.data();
        if (ptr_index != IR::NumScalarRegs) {
            constexpr size_t PointerDwords = sizeof(VAddr) / sizeof(u32);
            ASSERT(ptr_index + PointerDwords <= user_data.size());
            std::memcpy(&base, &user_data[ptr_index], sizeof(base));
            base = reinterpret_cast<const u32*>(VAddr(base) & 0xFFFFFFFFFFFFULL);
        }
        std::memcpy(&data, base + dword_offset, sizeof(T));
        return data;
    }

    void PushUd(const Info& analysis, Backend::Bindings& bindings, PushData& push) const;
    void ReadTessConstantBuffer(const Info& analysis,
                                TessellationDataConstantBuffer& tess_constants) const;
};

} // namespace Shader
