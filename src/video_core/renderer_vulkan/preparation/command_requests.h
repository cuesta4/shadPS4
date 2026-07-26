// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <span>
#include <vector>

#include "shader_recompiler/info.h"
#include "shader_recompiler/invocation.h"

namespace Vulkan {

struct BufferResourceRequest {
    AmdGpu::Buffer sharp{};
    Shader::BufferType type{};
    bool is_special : 1 {};
    bool is_storage : 1 {};
    bool is_written : 1 {};
    bool is_formatted : 1 {};
};

struct ImageResourceRequest {
    AmdGpu::Image sharp{};
    Shader::ImageResource descriptor{};
    u32 binding_count{};
};

struct SamplerResourceRequest {
    AmdGpu::Sampler sharp{};
};

struct StageResourceRequestSet {
    const Shader::Info* analysis{};
    const Shader::ShaderInvocationData* invocation{};
    u32 buffer_offset{};
    u32 buffer_count{};
    u32 image_offset{};
    u32 image_count{};
    u32 sampler_offset{};
    u32 sampler_count{};
};

struct ShaderResourceRequestSet {
    static constexpr size_t MaxStages = Shader::MaxStageTypes;

    std::array<StageResourceRequestSet, MaxStages> stages{};
    std::array<u8, MaxStages> active_stage_indices{};
    std::vector<BufferResourceRequest> buffers;
    std::vector<ImageResourceRequest> images;
    std::vector<SamplerResourceRequest> samplers;
    u32 active_stage_count{};
    size_t image_descriptor_count{};
    size_t descriptor_write_count{};

    void Reset() noexcept {
        for (u32 index = 0; index < active_stage_count; ++index) {
            stages[active_stage_indices[index]] = {};
        }
        buffers.clear();
        images.clear();
        samplers.clear();
        active_stage_count = 0;
        image_descriptor_count = 0;
        descriptor_write_count = 0;
    }

    [[nodiscard]] std::span<const u8> ActiveStageIndices() const noexcept {
        return {active_stage_indices.data(), active_stage_count};
    }

    [[nodiscard]] std::span<const BufferResourceRequest> Buffers(
        const StageResourceRequestSet& stage) const noexcept {
        return std::span{buffers}.subspan(stage.buffer_offset, stage.buffer_count);
    }

    [[nodiscard]] std::span<const ImageResourceRequest> Images(
        const StageResourceRequestSet& stage) const noexcept {
        return std::span{images}.subspan(stage.image_offset, stage.image_count);
    }

    [[nodiscard]] std::span<const SamplerResourceRequest> Samplers(
        const StageResourceRequestSet& stage) const noexcept {
        return std::span{samplers}.subspan(stage.sampler_offset, stage.sampler_count);
    }
};

inline void BuildShaderResourceRequests(
    ShaderResourceRequestSet& requests, std::span<const Shader::Info* const> analyses,
    std::span<const Shader::ShaderInvocationData> invocations, bool is_compute) {
    requests.Reset();
    ASSERT(analyses.size() <= requests.stages.size());
    ASSERT(is_compute ? !invocations.empty() : invocations.size() >= analyses.size());

    size_t buffer_count{};
    size_t image_count{};
    size_t sampler_count{};
    for (const auto* analysis : analyses) {
        if (analysis != nullptr) {
            buffer_count += analysis->buffers.size();
            image_count += analysis->images.size();
            sampler_count += analysis->samplers.size();
        }
    }
    requests.buffers.reserve(buffer_count);
    requests.images.reserve(image_count);
    requests.samplers.reserve(sampler_count);

    for (u32 stage_index = 0; stage_index < analyses.size(); ++stage_index) {
        const auto* analysis = analyses[stage_index];
        if (analysis == nullptr) {
            continue;
        }
        const auto& invocation = is_compute ? invocations.front() : invocations[stage_index];
        auto& stage = requests.stages[stage_index];
        stage.analysis = analysis;
        stage.invocation = &invocation;
        stage.buffer_offset = static_cast<u32>(requests.buffers.size());
        stage.image_offset = static_cast<u32>(requests.images.size());
        stage.sampler_offset = static_cast<u32>(requests.samplers.size());
        requests.active_stage_indices[requests.active_stage_count++] =
            static_cast<u8>(stage_index);

        for (const auto& descriptor : analysis->buffers) {
            const auto sharp = descriptor.GetSharp(invocation);
            const bool is_storage = descriptor.IsStorage(sharp);
            requests.buffers.push_back({
                .sharp = sharp,
                .type = descriptor.buffer_type,
                .is_special = descriptor.IsSpecial(),
                .is_storage = is_storage,
                .is_written = descriptor.is_written,
                .is_formatted = descriptor.is_formatted,
            });
        }
        for (const auto& descriptor : analysis->images) {
            const auto sharp = descriptor.GetSharp(invocation);
            const u32 binding_count = descriptor.NumBindings(sharp);
            requests.images.push_back({
                .sharp = sharp,
                .descriptor = descriptor,
                .binding_count = binding_count,
            });
            requests.image_descriptor_count += binding_count;
        }
        for (const auto& descriptor : analysis->samplers) {
            auto sharp = descriptor.GetSharp(invocation);
            if (descriptor.disable_aniso) {
                ASSERT(descriptor.associated_image < analysis->images.size());
                const auto& image = analysis->images[descriptor.associated_image];
                const auto image_sharp = image.GetSharp(invocation);
                if (image_sharp.base_level == 0 && image_sharp.last_level == 0) {
                    sharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
                }
            }
            requests.samplers.push_back({.sharp = sharp});
        }

        stage.buffer_count =
            static_cast<u32>(requests.buffers.size()) - stage.buffer_offset;
        stage.image_count = static_cast<u32>(requests.images.size()) - stage.image_offset;
        stage.sampler_count =
            static_cast<u32>(requests.samplers.size()) - stage.sampler_offset;
        requests.image_descriptor_count += stage.sampler_count;
        requests.descriptor_write_count +=
            stage.buffer_count + stage.image_count + stage.sampler_count;
    }
}

} // namespace Vulkan
