// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bitset>

#include "common/types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/profile.h"

namespace Shader {

struct VsAttribSpecialization {
    u32 divisor{};
    AmdGpu::NumberClass num_class{};
    AmdGpu::CompMapping dst_select{};

    bool operator==(const VsAttribSpecialization&) const = default;
};

struct BufferSpecialization {
    u32 stride : 14;
    u32 is_storage : 1;
    u32 is_formatted : 1;
    u32 swizzle_enable : 1;
    u32 data_format : 6;
    u32 num_format : 4;
    u32 index_stride : 2;
    u32 element_size : 2;
    AmdGpu::CompMapping dst_select{};
    AmdGpu::NumberConversion num_conversion{};

    bool operator==(const BufferSpecialization& other) const {
        return stride == other.stride && is_storage == other.is_storage &&
               is_formatted == other.is_formatted && swizzle_enable == other.swizzle_enable &&
               (!is_formatted ||
                (data_format == other.data_format && num_format == other.num_format &&
                 dst_select == other.dst_select && num_conversion == other.num_conversion)) &&
               (!swizzle_enable ||
                (index_stride == other.index_stride && element_size == other.element_size));
    }
};

struct ImageSpecialization {
    AmdGpu::ImageType type = AmdGpu::ImageType::Color2D;
    bool is_integer = false;
    bool is_storage = false;
    bool is_cube = false;
    bool is_srgb = false;
    AmdGpu::CompMapping dst_select{};
    AmdGpu::NumberConversion num_conversion{};
    // FIXME any pipeline cache changes needed?
    u32 num_bindings = 0;

    bool operator==(const ImageSpecialization&) const = default;
};

struct FMaskSpecialization {
    u32 width;
    u32 height;

    bool operator==(const FMaskSpecialization&) const = default;
};

struct SamplerSpecialization {
    u8 force_unnormalized : 1;
    u8 force_degamma : 1;

    bool operator==(const SamplerSpecialization&) const = default;
};

/**
 * Alongside runtime information, this structure also checks bound resources
 * for compatibility. Can be used as a key for storing shader permutations.
 * Is separate from runtime information, because resource layout can only be deduced
 * after the first compilation of a module.
 */
struct StageSpecialization {
    static constexpr size_t MaxStageResources = 128;

    const Info* info{};
    RuntimeInfo runtime_info{};
    std::bitset<MaxStageResources> bitset{};
    std::optional<Gcn::FetchShaderData> fetch_shader_data{};
    boost::container::small_vector<VsAttribSpecialization, 32> vs_attribs;
    boost::container::small_vector<BufferSpecialization, 16> buffers;
    boost::container::small_vector<ImageSpecialization, 16> images;
    boost::container::small_vector<FMaskSpecialization, 8> fmasks;
    boost::container::small_vector<SamplerSpecialization, 16> samplers;
    Backend::Bindings start{};

    StageSpecialization() = default;
    StageSpecialization(
        const Info& info_, RuntimeInfo runtime_info_, const Profile& profile_,
        Backend::Bindings start_,
        const std::optional<Gcn::FetchShaderData>* pre_parsed_fetch_shader_data = nullptr)
        : info{&info_}, runtime_info{runtime_info_}, start{start_} {
        fetch_shader_data = pre_parsed_fetch_shader_data != nullptr ? *pre_parsed_fetch_shader_data
                                                                    : Gcn::ParseFetchShader(info_);
        if (info_.stage == Stage::Vertex && fetch_shader_data) {
            // Specialize shader on VS input number types to follow spec.
            ForEachSharp(vs_attribs, fetch_shader_data->attributes,
                         [&profile_, this](auto& spec, const auto& desc, AmdGpu::Buffer sharp) {
                             using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
                             if (const auto step_rate = desc.GetStepRate();
                                 step_rate != InstanceIdType::None) {
                                 spec.divisor = step_rate == InstanceIdType::OverStepRate0
                                                    ? runtime_info.vs_info.step_rate_0
                                                    : (step_rate == InstanceIdType::OverStepRate1
                                                           ? runtime_info.vs_info.step_rate_1
                                                           : 1);
                             }
                             spec.num_class = profile_.support_legacy_vertex_attributes
                                                  ? AmdGpu::NumberClass{}
                                                  : AmdGpu::GetNumberClass(sharp.GetNumberFmt());
                             spec.dst_select = sharp.DstSelect();
                         });
        }
        u32 binding{};
        ForEachSharp(binding, buffers, info->buffers,
                     [](auto& spec, const auto& desc, AmdGpu::Buffer sharp) {
                         spec.stride = sharp.GetStride();
                         spec.is_storage = desc.IsStorage(sharp);
                         spec.is_formatted = desc.is_formatted;
                         spec.swizzle_enable = sharp.swizzle_enable;
                         if (spec.is_formatted) {
                             spec.data_format = static_cast<u32>(sharp.GetDataFmt());
                             spec.num_format = static_cast<u32>(sharp.GetNumberFmt());
                             spec.dst_select = sharp.DstSelect();
                             spec.num_conversion = sharp.GetNumberConversion();
                         }
                         if (spec.swizzle_enable) {
                             spec.index_stride = sharp.index_stride;
                             spec.element_size = sharp.element_size;
                         }
                     });
        ForEachSharp(binding, images, info->images,
                     [&](auto& spec, const auto& desc, AmdGpu::Image sharp) {
                         spec.type = sharp.GetViewType(desc.is_array);
                         spec.is_integer = AmdGpu::IsInteger(sharp.GetNumberFmt());
                         spec.is_storage = desc.is_written;
                         spec.is_cube = sharp.IsCube();
                         if (spec.is_storage) {
                             spec.dst_select = sharp.DstSelect();
                         } else {
                             spec.is_srgb = sharp.GetNumberFmt() == AmdGpu::NumberFormat::Srgb;
                         }
                         spec.num_conversion = sharp.GetNumberConversion();
                         spec.num_bindings = desc.NumBindings(*info);
                     });
        ForEachSharp(binding, fmasks, info->fmasks,
                     [](auto& spec, const auto& desc, AmdGpu::Image sharp) {
                         spec.width = sharp.width;
                         spec.height = sharp.height;
                     });
        ForEachSharp(samplers, info->samplers,
                     [](auto& spec, const auto& desc, AmdGpu::Sampler sharp) {
                         spec.force_unnormalized = sharp.force_unnormalized;
                         spec.force_degamma = sharp.force_degamma;
                     });

        // Initialize runtime_info fields that rely on analysis in tessellation passes
        if (info->l_stage == LogicalStage::TessellationControl ||
            info->l_stage == LogicalStage::TessellationEval) {
            TessellationDataConstantBuffer tess_constants{};
            info->ReadTessConstantBuffer(tess_constants);
            runtime_info.InitFromTessConstants(tess_constants);
        }
    }

    void ForEachSharp(auto& spec_list, auto& desc_list, auto&& func) {
        for (const auto& desc : desc_list) {
            auto& spec = spec_list.emplace_back();
            const auto sharp = desc.GetSharp(*info);
            if (!sharp) {
                continue;
            }
            func(spec, desc, sharp);
        }
    }

    void ForEachSharp(u32& binding, auto& spec_list, auto& desc_list, auto&& func) {
        for (const auto& desc : desc_list) {
            auto& spec = spec_list.emplace_back();
            const auto sharp = desc.GetSharp(*info);
            if (!sharp) {
                binding++;
                continue;
            }
            bitset.set(binding++);
            func(spec, desc, sharp);
        }
    }

    [[nodiscard]] bool Valid() const {
        return info != nullptr;
    }

    /**
     * Checks whether the currently bound resources are structurally compatible with this cached
     * specialization without constructing a temporary StageSpecialization object.
     *
     * Dynamic addresses are intentionally ignored in the same way as operator==(): only fields
     * that can change the generated shader permutation are compared. This is safe to use only after
     * Info::RefreshFlatBuf() has refreshed indirect SRT-backed descriptors.
     */
    [[nodiscard]] bool Matches(
        const Info& info_, RuntimeInfo runtime_info_, const Profile& profile_,
        Backend::Bindings start_,
        const std::optional<Gcn::FetchShaderData>* pre_parsed_fetch_shader_data = nullptr) const {
        if (!Valid()) {
            return false;
        }

        const auto current_fetch_shader_data = pre_parsed_fetch_shader_data != nullptr
                                                   ? *pre_parsed_fetch_shader_data
                                                   : Gcn::ParseFetchShader(info_);

        if (runtime_info != runtime_info_) {
            return false;
        }
        if (fetch_shader_data != current_fetch_shader_data) {
            return false;
        }

        if (info_.stage == Stage::Vertex && current_fetch_shader_data) {
            if (vs_attribs.size() != current_fetch_shader_data->attributes.size()) {
                return false;
            }
            for (u32 i = 0; i < current_fetch_shader_data->attributes.size(); ++i) {
                const auto& desc = current_fetch_shader_data->attributes[i];
                const auto sharp = desc.GetSharp(info_);
                VsAttribSpecialization current{};
                if (sharp) {
                    using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
                    if (const auto step_rate = desc.GetStepRate();
                        step_rate != InstanceIdType::None) {
                        current.divisor = step_rate == InstanceIdType::OverStepRate0
                                              ? runtime_info_.vs_info.step_rate_0
                                              : (step_rate == InstanceIdType::OverStepRate1
                                                     ? runtime_info_.vs_info.step_rate_1
                                                     : 1);
                    }
                    current.num_class = profile_.support_legacy_vertex_attributes
                                            ? AmdGpu::NumberClass{}
                                            : AmdGpu::GetNumberClass(sharp.GetNumberFmt());
                    current.dst_select = sharp.DstSelect();
                }
                if (current != vs_attribs[i]) {
                    return false;
                }
            }
        } else if (!vs_attribs.empty()) {
            return false;
        }

        if (fmasks.size() != info_.fmasks.size()) {
            return false;
        }
        for (u32 i = 0; i < info_.fmasks.size(); ++i) {
            const auto& desc = info_.fmasks[i];
            const auto sharp = desc.GetSharp(info_);
            FMaskSpecialization current{};
            if (sharp) {
                current.width = sharp.width;
                current.height = sharp.height;
            }
            if (current != fmasks[i]) {
                return false;
            }
        }

        if (buffers.size() != info_.buffers.size() || images.size() != info_.images.size() ||
            samplers.size() != info_.samplers.size()) {
            return false;
        }

        std::bitset<MaxStageResources> current_bitset{};
        u32 binding{};
        for (u32 i = 0; i < info_.buffers.size(); ++i) {
            const auto& desc = info_.buffers[i];
            const auto sharp = desc.GetSharp(info_);
            BufferSpecialization current{};
            if (sharp) {
                current_bitset.set(binding);
                current.stride = sharp.GetStride();
                current.is_storage = desc.IsStorage(sharp);
                current.is_formatted = desc.is_formatted;
                current.swizzle_enable = sharp.swizzle_enable;
                if (current.is_formatted) {
                    current.data_format = static_cast<u32>(sharp.GetDataFmt());
                    current.num_format = static_cast<u32>(sharp.GetNumberFmt());
                    current.dst_select = sharp.DstSelect();
                    current.num_conversion = sharp.GetNumberConversion();
                }
                if (current.swizzle_enable) {
                    current.index_stride = sharp.index_stride;
                    current.element_size = sharp.element_size;
                }
            }
            // Mirror operator==(): the newly observed resource controls whether its
            // specialization fields participate in the comparison.
            if (current_bitset[binding] && current != buffers[i]) {
                return false;
            }
            ++binding;
        }

        for (u32 i = 0; i < info_.images.size(); ++i) {
            const auto& desc = info_.images[i];
            const auto sharp = desc.GetSharp(info_);
            ImageSpecialization current{};
            if (sharp) {
                current_bitset.set(binding);
                current.type = sharp.GetViewType(desc.is_array);
                current.is_integer = AmdGpu::IsInteger(sharp.GetNumberFmt());
                current.is_storage = desc.is_written;
                current.is_cube = sharp.IsCube();
                if (current.is_storage) {
                    current.dst_select = sharp.DstSelect();
                } else {
                    current.is_srgb = sharp.GetNumberFmt() == AmdGpu::NumberFormat::Srgb;
                }
                current.num_conversion = sharp.GetNumberConversion();
                current.num_bindings = desc.NumBindings(info_);
            }
            if (current_bitset[binding] && current != images[i]) {
                return false;
            }
            ++binding;
        }

        // Preserve the existing operator==() semantics for stages without buffer/image bindings.
        if (current_bitset.none() && bitset.none()) {
            return true;
        }

        if (start != start_) {
            return false;
        }

        for (u32 i = 0; i < info_.samplers.size(); ++i) {
            const auto& desc = info_.samplers[i];
            const auto sharp = desc.GetSharp(info_);
            SamplerSpecialization current{};
            if (sharp) {
                current.force_unnormalized = sharp.force_unnormalized;
                current.force_degamma = sharp.force_degamma;
            }
            if (current != samplers[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator==(const StageSpecialization& other) const {
        if (!Valid()) {
            return false;
        }

        if (vs_attribs != other.vs_attribs) {
            return false;
        }

        if (runtime_info != other.runtime_info) {
            return false;
        }

        if (fetch_shader_data != other.fetch_shader_data) {
            return false;
        }

        if (fmasks != other.fmasks) {
            return false;
        }

        // For VS which only generates geometry and doesn't have any inputs, its start
        // bindings still may change as they depend on previously processed FS. The check below
        // handles this case and prevents generation of redundant permutations. This is also safe
        // for other types of shaders with no bindings.
        if (bitset.none() && other.bitset.none()) {
            return true;
        }

        if (start != other.start) {
            return false;
        }

        u32 binding{};
        for (u32 i = 0; i < buffers.size(); i++) {
            if (other.bitset[binding++] && buffers[i] != other.buffers[i]) {
                return false;
            }
        }
        for (u32 i = 0; i < images.size(); i++) {
            if (other.bitset[binding++] && images[i] != other.images[i]) {
                return false;
            }
        }

        for (u32 i = 0; i < samplers.size(); i++) {
            if (samplers[i] != other.samplers[i]) {
                return false;
            }
        }
        return true;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
