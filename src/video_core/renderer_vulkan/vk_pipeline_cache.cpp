// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <boost/container/static_vector.hpp>

#include "common/assert.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/performance_telemetry.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

void ResolvedStageResources::Bind(Shader::Info& info) const noexcept {
    info.resolved_buffers = std::span<const AmdGpu::Buffer>{buffers.data(), buffers.size()};
    info.resolved_images = std::span<const AmdGpu::Image>{images.data(), images.size()};
    info.resolved_samplers = std::span<const AmdGpu::Sampler>{samplers.data(), samplers.size()};
    info.resolved_fmasks = std::span<const AmdGpu::Image>{fmasks.data(), fmasks.size()};
    info.resolved_vertex_buffers =
        std::span<const AmdGpu::Buffer>{vertex_buffers.data(), vertex_buffers.size()};
}

namespace {

template <typename DescriptorList, typename ResourceList>
void ResolveResourceList(const DescriptorList& descriptors, ResourceList& resources,
                         const Shader::Info& info) {
    resources.resize(descriptors.size(), boost::container::default_init);
    for (size_t index = 0; index < descriptors.size(); ++index) {
        resources[index] = descriptors[index].GetSharp(info);
    }
}

void ResolveStageResources(Shader::Info& info,
                           const std::optional<Shader::Gcn::FetchShaderData>* fetch_shader,
                           ResolvedStageResources& resolved, bool telemetry_enabled) {
    Common::PerformanceTelemetry::SampledDuration<
        Common::PerformanceTelemetry::TimerSite::StageResolve>
        resolve_duration{telemetry_enabled, static_cast<u32>(info.l_stage)};
    ResolveResourceList(info.buffers, resolved.buffers, info);
    ResolveResourceList(info.images, resolved.images, info);
    ResolveResourceList(info.samplers, resolved.samplers, info);
    ResolveResourceList(info.fmasks, resolved.fmasks, info);
    if (fetch_shader && fetch_shader->has_value()) {
        ResolveResourceList((*fetch_shader)->attributes, resolved.vertex_buffers, info);
    } else {
        resolved.vertex_buffers.clear();
    }
    resolved.Bind(info);
}

bool BuildSpecializationPlan(Program& program) {
    if (program.specialization_plan_ready) {
        return program.specialization_plan_cacheable;
    }
    program.specialization_plan_ready = true;
    const auto& info = program.info;
    using Reason = Common::PerformanceTelemetry::StageUncacheableReason;
    u32 reasons{};
    if (info.srt_info.walker_func) {
        reasons |= static_cast<u32>(Reason::SrtWalker);
    }
    if (info.l_stage == LogicalStage::TessellationControl) {
        reasons |= static_cast<u32>(Reason::TessellationControl);
    }
    if (info.l_stage == LogicalStage::TessellationEval) {
        reasons |= static_cast<u32>(Reason::TessellationEvaluation);
    }
    const auto valid_range = [](u32 first, u32 dwords) {
        return first + dwords <= Shader::ShaderParams::NumShaderUserData;
    };
    for (const auto& resource : info.buffers) {
        if (!resource.inline_cbuf &&
            !valid_range(resource.sharp_idx, sizeof(AmdGpu::Buffer) / sizeof(u32))) {
            reasons |= static_cast<u32>(Reason::DescriptorOutsideUserData);
        }
    }
    for (const auto& resource : info.images) {
        if (!valid_range(resource.sharp_idx, sizeof(AmdGpu::Image) / sizeof(u32))) {
            reasons |= static_cast<u32>(Reason::DescriptorOutsideUserData);
        }
    }
    for (const auto& resource : info.samplers) {
        if (!resource.is_inline_sampler &&
            !valid_range(resource.sharp_idx, sizeof(AmdGpu::Sampler) / sizeof(u32))) {
            reasons |= static_cast<u32>(Reason::DescriptorOutsideUserData);
        }
    }
    for (const auto& resource : info.fmasks) {
        if (!valid_range(resource.sharp_idx, sizeof(AmdGpu::Image) / sizeof(u32))) {
            reasons |= static_cast<u32>(Reason::DescriptorOutsideUserData);
        }
    }
    if (info.has_fetch_shader && !valid_range(info.fetch_shader_sgpr_base, 2)) {
        reasons |= static_cast<u32>(Reason::FetchPointerOutsideUserData);
    }
    program.specialization_plan_reasons = static_cast<u8>(reasons);
    program.specialization_plan_cacheable = reasons == 0;
    return program.specialization_plan_cacheable;
}

void RefreshDynamicProgramData(
    Shader::Info& info, VAddr program_base,
    std::span<const u32, Shader::ShaderParams::NumShaderUserData> user_data,
    bool telemetry_enabled) {
    Common::PerformanceTelemetry::SampledDuration<
        Common::PerformanceTelemetry::TimerSite::StageRefresh>
        refresh_duration{telemetry_enabled, static_cast<u32>(info.l_stage)};
    info.pgm_base = program_base;
    info.user_data = user_data;
    info.RefreshFlatBuf(telemetry_enabled);
}

void RefreshDynamicProgramData(Shader::Info& info, const Shader::ShaderParams& params,
                               bool telemetry_enabled) {
    RefreshDynamicProgramData(info, params.Base(), params.user_data, telemetry_enabled);
}

struct CachedFetchShader {
    const std::optional<Shader::Gcn::FetchShaderData>* parsed{};
    u64 revision{};

    [[nodiscard]] bool IsUsable(const Shader::Info& info) const noexcept {
        return parsed && (!info.has_fetch_shader || parsed->has_value());
    }
};

constinit const std::optional<Shader::Gcn::FetchShaderData> EmptyFetchShader{};

#if defined(_MSC_VER)
#define SHAD_FETCH_SHADER_FORCE_INLINE __forceinline
#else
#define SHAD_FETCH_SHADER_FORCE_INLINE __attribute__((always_inline)) inline
#endif

#if defined(__AVX2__)
template <size_t WordCount>
[[nodiscard]] SHAD_FETCH_SHADER_FORCE_INLINE bool EqualFixedFetchShaderCode(
    const u32* cached, const u32* current) noexcept {
    static_assert(WordCount >= 4 && WordCount <= 28);
    constexpr size_t YmmCount = WordCount / 8;
    constexpr size_t VectorTail = WordCount % 8;
    constexpr size_t ScalarOffset = YmmCount * 8 + (VectorTail >= 4 ? 4 : 0);
    constexpr size_t ScalarWords = VectorTail - (VectorTail >= 4 ? 4 : 0);

    __m256i difference = _mm256_setzero_si256();
    if constexpr (YmmCount != 0) {
        difference = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cached)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current)));
    }
    if constexpr (YmmCount >= 2) {
        const __m256i block = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cached + 8)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current + 8)));
        difference = _mm256_or_si256(difference, block);
    }
    if constexpr (YmmCount >= 3) {
        const __m256i block = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cached + 16)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current + 16)));
        difference = _mm256_or_si256(difference, block);
    }
    if constexpr (VectorTail >= 4) {
        const __m128i block = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(cached + YmmCount * 8)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(current + YmmCount * 8)));
        if constexpr (YmmCount == 0) {
            difference = _mm256_zextsi128_si256(block);
        } else {
            difference = _mm256_or_si256(difference, _mm256_zextsi128_si256(block));
        }
    }

    u64 tail_difference{};
    if constexpr (ScalarWords >= 2) {
        u64 cached_tail;
        u64 current_tail;
        std::memcpy(&cached_tail, cached + ScalarOffset, sizeof(cached_tail));
        std::memcpy(&current_tail, current + ScalarOffset, sizeof(current_tail));
        tail_difference = cached_tail ^ current_tail;
    }
    if constexpr ((ScalarWords & 1) != 0) {
        tail_difference |= cached[ScalarOffset + (ScalarWords & 2)] ^
                           current[ScalarOffset + (ScalarWords & 2)];
    }
    if constexpr (ScalarWords == 0) {
        return _mm256_testz_si256(difference, difference) != 0;
    } else {
        return tail_difference == 0 && _mm256_testz_si256(difference, difference) != 0;
    }
}
#endif

[[nodiscard]] SHAD_FETCH_SHADER_FORCE_INLINE bool EqualFetchShaderCode(
    const u32* cached, const u32* current, size_t word_count) noexcept {
#if defined(__AVX2__)
    switch (word_count) {
    case 6:
        return EqualFixedFetchShaderCode<6>(cached, current);
    case 9:
        return EqualFixedFetchShaderCode<9>(cached, current);
    case 10:
        return EqualFixedFetchShaderCode<10>(cached, current);
    case 12:
        return EqualFixedFetchShaderCode<12>(cached, current);
    case 15:
        return EqualFixedFetchShaderCode<15>(cached, current);
    case 16:
        return EqualFixedFetchShaderCode<16>(cached, current);
    case 17:
        return EqualFixedFetchShaderCode<17>(cached, current);
    case 18:
        return EqualFixedFetchShaderCode<18>(cached, current);
    case 19:
        return EqualFixedFetchShaderCode<19>(cached, current);
    case 20:
        return EqualFixedFetchShaderCode<20>(cached, current);
    case 21:
        return EqualFixedFetchShaderCode<21>(cached, current);
    case 24:
        return EqualFixedFetchShaderCode<24>(cached, current);
    case 25:
        return EqualFixedFetchShaderCode<25>(cached, current);
    case 28:
        return EqualFixedFetchShaderCode<28>(cached, current);
    default:
        break;
    }
#endif
    return std::memcmp(cached, current, word_count * sizeof(u32)) == 0;
}

#undef SHAD_FETCH_SHADER_FORCE_INLINE

SHAD_NO_INLINE void ReportInvalidFetchShaderSize() {
    ASSERT(false);
}

[[nodiscard]] CachedFetchShader GetCachedFetchShader(Program& program);

[[nodiscard]] SHAD_NO_INLINE CachedFetchShader CacheFetchShaderMiss(Program& program,
                                                                    const u32* code);

[[nodiscard]] CachedFetchShader GetCachedFetchShader(Program& program) {
    auto& info = program.info;
    if (!info.has_fetch_shader) {
        return {.parsed = &EmptyFetchShader, .revision = 0};
    }

    const u32* code = Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
    if (!code) {
        return {.parsed = &EmptyFetchShader, .revision = 0};
    }

    for (auto& entry : program.fetch_shader_cache) {
        if (entry.address != code || !entry.parsed) {
            continue;
        }
        if (entry.parsed->size % sizeof(u32) != 0) [[unlikely]] {
            ReportInvalidFetchShaderSize();
        }
        const size_t word_count = entry.parsed->size / sizeof(u32);
        if (entry.code.size() == word_count &&
            EqualFetchShaderCode(entry.code.data(), code, word_count)) {
            if (Common::PerformanceTelemetry::Enabled()) {
                Common::PerformanceTelemetry::AddEnabled(
                    Common::PerformanceTelemetry::Counter::FetchShaderCacheHits, 1);
                Common::PerformanceTelemetry::ObserveMaxEnabled(
                    Common::PerformanceTelemetry::Counter::FetchShaderWords, word_count);
            }
            return {.parsed = &entry.parsed, .revision = entry.revision};
        }
    }

    return CacheFetchShaderMiss(program, code);
}

[[nodiscard]] SHAD_NO_INLINE CachedFetchShader CacheFetchShaderMiss(Program& program,
                                                                    const u32* code) {
    Common::PerformanceTelemetry::Add(
        Common::PerformanceTelemetry::Counter::FetchShaderCacheMisses);
    auto& info = program.info;
    auto parsed = Shader::Gcn::ParseFetchShader(info);
    if (!parsed) {
        return {.parsed = &EmptyFetchShader, .revision = 0};
    }
    ASSERT(parsed->size % sizeof(u32) == 0);
    const size_t word_count = parsed->size / sizeof(u32);

    u8 slot{};
    if (program.fetch_shader_cache.size() < Program::MaxFetchShaderCacheEntries) {
        slot = static_cast<u8>(program.fetch_shader_cache.size());
        program.fetch_shader_cache.emplace_back();
    } else {
        slot = program.next_fetch_shader_slot;
        program.next_fetch_shader_slot = static_cast<u8>((program.next_fetch_shader_slot + 1) %
                                                         Program::MaxFetchShaderCacheEntries);
    }

    auto& entry = program.fetch_shader_cache[slot];
    entry.address = code;
    entry.code.assign(code, code + word_count);
    entry.parsed = std::move(parsed);
    entry.revision = program.next_fetch_shader_revision++;
    if (program.next_fetch_shader_revision == 0) {
        program.next_fetch_shader_revision = 1;
        program.fetch_shader_cache.clear();
        return GetCachedFetchShader(program);
    }
    return {.parsed = &entry.parsed, .revision = entry.revision};
}

[[nodiscard]] constexpr u64 MixSpecializationFingerprint(u64 value) noexcept {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

class SpecializationFingerprintBuilder {
public:
    void Add(u64 value) noexcept {
        state = std::rotl(state, 17) ^ MixSpecializationFingerprint(value + state);
    }

    [[nodiscard]] u64 Finish() const noexcept {
        const u64 fingerprint = MixSpecializationFingerprint(state);
        return fingerprint != 0 ? fingerprint : 1;
    }

private:
    u64 state{0x6a09e667f3bcc909ULL};
};

[[nodiscard]] u32 PackCompMapping(const AmdGpu::CompMapping& mapping) noexcept {
    return std::bit_cast<u32>(mapping.array);
}

void AddFetchShaderFingerprint(
    SpecializationFingerprintBuilder& builder,
    const std::optional<Shader::Gcn::FetchShaderData>* fetch_shader) noexcept {
    const bool has_fetch_shader = fetch_shader && fetch_shader->has_value();
    builder.Add(has_fetch_shader);
    if (!has_fetch_shader) {
        return;
    }

    const auto& fetch = fetch_shader->value();
    builder.Add(fetch.attributes.size());
    for (const auto& attribute : fetch.attributes) {
        u64 packed{};
        std::memcpy(&packed, &attribute, sizeof(packed));
        builder.Add(packed & 0x0000ffffffffffffULL);
    }
    builder.Add(static_cast<u8>(fetch.vertex_offset_sgpr) |
                (static_cast<u64>(static_cast<u8>(fetch.instance_offset_sgpr)) << 8));
}

void AddVsAttribFingerprint(SpecializationFingerprintBuilder& builder,
                            const Shader::VsAttribSpecialization& spec) noexcept {
    builder.Add(spec.divisor | (static_cast<u64>(spec.num_class) << 32));
    builder.Add(PackCompMapping(spec.dst_select));
}

void AddBufferFingerprint(SpecializationFingerprintBuilder& builder,
                          const Shader::BufferSpecialization& spec) noexcept {
    builder.Add(spec.stride | (static_cast<u64>(spec.is_storage) << 14) |
                (static_cast<u64>(spec.is_formatted) << 15) |
                (static_cast<u64>(spec.swizzle_enable) << 16));
    if (spec.is_formatted) {
        builder.Add(spec.data_format | (static_cast<u64>(spec.num_format) << 8) |
                    (static_cast<u64>(spec.num_conversion) << 16));
        builder.Add(PackCompMapping(spec.dst_select));
    }
    if (spec.swizzle_enable) {
        builder.Add(spec.index_stride | (static_cast<u64>(spec.element_size) << 8));
    }
}

void AddImageFingerprint(SpecializationFingerprintBuilder& builder,
                         const Shader::ImageSpecialization& spec) noexcept {
    builder.Add(static_cast<u64>(spec.type) | (static_cast<u64>(spec.is_integer) << 32) |
                (static_cast<u64>(spec.is_storage) << 33) |
                (static_cast<u64>(spec.is_cube) << 34) |
                (static_cast<u64>(spec.is_srgb) << 35));
    builder.Add(PackCompMapping(spec.dst_select) |
                (static_cast<u64>(spec.num_conversion) << 32));
    builder.Add(spec.num_bindings);
}

struct SpecializationBindingMask {
    u64 low{};
    u64 high{};

    [[nodiscard]] bool Any() const noexcept {
        return (low | high) != 0;
    }

    bool operator==(const SpecializationBindingMask&) const noexcept = default;
};

void SetSpecializationBinding(SpecializationBindingMask& mask, size_t binding) noexcept {
    if (binding < 64) {
        mask.low |= 1ULL << binding;
    } else {
        mask.high |= 1ULL << (binding - 64);
    }
}

[[nodiscard]] SpecializationBindingMask GetCurrentBindingMask(const Shader::Info& info) noexcept {
    SpecializationBindingMask mask{};
    size_t binding{};
    for (const auto sharp : info.resolved_buffers) {
        if (sharp) {
            SetSpecializationBinding(mask, binding);
        }
        ++binding;
    }
    for (const auto sharp : info.resolved_images) {
        if (sharp) {
            SetSpecializationBinding(mask, binding);
        }
        ++binding;
    }
    for (const auto sharp : info.resolved_fmasks) {
        if (sharp) {
            SetSpecializationBinding(mask, binding);
        }
        ++binding;
    }
    return mask;
}

[[nodiscard]] SpecializationBindingMask GetStoredBindingMask(
    const Shader::StageSpecialization& specialization) noexcept {
    SpecializationBindingMask mask{};
    for (size_t binding = 0; binding < Shader::StageSpecialization::MaxStageResources; ++binding) {
        if (specialization.bitset.test(binding)) {
            SetSpecializationBinding(mask, binding);
        }
    }
    return mask;
}

void AddBindingMask(SpecializationFingerprintBuilder& builder,
                    SpecializationBindingMask mask) noexcept {
    builder.Add(mask.low);
    builder.Add(mask.high);
}

void AddBindingStart(SpecializationFingerprintBuilder& builder,
                     const Shader::Backend::Bindings& start) noexcept {
    builder.Add(start.unified | (static_cast<u64>(start.buffer) << 32));
    builder.Add(start.user_data);
}

[[nodiscard]] SHAD_NO_INLINE u64 BuildStoredSpecializationFingerprint(
    const Shader::StageSpecialization& specialization) noexcept {
    SpecializationFingerprintBuilder builder{};
    builder.Add(specialization.vs_attribs.size());
    builder.Add(specialization.buffers.size());
    builder.Add(specialization.images.size());
    builder.Add(specialization.fmasks.size());
    builder.Add(specialization.samplers.size());

    const auto binding_mask = GetStoredBindingMask(specialization);
    AddBindingMask(builder, binding_mask);
    AddFetchShaderFingerprint(builder, &specialization.fetch_shader_data);
    for (const auto& spec : specialization.vs_attribs) {
        AddVsAttribFingerprint(builder, spec);
    }
    for (const auto& spec : specialization.fmasks) {
        builder.Add(std::bit_cast<u64>(spec));
    }
    if (binding_mask.Any()) {
        AddBindingStart(builder, specialization.start);
        for (const auto& spec : specialization.buffers) {
            AddBufferFingerprint(builder, spec);
        }
        for (const auto& spec : specialization.images) {
            AddImageFingerprint(builder, spec);
        }
        for (const auto& spec : specialization.samplers) {
            builder.Add(spec.force_unnormalized |
                        (static_cast<u64>(spec.force_degamma) << 1));
        }
    }
    return builder.Finish();
}

[[nodiscard]] SHAD_NO_INLINE u64 BuildCurrentSpecializationFingerprint(
    const Shader::Info& info, const Shader::RuntimeInfo& runtime_info,
    const Shader::Backend::Bindings& start,
    const std::optional<Shader::Gcn::FetchShaderData>* fetch_shader) noexcept {
    SpecializationFingerprintBuilder builder{};
    const bool has_vertex_attributes =
        info.stage == Stage::Vertex && fetch_shader && fetch_shader->has_value();
    const size_t vertex_attribute_count =
        has_vertex_attributes ? fetch_shader->value().attributes.size() : 0;
    builder.Add(vertex_attribute_count);
    builder.Add(info.buffers.size());
    builder.Add(info.images.size());
    builder.Add(info.fmasks.size());
    builder.Add(info.samplers.size());

    const auto binding_mask = GetCurrentBindingMask(info);
    AddBindingMask(builder, binding_mask);
    AddFetchShaderFingerprint(builder, fetch_shader);
    if (has_vertex_attributes) {
        const auto& attributes = fetch_shader->value().attributes;
        for (size_t index = 0; index < attributes.size(); ++index) {
            Shader::VsAttribSpecialization spec{};
            const auto sharp = info.resolved_vertex_buffers[index];
            if (sharp) {
                spec = Shader::MakeVsAttribSpecialization(attributes[index], sharp, runtime_info);
            }
            AddVsAttribFingerprint(builder, spec);
        }
    }
    for (size_t index = 0; index < info.fmasks.size(); ++index) {
        Shader::FMaskSpecialization spec{};
        const auto sharp = info.resolved_fmasks[index];
        if (sharp) {
            spec = Shader::MakeFMaskSpecialization(sharp);
        }
        builder.Add(std::bit_cast<u64>(spec));
    }
    if (binding_mask.Any()) {
        AddBindingStart(builder, start);
        for (size_t index = 0; index < info.buffers.size(); ++index) {
            Shader::BufferSpecialization spec{};
            const auto sharp = info.resolved_buffers[index];
            if (sharp) {
                spec = Shader::MakeBufferSpecialization(info.buffers[index], sharp);
            }
            AddBufferFingerprint(builder, spec);
        }
        for (size_t index = 0; index < info.images.size(); ++index) {
            Shader::ImageSpecialization spec{};
            const auto sharp = info.resolved_images[index];
            if (sharp) {
                spec = Shader::MakeImageSpecialization(info.images[index], sharp);
            }
            AddImageFingerprint(builder, spec);
        }
        for (size_t index = 0; index < info.samplers.size(); ++index) {
            Shader::SamplerSpecialization spec{};
            const auto sharp = info.resolved_samplers[index];
            if (sharp) {
                spec = Shader::MakeSamplerSpecialization(sharp);
            }
            builder.Add(spec.force_unnormalized |
                        (static_cast<u64>(spec.force_degamma) << 1));
        }
    }
    return builder.Finish();
}

[[nodiscard]] SHAD_NO_INLINE bool MatchesCurrentSpecialization(
    const Shader::StageSpecialization& candidate, const Shader::Info& info,
    const Shader::RuntimeInfo& runtime_info, const Shader::Backend::Bindings& start,
    const std::optional<Shader::Gcn::FetchShaderData>* fetch_shader) noexcept {
    if (!candidate.Valid() || candidate.runtime_info != runtime_info || !fetch_shader) {
        return false;
    }

    const bool has_vertex_attributes =
        info.stage == Stage::Vertex && fetch_shader->has_value();
    const size_t vertex_attribute_count =
        has_vertex_attributes ? fetch_shader->value().attributes.size() : 0;
    if (candidate.vs_attribs.size() != vertex_attribute_count ||
        candidate.buffers.size() != info.buffers.size() ||
        candidate.images.size() != info.images.size() ||
        candidate.fmasks.size() != info.fmasks.size() ||
        candidate.samplers.size() != info.samplers.size() ||
        candidate.fetch_shader_data != *fetch_shader) {
        return false;
    }

    const auto binding_mask = GetCurrentBindingMask(info);
    if (binding_mask != GetStoredBindingMask(candidate)) {
        return false;
    }

    if (has_vertex_attributes) {
        const auto& attributes = fetch_shader->value().attributes;
        if (info.resolved_vertex_buffers.size() != attributes.size()) {
            return false;
        }
        for (size_t index = 0; index < attributes.size(); ++index) {
            Shader::VsAttribSpecialization current{};
            const auto sharp = info.resolved_vertex_buffers[index];
            if (sharp) {
                current =
                    Shader::MakeVsAttribSpecialization(attributes[index], sharp, runtime_info);
            }
            if (candidate.vs_attribs[index] != current) {
                return false;
            }
        }
    }

    for (size_t index = 0; index < info.fmasks.size(); ++index) {
        Shader::FMaskSpecialization current{};
        const auto sharp = info.resolved_fmasks[index];
        if (sharp) {
            current = Shader::MakeFMaskSpecialization(sharp);
        }
        if (candidate.fmasks[index] != current) {
            return false;
        }
    }

    if (!binding_mask.Any()) {
        return true;
    }
    if (candidate.start != start) {
        return false;
    }

    for (size_t index = 0; index < info.buffers.size(); ++index) {
        Shader::BufferSpecialization current{};
        const auto sharp = info.resolved_buffers[index];
        if (sharp) {
            current = Shader::MakeBufferSpecialization(info.buffers[index], sharp);
        }
        if (candidate.buffers[index] != current) {
            return false;
        }
    }
    for (size_t index = 0; index < info.images.size(); ++index) {
        Shader::ImageSpecialization current{};
        const auto sharp = info.resolved_images[index];
        if (sharp) {
            current = Shader::MakeImageSpecialization(info.images[index], sharp);
        }
        if (candidate.images[index] != current) {
            return false;
        }
    }
    for (size_t index = 0; index < info.samplers.size(); ++index) {
        Shader::SamplerSpecialization current{};
        const auto sharp = info.resolved_samplers[index];
        if (sharp) {
            current = Shader::MakeSamplerSpecialization(sharp);
        }
        if (candidate.samplers[index] != current) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] SHAD_NO_INLINE size_t FindCachedPermutation(
    Program& program, const Shader::Info& info, const Shader::RuntimeInfo& runtime_info,
    const Shader::Backend::Bindings& start,
    const std::optional<Shader::Gcn::FetchShaderData>* fetch_shader,
    size_t excluded_permutation, bool telemetry_enabled) noexcept {
    const u64 fingerprint =
        BuildCurrentSpecializationFingerprint(info, runtime_info, start, fetch_shader);
    for (size_t permutation = 0; permutation < program.modules.size(); ++permutation) {
        if (permutation == excluded_permutation) {
            continue;
        }
        auto& module = program.modules[permutation];
        if (module.spec.runtime_info != runtime_info) {
            continue;
        }
        if (module.specialization_fingerprint == 0) {
            module.specialization_fingerprint = BuildStoredSpecializationFingerprint(module.spec);
        }
        if (module.specialization_fingerprint != fingerprint) {
            continue;
        }
        if (MatchesCurrentSpecialization(module.spec, info, runtime_info, start, fetch_shader)) {
            return permutation;
        }
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::StageFingerprintCollisions, 1);
        }
    }
    return Program::InvalidPermutation;
}

struct StageRawDependencyKey {
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    u64 fetch_shader_revision{};
};

[[nodiscard]] bool MatchesUserData(
    const StageRawDependencyKey& expected,
    std::span<const u32, Shader::ShaderParams::NumShaderUserData> user_data) noexcept {
#if defined(__AVX2__)
    const auto* expected_words = expected.user_data.data();
    const auto* current_words = user_data.data();
    __m256i difference = _mm256_xor_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(expected_words)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_words)));
    difference = _mm256_or_si256(
        difference,
        _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(expected_words + 8)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_words + 8))));
    return _mm256_testz_si256(difference, difference) != 0;
#else
    return std::ranges::equal(expected.user_data, user_data);
#endif
}

struct GraphicsDependencyKey {
    u64 fixed_generation{};
    std::array<StageRawDependencyKey, MaxShaderStages> stage_keys{};
    std::array<VAddr, MaxShaderStages> stage_bases{};
    u32 active_mask{};
};

struct StageCurrentEntry {
    Program* program{};
    VAddr program_base{};
    Stage stage{};
};

} // namespace

struct PipelineCache::OptimizationState {
    GraphicsDependencyKey graphics_dependency{};
    const GraphicsPipeline* graphics_pipeline{};
    bool graphics_valid{};
    bool graphics_cacheable{};
    Common::PerformanceTelemetry::Gate telemetry_enabled{};
    std::array<StageCurrentEntry, MaxShaderStages> current_stages{};

    [[nodiscard]] bool MatchesGraphicsDependency(PipelineCache& cache);
    [[nodiscard]] bool CaptureGraphicsDependency(PipelineCache& cache);
};

SHAD_NO_INLINE bool PipelineCache::OptimizationState::MatchesGraphicsDependency(
    PipelineCache& cache) {
    const u32 expected_active_mask = graphics_dependency.active_mask;
    for (u32 logical_index = 0; logical_index < MaxShaderStages; ++logical_index) {
        const bool active = cache.infos[logical_index] != nullptr;
        if (active != ((expected_active_mask >> logical_index) & 1U)) {
            return false;
        }
        if (!active) {
            continue;
        }
        const auto& stage_cache = current_stages[logical_index];
        if (!stage_cache.program || cache.infos[logical_index] != &stage_cache.program->info) {
            return false;
        }
        const auto* shader_program =
            cache.liverpool->regs.ProgramForStage(static_cast<u32>(stage_cache.stage));
        if (!shader_program || !shader_program->Address<u32*>()) {
            return false;
        }
        const VAddr program_base = shader_program->Address<VAddr>();
        auto& program = *stage_cache.program;
        if (!program.specialization_plan_ready || !program.specialization_plan_cacheable ||
            program_base != stage_cache.program_base ||
            program_base != graphics_dependency.stage_bases[logical_index] ||
            !MatchesUserData(graphics_dependency.stage_keys[logical_index],
                             shader_program->user_data)) {
            return false;
        }

        program.info.pgm_base = program_base;
        program.info.user_data = shader_program->user_data;
        const auto cached_fetch_shader = GetCachedFetchShader(program);
        if (!cached_fetch_shader.IsUsable(program.info) ||
            cached_fetch_shader.revision !=
                graphics_dependency.stage_keys[logical_index].fetch_shader_revision) {
            return false;
        }
    }
    return true;
}

SHAD_NO_INLINE bool PipelineCache::OptimizationState::CaptureGraphicsDependency(
    PipelineCache& cache) {
    graphics_dependency = {};
    graphics_dependency.fixed_generation = cache.liverpool->GraphicsPipelineGeneration();
    for (u32 logical_index = 0; logical_index < MaxShaderStages; ++logical_index) {
        if (!cache.infos[logical_index]) {
            continue;
        }
        const auto& stage_cache = current_stages[logical_index];
        if (!stage_cache.program || cache.infos[logical_index] != &stage_cache.program->info) {
            return false;
        }
        const auto* shader_program =
            cache.liverpool->regs.ProgramForStage(static_cast<u32>(stage_cache.stage));
        if (!shader_program || !shader_program->Address<u32*>()) {
            return false;
        }
        const VAddr program_base = shader_program->Address<VAddr>();
        auto& program = *stage_cache.program;
        if (!BuildSpecializationPlan(program) || program_base != stage_cache.program_base) {
            return false;
        }

        program.info.pgm_base = program_base;
        program.info.user_data = shader_program->user_data;
        const auto cached_fetch_shader = GetCachedFetchShader(program);
        if (!cached_fetch_shader.IsUsable(program.info)) {
            return false;
        }

        graphics_dependency.active_mask |= 1U << logical_index;
        graphics_dependency.stage_bases[logical_index] = program_base;
        auto& stage_dependency = graphics_dependency.stage_keys[logical_index];
        std::ranges::copy(shader_program->user_data, stage_dependency.user_data.begin());
        stage_dependency.fetch_shader_revision = cached_fetch_shader.revision;
    }
    return true;
}

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(Stage stage, LogicalStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto& regs = liverpool->regs;
    const auto BuildCommon = [&](const auto& program) {
        info.num_user_data = program.settings.num_user_regs;
        info.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.num_allocated_vgprs = program.NumVgprs();
        info.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.fp_round_mode32 = program.settings.fp_round_mode32;
        info.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage);
    switch (stage) {
    case Stage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = infos[u32(Shader::LogicalStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.ls_info.ls_stride = tess_constants.ls_stride;
        break;
    }
    case Stage::Hull: {
        BuildCommon(regs.hs_program);
        info.hs_info.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.hs_info.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.hs_info.tess_type = regs.tess_config.type;
        info.hs_info.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;

        // We need to initialize most hs_info fields after finding the V# with tess constants
        break;
    }
    case Stage::Export: {
        BuildCommon(regs.es_program);
        info.es_info.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Vertex: {
        BuildCommon(regs.vs_program);
        info.vs_info.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.vs_info.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.vs_info.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.vs_info.num_outputs = MapOutputs(info.vs_info.outputs, regs.vs_output_control);
        info.vs_info.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.vs_info.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        info.vs_info.clip_disable = regs.IsClipDisabled();
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Geometry: {
        BuildCommon(regs.gs_program);
        auto& gs_info = info.gs_info;
        gs_info.num_outputs = MapOutputs(gs_info.outputs, regs.vs_output_control);
        gs_info.output_vertices = regs.vgt_gs_max_vert_out;
        gs_info.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            gs_info.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            gs_info.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            gs_info.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        gs_info.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        gs_info.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        gs_info.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        gs_info.vs_copy = params_vc.code;
        gs_info.vs_copy_hash = params_vc.hash;
        DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, 0, "copy.bin");
        break;
    }
    case Stage::Fragment: {
        BuildCommon(regs.ps_program);
        info.fs_info.en_flags = regs.ps_input_ena;
        info.fs_info.addr_flags = regs.ps_input_addr;
        info.fs_info.num_inputs = regs.num_interp;
        info.fs_info.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.fs_info.mrtz_mask = regs.depth_shader_control.z_export_enable |
                                 (stencil_ref_export_enable << 1) |
                                 (regs.depth_shader_control.mask_export_enable << 2) |
                                 (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.fs_info.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.fs_info.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.fs_info.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.fs_info.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.fs_info.color_buffers[i] = graphics_key.color_buffers[i];
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Geometry));
        info.fs_info.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    case Stage::Compute: {
        const auto& cs_pgm = liverpool->GetCsRegs();
        info.num_user_data = cs_pgm.settings.num_user_regs;
        info.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.cs_info.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                       cs_pgm.num_thread_z.full};
        info.cs_info.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                    cs_pgm.IsTgidEnabled(2)};
        info.cs_info.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes},
      optimization{std::make_unique<OptimizationState>()} {
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };
    WarmUp();

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
}

PipelineCache::~PipelineCache() = default;

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline() {
    auto& opt = *optimization;
    opt.telemetry_enabled = Common::PerformanceTelemetry::Enabled();

    if (opt.graphics_valid && opt.graphics_cacheable && opt.graphics_pipeline &&
        liverpool->GraphicsPipelineGeneration() == opt.graphics_dependency.fixed_generation) {
        if (opt.MatchesGraphicsDependency(*this)) {
            if (opt.telemetry_enabled) {
                Common::PerformanceTelemetry::AddEnabled(
                    Common::PerformanceTelemetry::Counter::PipelineHits, 1);
            }
            return opt.graphics_pipeline;
        }
    }

    const auto* pipeline = ResolveGraphicsPipelineSlow();
    opt.graphics_pipeline = pipeline;
    opt.graphics_valid = pipeline != nullptr;
    opt.graphics_cacheable = pipeline && CanReuseGraphicsPipeline();
    if (opt.graphics_cacheable && !opt.CaptureGraphicsDependency(*this)) {
        opt.graphics_cacheable = false;
    }
    return pipeline;
}

const GraphicsPipeline* PipelineCache::ResolveGraphicsPipelineSlow() {
    if (!RefreshGraphicsKey()) {
        if (optimization->telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::PipelineMisses, 1);
        }
        return nullptr;
    }
    const auto it = graphics_pipelines.find(graphics_key);
    if (it != graphics_pipelines.end()) [[likely]] {
        if (optimization->telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::PipelineHits, 1);
        }
        return it.value().get();
    }
    return CreateGraphicsPipeline();
}

SHAD_NO_INLINE const GraphicsPipeline* PipelineCache::CreateGraphicsPipeline() {
    auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (!is_new) [[unlikely]] {
        return it.value().get();
    }
    if (optimization->telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::PipelineMisses, 1);
    }
    const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
    LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

    std::optional<const Shader::Gcn::FetchShaderData> pipeline_fetch_shader{};
    if (fetch_shader && fetch_shader->has_value()) {
        pipeline_fetch_shader.emplace(fetch_shader->value());
    }

    GraphicsPipeline::SerializationSupport sdata{};
    Common::PerformanceTelemetry::ScopedDuration compile_duration{
        optimization->telemetry_enabled,
        Common::PerformanceTelemetry::Counter::PipelineCompileNs};
    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, std::move(pipeline_fetch_shader), modules, sdata, false);

    RegisterPipelineData(graphics_key, pipeline_hash, sdata);
    ++num_new_pipelines;

    if (EmulatorSettings.IsShaderCollect()) {
        for (auto stage = 0; stage < MaxShaderStages; ++stage) {
            if (infos[stage]) {
                auto& m = modules[stage];
                module_related_pipelines[m].emplace_back(graphics_key);
            }
        }
    }
    fetch_shader = nullptr;
    return it.value().get();
}

bool PipelineCache::CanReuseGraphicsPipeline() const {
    for (const auto* info : infos) {
        if (!info) {
            continue;
        }
        const auto program_it = program_cache.find(info->pgm_hash);
        if (program_it == program_cache.end() || !program_it.value()->specialization_plan_ready ||
            !program_it.value()->specialization_plan_cacheable) {
            return false;
        }
    }
    return true;
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    optimization->telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    if (!RefreshComputeKey()) {
        if (optimization->telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::PipelineMisses, 1);
        }
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        if (optimization->telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::PipelineMisses, 1);
        }
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        Common::PerformanceTelemetry::ScopedDuration compile_duration{
            optimization->telemetry_enabled,
            Common::PerformanceTelemetry::Counter::PipelineCompileNs};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    } else {
        if (optimization->telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::PipelineHits, 1);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = nullptr;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in, Shader::LogicalStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        const FetchShader* fetch_shader_{};
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        if (fetch_shader_ && fetch_shader_->has_value()) {
            fetch_shader = fetch_shader_;
        }
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    bind_stage(Stage::Fragment, LogicalStage::Fragment);

    const auto* fs_info = infos[static_cast<u32>(LogicalStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Vertex, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(Stage::Vertex, LogicalStage::Vertex);
        break;
    default:
        UNREACHABLE_MSG("unhandled stage_en: {}", (u32)regs.stage_enable.raw);
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader && fetch_shader->has_value() &&
        !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        ASSERT_MSG(vs_info->resolved_vertex_buffers.size() == (*fetch_shader)->attributes.size(),
                   "Resolved vertex buffer count does not match fetch shader attributes: {} != {}",
                   vs_info->resolved_vertex_buffers.size(), (*fetch_shader)->attributes.size());
        u32 vertex_binding = 0;
        for (const auto& attrib : (*fetch_shader)->attributes) {
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            const auto buffer = vs_info->resolved_vertex_buffers[vertex_binding];
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    std::tie(infos[0], modules[0], fetch_shader, compute_key.value) =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding);
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");

    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.stage, perm_idx, "spv");

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    const auto& runtime_info = BuildRuntimeInfo(stage, l_stage);
    auto& opt = *optimization;
    const u32 stage_index = static_cast<u32>(l_stage);
    auto& current_stage = opt.current_stages[stage_index];
    const auto start = binding;

    auto program_it = program_cache.find(params.hash);
    if (program_it == program_cache.end()) [[unlikely]] {
        if (opt.telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::StageCacheMisses, 1);
        }
        const auto result = CreateProgram(stage, l_stage, params, runtime_info, binding);
        auto& program = *program_cache.find(params.hash).value();
        current_stage = {.program = &program, .program_base = params.Base(), .stage = stage};
        return result;
    }

    auto& program = *program_it.value();
    auto& info = program.info;
    RefreshDynamicProgramData(info, params, opt.telemetry_enabled);
    const auto cached_fetch_shader = GetCachedFetchShader(program);
    ResolveStageResources(info, cached_fetch_shader.parsed, program.resolved_resources,
                          opt.telemetry_enabled);

    const bool plan_cacheable = BuildSpecializationPlan(program);
    const bool fetch_shader_usable = cached_fetch_shader.IsUsable(info);
    const bool cacheable = plan_cacheable && fetch_shader_usable;
    if (cacheable) {
        const size_t current_permutation = program.current_permutation;
        if (current_permutation < program.modules.size()) [[likely]] {
            auto& module = program.modules[current_permutation];
            bool current_matches;
            {
                Common::PerformanceTelemetry::SampledDuration<
                    Common::PerformanceTelemetry::TimerSite::StageCurrentMatch>
                    match_duration{opt.telemetry_enabled, stage_index};
                current_matches = MatchesCurrentSpecialization(
                    module.spec, info, runtime_info, start, cached_fetch_shader.parsed);
            }
            if (current_matches) [[likely]] {
                info.AddBindings(binding);
                current_stage = {
                    .program = &program,
                    .program_base = params.Base(),
                    .stage = stage,
                };
                if (opt.telemetry_enabled) {
                    Common::PerformanceTelemetry::AddEnabled(
                        Common::PerformanceTelemetry::Counter::StageCacheCurrentHits, 1);
                }
                return {&info, module.module, cached_fetch_shader.parsed,
                        HashCombine(params.hash, current_permutation)};
            }
        }

        size_t permutation;
        {
            Common::PerformanceTelemetry::SampledDuration<
                Common::PerformanceTelemetry::TimerSite::StageSearch>
                search_duration{opt.telemetry_enabled, stage_index};
            permutation = FindCachedPermutation(program, info, runtime_info, start,
                                                cached_fetch_shader.parsed,
                                                current_permutation, opt.telemetry_enabled);
        }
        if (permutation != Program::InvalidPermutation) {
            auto& module = program.modules[permutation];
            program.current_permutation = permutation;
            info.AddBindings(binding);
            current_stage = {
                .program = &program,
                .program_base = params.Base(),
                .stage = stage,
            };
            if (opt.telemetry_enabled) {
                Common::PerformanceTelemetry::AddEnabled(
                    Common::PerformanceTelemetry::Counter::StageCacheSearchHits, 1);
            }
            return {&info, module.module, cached_fetch_shader.parsed,
                    HashCombine(params.hash, permutation)};
        }
    } else if (opt.telemetry_enabled) {
        using Reason = Common::PerformanceTelemetry::StageUncacheableReason;
        u32 reasons = program.specialization_plan_reasons;
        if (!fetch_shader_usable) {
            reasons |= static_cast<u32>(Reason::FetchShaderUnavailable);
        }
        Common::PerformanceTelemetry::RecordStageUncacheableEnabled(stage_index, reasons);
    }

    if (opt.telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::StageCacheMisses, 1);
    }
    const auto result = GetProgramSlow(program, stage, l_stage, params, runtime_info, binding,
                                       cached_fetch_shader.parsed);
    current_stage = {.program = &program, .program_base = params.Base(), .stage = stage};
    return result;
}

SHAD_NO_INLINE PipelineCache::Result PipelineCache::GetProgramSlow(
    Program& program, Stage stage, LogicalStage l_stage, const Shader::ShaderParams& params,
    const Shader::RuntimeInfo& runtime_info, Shader::Backend::Bindings& binding,
    const FetchShader* fetch_shader_) {
    auto& info = program.info;
    const bool telemetry_enabled = optimization->telemetry_enabled;
    Common::PerformanceTelemetry::SampledDuration<
        Common::PerformanceTelemetry::TimerSite::StageSlowPath>
        slow_path_duration{telemetry_enabled, static_cast<u32>(l_stage)};
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::StageSpecializationBuilds, 1);
    }
    auto spec = [&] {
        Common::PerformanceTelemetry::SampledDuration<
            Common::PerformanceTelemetry::TimerSite::StageSpecializationBuild>
            specialization_duration{telemetry_enabled, static_cast<u32>(l_stage)};
        return Shader::StageSpecialization(info, runtime_info, profile, binding, fetch_shader_);
    }();

    const size_t previous_permutation = program.current_permutation;
    size_t perm_idx = program.modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program.modules, spec, &Program::Module::spec);
    if (it == program.modules.end()) [[unlikely]] {
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::RecordStageSlowResultEnabled(
                false, false, program.modules.size());
        }
        module = CompilePermutation(program, stage, l_stage, params, runtime_info, binding,
                                    std::move(spec), perm_idx, perm_hash);
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program.modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::RecordStageSlowResultEnabled(
                true, perm_idx == previous_permutation, perm_idx + 1);
        }
    }
    program.current_permutation = perm_idx;
    return {&program.info, module, fetch_shader_, perm_hash};
}

SHAD_NO_INLINE PipelineCache::Result PipelineCache::CreateProgram(
    Stage stage, LogicalStage l_stage, const Shader::ShaderParams& params,
    const Shader::RuntimeInfo& runtime_info, Shader::Backend::Bindings& binding) {
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    ASSERT(new_program);
    it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
    auto& program = *it_pgm.value();
    auto start = binding;
    auto compile_runtime_info = runtime_info;
    const auto module =
        CompileModule(program.info, compile_runtime_info, params.code, 0, binding);
    const bool telemetry_enabled = optimization->telemetry_enabled;
    RefreshDynamicProgramData(program.info, params, telemetry_enabled);
    const auto cached_fetch_shader = GetCachedFetchShader(program);
    ResolveStageResources(program.info, cached_fetch_shader.parsed, program.resolved_resources,
                          telemetry_enabled);
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::StageSpecializationBuilds, 1);
    }
    auto spec = [&] {
        Common::PerformanceTelemetry::SampledDuration<
            Common::PerformanceTelemetry::TimerSite::StageSpecializationBuild>
            specialization_duration{telemetry_enabled, static_cast<u32>(l_stage)};
        return Shader::StageSpecialization(program.info, compile_runtime_info, profile, start,
                                           cached_fetch_shader.parsed);
    }();
    const auto perm_hash = HashCombine(params.hash, 0);

    RegisterShaderMeta(program.info, spec.fetch_shader_data, spec, perm_hash, 0);
    program.AddPermut(module, std::move(spec));
    program.modules[0].specialization_fingerprint =
        BuildStoredSpecializationFingerprint(program.modules[0].spec);
    program.current_permutation = 0;
    if (optimization->telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::StageProgramCreates, 1);
    }
    return {&program.info, module, cached_fetch_shader.parsed, perm_hash};
}

SHAD_NO_INLINE vk::ShaderModule PipelineCache::CompilePermutation(
    Program& program, Stage stage, LogicalStage l_stage, const Shader::ShaderParams& params,
    const Shader::RuntimeInfo& runtime_info, Shader::Backend::Bindings& binding,
    Shader::StageSpecialization&& specialization, size_t permutation_index,
    u64 permutation_hash) {
    auto compile_runtime_info = runtime_info;
    auto new_info = Shader::Info(stage, l_stage, params);
    const auto module =
        CompileModule(new_info, compile_runtime_info, params.code, permutation_index, binding);

    RegisterShaderMeta(program.info, specialization.fetch_shader_data, specialization,
                       permutation_hash, permutation_index);
    program.AddPermut(module, std::move(specialization));
    program.modules.back().specialization_fingerprint =
        BuildStoredSpecializationFingerprint(program.modules.back().spec);
    if (optimization->telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::StagePermutationCompiles, 1);
    }
    return module;
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    optimization->graphics_valid = false;
    optimization->graphics_pipeline = nullptr;
    for (auto& current_stage : optimization->current_stages) {
        current_stage = {};
    }
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::Stage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::Stage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
