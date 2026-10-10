// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <sirit/sirit.h>

#include "shader_recompiler/backend/spirv/emit_spirv_discard_frag.h"

namespace Shader::Backend::SPIRV {

using Sirit::Id;

constexpr u32 SPIRV_VERSION_1_5 = 0x00010500;

struct DiscardShaderEmitter : public Sirit::Module {
    explicit DiscardShaderEmitter(u8 clip_distance_mask_)
        : Sirit::Module{SPIRV_VERSION_1_5}, clip_distance_mask{clip_distance_mask_} {
        void_id = TypeVoid();
        bool_id = TypeBool();
        float_id = TypeFloat(32);
        vec4_id = TypeVector(float_id, 4);

        float_zero = Constant(float_id, 0.0f);
    }

    void EmitDiscardShader() {
        AddCapability(spv::Capability::Shader);
        AddExtension("SPV_EXT_demote_to_helper_invocation");
        AddCapability(spv::Capability::DemoteToHelperInvocation);
        SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
        const Id void_function{TypeFunction(void_id)};
        main = OpFunction(void_id, spv::FunctionControlMask::MaskNone, void_function);

        auto ptr_id = TypePointer(spv::StorageClass::Input, vec4_id);

        std::array<Id, MaxEmulatedClipDistances / 4> locations_id{};
        const u32 num_locations = NumClipDistanceAttributes(clip_distance_mask);
        for (u32 i = 0; i < num_locations; ++i) {
            locations_id[i] = AddGlobalVariable(ptr_id, spv::StorageClass::Input);
            Decorate(locations_id[i], spv::Decoration::Location, i);
            Name(locations_id[i], fmt::format("ccdist{}_in", i));
        }

        AddEntryPoint(spv::ExecutionModel::Fragment, main, "main",
                      std::span(locations_id).first(num_locations));
        AddExecutionMode(main, spv::ExecutionMode::OriginUpperLeft);
        AddLabel(OpLabel());

        std::array<Id, MaxEmulatedClipDistances / 4> values{};
        for (u32 i = 0; i < num_locations; ++i) {
            values[i] = OpLoad(vec4_id, locations_id[i]);
        }
        Id clipped = ConstantFalse(bool_id);
        for (u32 comp = 0; comp < MaxEmulatedClipDistances; ++comp) {
            if ((clip_distance_mask & (1u << comp)) == 0) {
                continue;
            }
            const u32 packed = PackedClipDistanceIndex(clip_distance_mask, comp);
            const Id plane = OpCompositeExtract(float_id, values[packed / 4], packed % 4);
            clipped = OpLogicalOr(bool_id, clipped, OpFOrdLessThan(bool_id, plane, float_zero));
        }
        const Id kill_label{OpLabel()};
        const Id merge_label{OpLabel()};
        OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(clipped, kill_label, merge_label);
        AddLabel(kill_label);
        OpDemoteToHelperInvocationEXT();
        OpBranch(merge_label);
        AddLabel(merge_label);

        OpReturn();
        OpFunctionEnd();
    }

private:
    Id main;
    Id void_id;
    Id bool_id;
    Id float_id;
    Id vec4_id;
    Id float_zero;

    u8 clip_distance_mask;
};

std::vector<u32> EmitDiscardFragmentShader(u8 clip_distance_mask) {
    DiscardShaderEmitter ctx{clip_distance_mask};
    ctx.EmitDiscardShader();
    return ctx.Assemble();
}

} // namespace Shader::Backend::SPIRV
