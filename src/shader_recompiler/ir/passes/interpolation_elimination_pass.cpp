// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/program.h"

namespace Shader::Optimization {
namespace {

struct InterpolationComponent {
    std::array<IR::Inst*, 3> vertices{};
    std::array<IR::Inst*, 2> differences{};
    std::array<IR::Inst*, 2> results{};
};

bool IsPerVertexRead(const IR::Inst* inst, const Info& info) {
    return inst && inst->GetOpcode() == IR::Opcode::GetAttribute &&
           IR::IsParam(inst->Arg(0).Attribute()) &&
           info.fs_interpolation[u32(inst->Arg(0).Attribute()) - u32(IR::Attribute::Param0)]
                   .primary == Qualifier::PerVertex;
}

IR::Inst* DifferenceVertex(const IR::Inst* inst, const Info& info) {
    if (!inst || inst->GetOpcode() != IR::Opcode::FPSub32) {
        return nullptr;
    }
    auto* vertex = inst->Arg(0).TryInstRecursive();
    auto* origin = inst->Arg(1).TryInstRecursive();
    if (!IsPerVertexRead(vertex, info) || !IsPerVertexRead(origin, info) ||
        vertex->Arg(0) != origin->Arg(0) || vertex->Arg(1) != origin->Arg(1) ||
        origin->Arg(2).U32() != 0 || vertex->Arg(2).U32() == 0 ||
        vertex->Arg(2).U32() > 2) {
        return nullptr;
    }
    return vertex;
}

void ReuseInstruction(IR::Inst& inst, IR::Inst*& previous) {
    if (previous && previous->Flags<u64>() == inst.Flags<u64>()) {
        size_t arg{};
        for (; arg < inst.NumArgs(); ++arg) {
            if (inst.Arg(arg).Resolve() != previous->Arg(arg).Resolve()) {
                break;
            }
        }
        if (arg == inst.NumArgs()) {
            inst.ReplaceUsesWith(IR::Value{previous});
            return;
        }
    }
    previous = &inst;
}

} // namespace

void InterpolationEliminationPass(IR::Program& program) {
    if (program.info.l_stage != LogicalStage::Fragment) {
        return;
    }
    for (IR::Block* const block : program.blocks) {
        std::array<InterpolationComponent, IR::NumParams * 4> components{};
        for (IR::Inst& inst : block->Instructions()) {
            if (IsPerVertexRead(&inst, program.info)) {
                const u32 attr = u32(inst.Arg(0).Attribute()) - u32(IR::Attribute::Param0);
                const u32 comp = inst.Arg(1).U32();
                const u32 vertex = inst.Arg(2).U32();
                ASSERT(comp < 4 && vertex < 3);
                ReuseInstruction(inst, components[attr * 4 + comp].vertices[vertex]);
                continue;
            }
            const bool is_fma = inst.GetOpcode() == IR::Opcode::FPFma32;
            auto* vertex = DifferenceVertex(
                is_fma ? inst.Arg(0).TryInstRecursive() : &inst, program.info);
            if (!vertex) {
                continue;
            }
            const u32 attr = u32(vertex->Arg(0).Attribute()) - u32(IR::Attribute::Param0);
            auto& component = components[attr * 4 + vertex->Arg(1).U32()];
            auto& previous = is_fma ? component.results[vertex->Arg(2).U32() - 1]
                                    : component.differences[vertex->Arg(2).U32() - 1];
            ReuseInstruction(inst, previous);
        }
    }
}

} // namespace Shader::Optimization
