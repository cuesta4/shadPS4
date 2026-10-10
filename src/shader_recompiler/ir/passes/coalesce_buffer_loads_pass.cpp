// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"

namespace Shader::Optimization {

void CoalesceBufferLoadsPass(IR::Program& program) {
    const auto split_index = [](IR::Value index) {
        if (index.IsImmediate()) {
            return std::pair{IR::Value{}, index.U32()};
        }
        const auto* inst = index.Inst();
        if (inst->GetOpcode() == IR::Opcode::IAdd32 && inst->Arg(1).IsImmediate()) {
            return std::pair{inst->Arg(0), inst->Arg(1).U32()};
        }
        return std::pair{index, u32{0}};
    };
    for (auto* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            if (it->GetOpcode() != IR::Opcode::ReadConstBuffer) {
                continue;
            }
            const auto [base, offset] = split_index(it->Arg(1));
            const auto handle = it->Arg(0);
            const auto flags = it->Flags<IR::BufferInstInfo>();
            std::array<IR::Inst*, 4> group{&*it};
            u32 count = 1;
            for (auto next = std::next(it); next != block->end() && count < 4; ++next) {
                if (next->MayHaveSideEffects()) {
                    break;
                }
                if (next->GetOpcode() == IR::Opcode::ReadConstBuffer) {
                    const auto [next_base, next_offset] = split_index(next->Arg(1));
                    if (next->Flags<IR::BufferInstInfo>().raw != flags.raw ||
                        next->Arg(0) != handle || next_base != base ||
                        next_offset != offset + count) {
                        break;
                    }
                    group[count++] = &*next;
                }
            }
            if (count < 2) {
                continue;
            }
            IR::IREmitter ir{*block, it};
            const auto vector = ir.LoadBufferU32(count, handle, it->Arg(1), flags);
            for (u32 i = 0; i < count; ++i) {
                group[i]->ReplaceUsesWith(ir.CompositeExtract(vector, i));
            }
        }
    }
}

} // namespace Shader::Optimization
