// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <map>
#include <span>
#include <spirv/unified1/spirv.hpp11>

#include "shader_recompiler/backend/spirv/raw_access_chains.h"

namespace Shader::Backend::SPIRV {

void ConvertRawAccessChains(std::vector<u32>& code) {
    struct Access {
        u32 pointer_type{}, base{}, stride{}, index{}, offset{}, alignment{}, raw_id{};
        u32 qualifiers{};
    };
    const u32 bound = code[3];
    std::vector<u32> definitions(bound), strides(bound), memory_users(bound);
    std::vector<bool> blocks(bound), buffer_blocks(bound);
    std::vector<bool> types(bound);
    std::vector<u32> qualifiers(bound);
    std::map<std::pair<u32, u32>, u32> member_offsets;
    std::map<std::pair<u32, u32>, u32> member_qualifiers;
    const auto qualifier = [](u32 decoration) {
        switch (static_cast<spv::Decoration>(decoration)) {
        case spv::Decoration::Restrict:
        case spv::Decoration::Aliased:
        case spv::Decoration::Volatile:
        case spv::Decoration::Coherent:
        case spv::Decoration::NonWritable:
        case spv::Decoration::NonReadable:
            return u32{1} << decoration;
        default:
            return u32{0};
        }
    };
    u32 uint_type{};
    for (size_t pos = 5; pos < code.size(); pos += code[pos] >> 16) {
        const auto op = static_cast<spv::Op>(code[pos] & 0xffff);
        const auto words = std::span{code}.subspan(pos, code[pos] >> 16);
        if (op >= spv::Op::OpTypeVoid && op <= spv::Op::OpTypeForwardPointer) {
            definitions[words[1]] = static_cast<u32>(pos);
            types[words[1]] = true;
        } else if (words.size() >= 3 && words[1] < bound && types[words[1]]) {
            definitions[words[2]] = static_cast<u32>(pos);
        }
        if (op == spv::Op::OpTypeInt && words[2] == 32 && words[3] == 0) {
            uint_type = words[1];
        } else if (op == spv::Op::OpDecorate) {
            qualifiers[words[1]] |= qualifier(words[2]);
            if (words[2] == static_cast<u32>(spv::Decoration::ArrayStride)) {
                strides[words[1]] = words[3];
            } else if (words[2] == static_cast<u32>(spv::Decoration::Block)) {
                blocks[words[1]] = true;
            } else if (words[2] == static_cast<u32>(spv::Decoration::BufferBlock)) {
                buffer_blocks[words[1]] = true;
            }
        } else if (op == spv::Op::OpMemberDecorate) {
            member_qualifiers[{words[1], words[2]}] |= qualifier(words[3]);
            if (words[3] == static_cast<u32>(spv::Decoration::Offset)) {
                member_offsets[{words[1], words[2]}] = words[4];
            }
        } else if (op == spv::Op::OpLoad) {
            ++memory_users[words[3]];
        } else if (op == spv::Op::OpStore) {
            ++memory_users[words[1]];
        }
    }
    const auto instruction = [&](u32 id) {
        const u32 pos = definitions[id];
        return std::span<const u32>{code}.subspan(pos, code[pos] >> 16);
    };
    const auto constant = [&](u32 id, u32& value) {
        const auto inst = instruction(id);
        if (static_cast<spv::Op>(inst[0] & 0xffff) != spv::Op::OpConstant || inst.size() != 4) {
            return false;
        }
        value = inst[3];
        return true;
    };
    std::vector<Access> accesses(bound);
    u32 next_id = bound;
    u32 converted{};
    for (size_t pos = 5; pos < code.size(); pos += code[pos] >> 16) {
        const auto words = std::span{code}.subspan(pos, code[pos] >> 16);
        const auto op = static_cast<spv::Op>(words[0] & 0xffff);
        if ((op != spv::Op::OpAccessChain && op != spv::Op::OpInBoundsAccessChain) ||
            !memory_users[words[2]]) {
            continue;
        }
        const auto pointer = instruction(words[1]);
        const auto storage = static_cast<spv::StorageClass>(pointer[2]);
        if (storage != spv::StorageClass::StorageBuffer && storage != spv::StorageClass::Uniform) {
            continue;
        }
        const auto base = instruction(words[3]);
        if (static_cast<spv::Op>(base[0] & 0xffff) != spv::Op::OpVariable) {
            continue;
        }
        u32 current_type = instruction(base[1])[3];
        if (!(storage == spv::StorageClass::StorageBuffer ? blocks[current_type]
                                                       : buffer_blocks[current_type])) {
            continue;
        }
        Access access{.pointer_type = words[1], .base = words[3],
                      .qualifiers = qualifiers[words[3]] | qualifiers[words[2]]};
        bool valid = true;
        for (u32 i = 4; i < words.size() && valid; ++i) {
            const auto type = instruction(current_type);
            access.qualifiers |= qualifiers[current_type];
            switch (static_cast<spv::Op>(type[0] & 0xffff)) {
            case spv::Op::OpTypeStruct: {
                u32 member{};
                valid = constant(words[i], member) && member < type.size() - 2;
                if (valid) {
                    const auto offset = member_offsets.find({current_type, member});
                    valid = offset != member_offsets.end();
                    if (valid) {
                        access.offset += offset->second;
                        access.qualifiers |= member_qualifiers[{current_type, member}];
                        current_type = type[member + 2];
                    }
                }
                break;
            }
            case spv::Op::OpTypeArray:
            case spv::Op::OpTypeRuntimeArray: {
                const u32 stride = strides[current_type];
                valid = stride != 0 && access.stride == 0;
                if (valid) {
                    access.stride = stride;
                    access.index = words[i];
                    current_type = type[2];
                }
                break;
            }
            case spv::Op::OpTypeVector: {
                u32 component{};
                valid = constant(words[i], component) && component < type[3];
                if (valid) {
                    const auto scalar = instruction(type[2]);
                    access.offset += component * (scalar[2] / 8);
                    current_type = type[2];
                }
                break;
            }
            default:
                valid = false;
                break;
            }
        }
        const auto type = instruction(current_type);
        const bool vector = static_cast<spv::Op>(type[0] & 0xffff) == spv::Op::OpTypeVector;
        const auto scalar = vector ? instruction(type[2]) : type;
        const auto scalar_op = static_cast<spv::Op>(scalar[0] & 0xffff);
        valid &= scalar_op == spv::Op::OpTypeInt || scalar_op == spv::Op::OpTypeFloat;
        if (!valid || access.stride == 0 || current_type != pointer[3]) {
            continue;
        }
        access.alignment = scalar[2] / 8;
        const u32 bytes = access.alignment * (vector ? type[3] : 1);
        const auto index = instruction(access.index);
        const auto index_type = instruction(index[1]);
        if (static_cast<spv::Op>(index_type[0] & 0xffff) != spv::Op::OpTypeInt ||
            index_type[2] != 32 || access.offset % access.alignment != 0 ||
            access.stride % access.alignment != 0 || access.offset > access.stride ||
            bytes > access.stride - access.offset) {
            continue;
        }
        access.raw_id = next_id++;
        accesses[words[2]] = access;
        ++converted;
    }
    if (!converted) {
        return;
    }

    std::vector<u32> globals, annotations;
    const auto emit = [](std::vector<u32>& out, spv::Op op, std::initializer_list<u32> args) {
        out.push_back((static_cast<u32>(args.size() + 1) << 16) | static_cast<u32>(op));
        out.insert(out.end(), args.begin(), args.end());
    };
    if (!uint_type) {
        uint_type = next_id++;
        emit(globals, spv::Op::OpTypeInt, {uint_type, 32, 0});
    }
    std::map<u32, u32> constants;
    const auto make_constant = [&](u32 value) {
        auto [it, added] = constants.try_emplace(value, 0);
        if (added) {
            it->second = next_id++;
            emit(globals, spv::Op::OpConstant, {uint_type, it->second, value});
        }
        return it->second;
    };
    for (auto& access : accesses) {
        if (access.raw_id) {
            access.stride = make_constant(access.stride);
            access.offset = make_constant(access.offset);
            for (u32 decoration = 0; decoration < 32; ++decoration) {
                if (access.qualifiers & (u32{1} << decoration)) {
                    emit(annotations, spv::Op::OpDecorate, {access.raw_id, decoration});
                }
            }
        }
    }

    std::vector<u32> output(code.begin(), code.begin() + 5);
    output.reserve(code.size() + converted * 12 + globals.size());
    bool capability_added{}, extension_added{}, annotations_added{}, globals_added{};
    for (size_t pos = 5; pos < code.size(); pos += code[pos] >> 16) {
        const auto words = std::span{code}.subspan(pos, code[pos] >> 16);
        const auto op = static_cast<spv::Op>(words[0] & 0xffff);
        if (!capability_added && op != spv::Op::OpCapability) {
            emit(output, spv::Op::OpCapability,
                 {static_cast<u32>(spv::Capability::RawAccessChainsNV)});
            capability_added = true;
        }
        if (!extension_added && op != spv::Op::OpCapability && op != spv::Op::OpExtension) {
            constexpr char name[] = "SPV_NV_raw_access_chains";
            std::vector<u32> extension((sizeof(name) + 3) / 4);
            std::memcpy(extension.data(), name, sizeof(name));
            output.push_back((static_cast<u32>(extension.size() + 1) << 16) |
                             static_cast<u32>(spv::Op::OpExtension));
            output.insert(output.end(), extension.begin(), extension.end());
            extension_added = true;
        }
        if (!globals_added && op == spv::Op::OpFunction) {
            output.insert(output.end(), globals.begin(), globals.end());
            globals_added = true;
        }
        if (!annotations_added && op >= spv::Op::OpTypeVoid &&
            op <= spv::Op::OpTypeForwardPointer) {
            output.insert(output.end(), annotations.begin(), annotations.end());
            annotations_added = true;
        }
        if (op == spv::Op::OpLoad || op == spv::Op::OpStore) {
            const u32 pointer_position = op == spv::Op::OpLoad ? 3 : 1;
            const auto& access = accesses[words[pointer_position]];
            if (access.raw_id) {
                const size_t start = output.size();
                output.insert(output.end(), words.begin(), words.end());
                output[start + pointer_position] = access.raw_id;
                const u32 mask_position = op == spv::Op::OpLoad ? 4 : 3;
                if (words.size() == mask_position) {
                    output.push_back(static_cast<u32>(spv::MemoryAccessMask::Aligned));
                    output.push_back(access.alignment);
                } else if (!(words[mask_position] &
                             static_cast<u32>(spv::MemoryAccessMask::Aligned))) {
                    output[start + mask_position] |=
                        static_cast<u32>(spv::MemoryAccessMask::Aligned);
                    output.insert(output.begin() + start + mask_position + 1, access.alignment);
                }
                output[start] = (static_cast<u32>(output.size() - start) << 16) |
                                static_cast<u32>(op);
                continue;
            }
        }
        output.insert(output.end(), words.begin(), words.end());
        if (op == spv::Op::OpAccessChain || op == spv::Op::OpInBoundsAccessChain) {
            const auto& access = accesses[words[2]];
            if (access.raw_id) {
                emit(output, spv::Op::OpRawAccessChainNV,
                     {access.pointer_type, access.raw_id, access.base, access.stride, access.index,
                      access.offset,
                      static_cast<u32>(
                          spv::RawAccessChainOperandsMask::RobustnessPerComponentNV)});
            }
        }
    }
    output[3] = next_id;
    code = std::move(output);
}

} // namespace Shader::Backend::SPIRV
