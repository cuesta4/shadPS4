// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/pm4_sync.h"

namespace {

AmdGpu::PM4CmdWaitRegMem MakeMemoryWait(
    const VAddr address, const u32 reference, const u32 mask = 0xFFFFFFFF,
    const AmdGpu::PM4CmdWaitRegMem::Function function =
        AmdGpu::PM4CmdWaitRegMem::Function::Equal) {
    AmdGpu::PM4CmdWaitRegMem wait{
        .header = AmdGpu::PM4Type3Header{AmdGpu::PM4ItOpcode::WaitRegMem, 5},
        .raw = 0,
        .poll_addr_lo_raw = 0,
        .poll_addr_hi = 0,
        .ref = reference,
        .mask = mask,
        .poll_interval = 0,
    };
    wait.function.Assign(function);
    wait.mem_space.Assign(AmdGpu::PM4CmdWaitRegMem::MemSpace::Memory);
    wait.poll_addr_lo.Assign(static_cast<u32>(address >> 2));
    wait.poll_addr_hi = static_cast<u32>(address >> 32);
    return wait;
}

std::span<const u32> AsWords(const AmdGpu::PM4CmdWaitRegMem& wait) {
    return {reinterpret_cast<const u32*>(&wait), sizeof(wait) / sizeof(u32)};
}

TEST(Pm4SyncTest, RecognizesSatisfiedImmediateMemoryWait) {
    alignas(u32) u32 label{};
    const VAddr address = reinterpret_cast<VAddr>(&label);
    const auto wait = MakeMemoryWait(address, 0x34, 0xFF);

    EXPECT_TRUE(AmdGpu::WaitMatchesImmediateSignal(AsWords(wait), address, 0x1234));
}

TEST(Pm4SyncTest, RejectsUnsatisfiedComparison) {
    alignas(u32) u32 label{};
    const VAddr address = reinterpret_cast<VAddr>(&label);
    const auto wait =
        MakeMemoryWait(address, 8, 0xFFFFFFFF,
                       AmdGpu::PM4CmdWaitRegMem::Function::GreaterThan);

    EXPECT_FALSE(AmdGpu::WaitMatchesImmediateSignal(AsWords(wait), address, 7));
}

TEST(Pm4SyncTest, RejectsDifferentAddressAndRegisterWaits) {
    alignas(u32) u32 labels[2]{};
    const VAddr address = reinterpret_cast<VAddr>(&labels[0]);
    auto wait = MakeMemoryWait(reinterpret_cast<VAddr>(&labels[1]), 1);

    EXPECT_FALSE(AmdGpu::WaitMatchesImmediateSignal(AsWords(wait), address, 1));

    wait = MakeMemoryWait(address, 1);
    wait.mem_space.Assign(AmdGpu::PM4CmdWaitRegMem::MemSpace::Register);
    EXPECT_FALSE(AmdGpu::WaitMatchesImmediateSignal(AsWords(wait), address, 1));
}

TEST(Pm4SyncTest, RejectsTruncatedOrMalformedPackets) {
    alignas(u32) u32 label{};
    const VAddr address = reinterpret_cast<VAddr>(&label);
    auto wait = MakeMemoryWait(address, 1);
    const auto words = AsWords(wait);

    EXPECT_FALSE(AmdGpu::WaitMatchesImmediateSignal(words.first(words.size() - 1), address, 1));

    wait.header.count.Assign(4);
    EXPECT_FALSE(AmdGpu::WaitMatchesImmediateSignal(AsWords(wait), address, 1));
}

} // namespace
