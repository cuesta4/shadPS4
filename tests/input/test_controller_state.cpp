// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "core/libraries/pad/pad.h"
#include "input/controller.h"
#include "tests/stubs/kernel_stub.h"

namespace {

using Input::GameController;
using Input::State;
using Input::StatePublication;
using Libraries::Pad::OrbisPadButtonDataOffset;

class ControllerStateTest : public testing::Test {
protected:
    void SetUp() override {
        Libraries::Kernel::TestSetProcessTime(100);
    }

    void TearDown() override {
        Libraries::Kernel::TestResetProcessTime();
    }
};

TEST_F(ControllerStateTest, StartsDisconnected) {
    GameController controller;

    const State state = controller.ReadState();

    EXPECT_FALSE(state.connected);
    EXPECT_EQ(state.connected_count, 0);
}

TEST_F(ControllerStateTest, InitialConnectionCanSuppressPreKernelReport) {
    GameController controller;
    controller.ConnectController(nullptr, StatePublication::Suppress);

    const State current = controller.ReadState();
    std::array<State, 2> history{};

    EXPECT_TRUE(current.connected);
    EXPECT_EQ(current.connected_count, 1);
    EXPECT_EQ(controller.ReadStates(history.data(), history.size()), 0);
}

TEST_F(ControllerStateTest, MultiSampleReadPreservesReportOrder) {
    GameController controller;
    controller.ConnectController(nullptr, StatePublication::Suppress);

    Libraries::Kernel::TestSetProcessTime(200);
    controller.Button(OrbisPadButtonDataOffset::Cross, true);
    Libraries::Kernel::TestSetProcessTime(300);
    controller.Button(OrbisPadButtonDataOffset::Cross, false);

    std::array<State, 2> history{};
    ASSERT_EQ(controller.ReadStates(history.data(), history.size()), 2);
    EXPECT_EQ(history[0].time, 200);
    EXPECT_NE(history[0].buttonsState & OrbisPadButtonDataOffset::Cross,
              OrbisPadButtonDataOffset{});
    EXPECT_EQ(history[1].time, 300);
    EXPECT_EQ(history[1].buttonsState & OrbisPadButtonDataOffset::Cross,
              OrbisPadButtonDataOffset{});
}

TEST_F(ControllerStateTest, SingleSampleReadReturnsCurrentStateWithoutConsumingHistory) {
    GameController controller;
    controller.ConnectController(nullptr, StatePublication::Suppress);

    Libraries::Kernel::TestSetProcessTime(200);
    controller.Button(OrbisPadButtonDataOffset::Cross, true);
    Libraries::Kernel::TestSetProcessTime(300);
    controller.Button(OrbisPadButtonDataOffset::Cross, false);

    State current{};
    ASSERT_EQ(controller.ReadStates(&current, 1), 1);
    EXPECT_EQ(current.time, 300);

    std::array<State, 2> history{};
    ASSERT_EQ(controller.ReadStates(history.data(), history.size()), 2);
    EXPECT_EQ(history[0].time, 200);
    EXPECT_EQ(history[1].time, 300);
}

TEST_F(ControllerStateTest, DisconnectedControllerContinuesPublishingNeutralReports) {
    GameController controller;
    controller.ConnectController(nullptr, StatePublication::Suppress);

    Libraries::Kernel::TestSetProcessTime(200);
    controller.DisconnectController();
    std::array<State, 2> history{};
    ASSERT_EQ(controller.ReadStates(history.data(), history.size()), 1);
    EXPECT_FALSE(history[0].connected);
    EXPECT_EQ(history[0].time, 200);

    Libraries::Kernel::TestSetProcessTime(300);
    controller.PollState();
    Libraries::Kernel::TestSetProcessTime(400);
    controller.PollState();

    ASSERT_EQ(controller.ReadStates(history.data(), history.size()), 2);
    EXPECT_FALSE(history[0].connected);
    EXPECT_EQ(history[0].time, 300);
    EXPECT_FALSE(history[1].connected);
    EXPECT_EQ(history[1].time, 400);
}

TEST_F(ControllerStateTest, ReconnectionClearsHistoryAndIncrementsConnectionCount) {
    GameController controller;
    controller.ConnectController(nullptr, StatePublication::Suppress);
    controller.Button(OrbisPadButtonDataOffset::Cross, true);
    controller.DisconnectController();
    controller.ConnectController(nullptr, StatePublication::Suppress);

    const State current = controller.ReadState();
    std::array<State, 2> history{};

    EXPECT_TRUE(current.connected);
    EXPECT_EQ(current.connected_count, 2);
    EXPECT_EQ(current.buttonsState, OrbisPadButtonDataOffset{});
    EXPECT_EQ(current.axes[static_cast<int>(Input::Axis::LeftX)], 128);
    EXPECT_FALSE(current.touchpad[0].state);
    EXPECT_FALSE(current.touchpad[1].state);
    EXPECT_EQ(controller.ReadStates(history.data(), history.size()), 0);
}

} // namespace
