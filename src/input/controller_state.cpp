// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <utility>

#include <SDL3/SDL_gamepad.h>

#include "common/types.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/pad/pad.h"
#include "input/controller.h"

namespace Input {

using Libraries::Pad::OrbisPadButtonDataOffset;

void State::OnButton(OrbisPadButtonDataOffset button, bool is_pressed) {
    if (is_pressed) {
        buttonsState |= button;
    } else {
        buttonsState &= ~button;
    }
}

void State::OnAxis(Axis axis, int value, u64 timestamp, bool smooth) {
    const auto index = std::to_underlying(axis);
    axes[index] = axis_smoothing_end_values[index];

    axis_smoothing_start_times[index] = timestamp;
    axis_smoothing_start_values[index] = axes[index];
    axis_smoothing_end_values[index] = value;
    axis_smoothing_flags[index] = smooth;
    const auto toggle = [&](const auto button) {
        if (value > 0) {
            buttonsState |= button;
        } else {
            buttonsState &= ~button;
        }
    };
    switch (axis) {
    case Axis::TriggerLeft:
        toggle(OrbisPadButtonDataOffset::L2);
        break;
    case Axis::TriggerRight:
        toggle(OrbisPadButtonDataOffset::R2);
        break;
    default:
        break;
    }
}

void State::OnTouchpad(int touch_index, bool is_down, float x, float y) {
    touchpad[touch_index].state = is_down;
    touchpad[touch_index].x = static_cast<u16>(x * 1920);
    touchpad[touch_index].y = static_cast<u16>(y * 941);
}

void State::OnGyro(const float gyro[3]) {
    angularVelocity.x = gyro[0];
    angularVelocity.y = gyro[1];
    angularVelocity.z = gyro[2];
}

void State::OnAccel(const float accel[3]) {
    acceleration.x = accel[0];
    acceleration.y = accel[1];
    acceleration.z = accel[2];
}

void State::UpdateAxisSmoothing(u64 timestamp) {
    for (int i = 0; i < std::to_underlying(Axis::AxisMax); ++i) {
        if (!axis_smoothing_flags[i] || std::abs(axes[i] - axis_smoothing_end_values[i]) < 16) {
            axes[i] = axis_smoothing_end_values[i];
            continue;
        }
        const f32 t = std::clamp(
            (timestamp - axis_smoothing_start_times[i]) / f32{axis_smoothing_time}, 0.f, 1.f);
        axes[i] = s32(axis_smoothing_start_values[i] * (1 - t) + axis_smoothing_end_values[i] * t);
    }
}

GameController::GameController() : m_states_queue(64) {}

State GameController::ReadState() {
    std::lock_guard lock{m_state_mutex};
    return m_state;
}

int GameController::ReadStates(State* states, int states_num) {
    std::lock_guard lock{m_state_mutex};
    if (states_num <= 0) {
        return 0;
    }

    if (states_num == 1) {
        // Retained history can make a later multi-sample read return up to 64 stale reports, so
        // mixed single- and multi-sample reads require dedicated tests.
        states[0] = m_state;
        return 1;
    }

    int read_count = 0;
    while (read_count < states_num) {
        auto state = m_states_queue.Pop();
        if (!state) {
            break;
        }
        states[read_count++] = std::move(*state);
    }
    return read_count;
}

void GameController::Button(OrbisPadButtonDataOffset button, bool is_pressed) {
    std::lock_guard lock{m_state_mutex};
    m_state.OnButton(button, is_pressed);
    PushStateLocked();
}

void GameController::Axis(Input::Axis axis, int value, bool smooth) {
    std::lock_guard lock{m_state_mutex};
    const u64 timestamp = Libraries::Kernel::sceKernelGetProcessTime();
    m_state.OnAxis(axis, value, timestamp, smooth);
    PushStateLocked(timestamp);
}

void GameController::UpdateGyro(const float gyro[3]) {
    std::lock_guard lock{m_state_mutex};
    std::memcpy(gyro_buf, gyro, sizeof(gyro_buf));
}

void GameController::UpdateAcceleration(const float acceleration[3]) {
    std::lock_guard lock{m_state_mutex};
    std::memcpy(accel_buf, acceleration, sizeof(accel_buf));
}

void GameController::PollState() {
    std::lock_guard lock{m_state_mutex};
    PushStateLocked();
}

void GameController::ResetOrientation() {
    std::lock_guard lock{m_state_mutex};
    m_state.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    m_last_orientation_update = 0;
    PushStateLocked();
}

void GameController::SetTouchpadState(int touch_index, bool touch_down, float x, float y) {
    if (touch_index < 0 || touch_index >= 2) {
        return;
    }

    std::lock_guard lock{m_state_mutex};
    const u64 timestamp = Libraries::Kernel::sceKernelGetProcessTime();
    const bool was_pressed = m_state.touchpad[0].state || m_state.touchpad[1].state;
    auto& touch = m_state.touchpad[touch_index];
    if (touch_down && !touch.state) {
        touch.ID = m_next_touch_id;
        m_next_touch_id = m_next_touch_id == 127 ? 1 : m_next_touch_id + 1;
    }
    m_state.OnTouchpad(touch_index, touch_down, x, y);
    const bool is_pressed = m_state.touchpad[0].state || m_state.touchpad[1].state;
    if (!was_pressed && is_pressed) {
        m_touch_down_timestamp = timestamp;
    } else if (was_pressed && !is_pressed) {
        m_touch_down_timestamp = 0;
    }
    PushStateLocked(timestamp);
}

void GameControllers::CalculateOrientation(const Libraries::Pad::OrbisFVector3& angular_velocity,
                                           float delta_time,
                                           const Libraries::Pad::OrbisFQuaternion& last_orientation,
                                           Libraries::Pad::OrbisFQuaternion& orientation) {
    if (delta_time > 1.0f) {
        orientation = last_orientation;
        return;
    }
    Libraries::Pad::OrbisFQuaternion q = last_orientation;
    const Libraries::Pad::OrbisFQuaternion omega = {angular_velocity.x, angular_velocity.y,
                                                    angular_velocity.z, 0.0f};

    const Libraries::Pad::OrbisFQuaternion q_omega = {
        q.w * omega.x + q.x * omega.w + q.y * omega.z - q.z * omega.y,
        q.w * omega.y + q.y * omega.w + q.z * omega.x - q.x * omega.z,
        q.w * omega.z + q.z * omega.w + q.x * omega.y - q.y * omega.x,
        q.w * omega.w - q.x * omega.x - q.y * omega.y - q.z * omega.z};

    const Libraries::Pad::OrbisFQuaternion q_dot = {0.5f * q_omega.x, 0.5f * q_omega.y,
                                                    0.5f * q_omega.z, 0.5f * q_omega.w};

    q.x += q_dot.x * delta_time;
    q.y += q_dot.y * delta_time;
    q.z += q_dot.z * delta_time;
    q.w += q_dot.w * delta_time;

    const float norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    q.x /= norm;
    q.y /= norm;
    q.z /= norm;
    q.w /= norm;
    orientation = q;
}

void GameController::ConnectController(SDL_Gamepad* pad, StatePublication publication) {
    std::lock_guard lock{m_state_mutex};
    m_sdl_gamepad = pad;
    m_states_queue.Clear();
    if (!m_state.connected) {
        ++m_state.connected_count;
        if (m_state.connected_count == 0) {
            m_state.connected_count = 1;
        }
    }
    m_state.connected = true;
    m_last_orientation_update = 0;
    if (publication == StatePublication::Publish) {
        PushStateLocked();
    }
}

void GameController::DisconnectController() {
    std::lock_guard lock{m_state_mutex};
    m_states_queue.Clear();
    m_sdl_gamepad = nullptr;

    const u8 connected_count = m_state.connected_count;
    m_state = {};
    m_state.connected_count = connected_count;
    std::fill(gyro_buf, gyro_buf + 3, 0.0f);
    std::fill(accel_buf, accel_buf + 3, 0.0f);
    accel_buf[1] = 9.81f;
    m_next_touch_id = 1;
    m_touch_down_timestamp = 0;
    m_state.connected = false;
    m_last_orientation_update = 0;
    PushStateLocked();
}

void GameController::UpdateOrientationLocked(u64 timestamp) {
    if (m_last_orientation_update == 0 || timestamp <= m_last_orientation_update) {
        m_last_orientation_update = timestamp;
        return;
    }
    const float delta_time =
        static_cast<float>(timestamp - m_last_orientation_update) / 1'000'000.f;
    Libraries::Pad::OrbisFQuaternion orientation{};
    GameControllers::CalculateOrientation(m_state.angularVelocity, delta_time, m_state.orientation,
                                          orientation);
    m_state.orientation = orientation;
    m_last_orientation_update = timestamp;
}

void GameController::PushStateLocked(u64 timestamp) {
    if (timestamp == 0) {
        timestamp = Libraries::Kernel::sceKernelGetProcessTime();
    }
    m_state.UpdateAxisSmoothing(timestamp);
    m_state.OnGyro(gyro_buf);
    m_state.OnAccel(accel_buf);
    UpdateOrientationLocked(timestamp);
    m_state.time = timestamp;
    m_state.touch_time_since_held_down =
        m_touch_down_timestamp == 0 ? 0 : timestamp - m_touch_down_timestamp;
    m_states_queue.Push(m_state);
}

} // namespace Input
