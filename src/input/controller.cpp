// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_set>
#include <SDL3/SDL.h>
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "core/libraries/system/userservice.h"
#include "core/user_settings.h"
#include "input/controller.h"

namespace Input {

void GameController::SetLightBarRGB(u8 const r, u8 const g, u8 const b) {
    if (override_colour.has_value()) {
        return;
    }
    colour = {r, g, b};
    if (m_sdl_gamepad != nullptr) {
        SDL_SetGamepadLED(m_sdl_gamepad, r, g, b);
    }
}

void GameController::SetLightBarRGB(Colour const c) {
    SetLightBarRGB(c.r, c.g, c.b);
}

Colour GameController::GetLightBarRGB() {
    return colour;
}

void GameController::PollLightColour() {
    if (m_sdl_gamepad != nullptr) {
        SDL_SetGamepadLED(m_sdl_gamepad, colour.r, colour.g, colour.b);
    }
}

void GameControllers::ResetLightbarColors() {
    for (auto& c : controllers) {
        auto const* u = UserManagement.GetUserByID(c->user_id);
        if (!u || !c->m_sdl_gamepad) {
            continue;
        }
        auto const i = u->user_color - 1;
        if (i < 0 || i > 3) {
            continue;
        }
        auto const& col = g_user_colours[i];
        c->override_colour = std::nullopt;
        c->SetLightBarRGB(col);
    }
}

bool GameController::SetVibration(u8 smallMotor, u8 largeMotor) {
    if (m_sdl_gamepad != nullptr) {
        return SDL_RumbleGamepad(m_sdl_gamepad, (smallMotor / 255.0f) * 0xFFFF,
                                 (largeMotor / 255.0f) * 0xFFFF, -1);
    }
    return true;
}

static bool is_first_check = true;

void GameControllers::TryOpenSDLControllers(StatePublication publication) {
    using namespace Libraries::UserService;
    int controller_count;
    SDL_JoystickID* new_joysticks = SDL_GetGamepads(&controller_count);
    LOG_INFO(Input, "{} controllers are currently connected", controller_count);

    std::unordered_set<SDL_JoystickID> assigned_ids;
    std::array<bool, 4> slot_taken{false, false, false, false};

    for (int i = 0; i < 4; i++) {
        SDL_Gamepad* pad = controllers[i]->m_sdl_gamepad;
        if (pad) {
            SDL_JoystickID id = SDL_GetGamepadID(pad);
            bool still_connected = false;
            for (int j = 0; j < controller_count; j++) {
                if (new_joysticks[j] == id) {
                    still_connected = true;
                    assigned_ids.insert(id);
                    slot_taken[i] = true;
                    break;
                }
            }
            if (!still_connected) {
                SDL_CloseGamepad(pad);
                controllers[i]->DisconnectController();
                controllers[i]->user_id = -1;
                slot_taken[i] = false;
            }
        }
    }

    for (int j = 0; j < controller_count; j++) {
        SDL_JoystickID id = new_joysticks[j];
        if (assigned_ids.contains(id))
            continue;

        SDL_Gamepad* pad = SDL_OpenGamepad(id);
        if (!pad) {
            continue;
        }

        for (int i = 0; i < 4; i++) {
            if (!slot_taken[i]) {
                auto u = UserManagement.GetUserByPlayerIndex(i + 1);
                if (!u) {
                    LOG_INFO(Input, "User {} not found", i + 1);
                    continue; // for now, if you don't specify who Player N is in the config,
                              // Player N won't be registered at all
                }
                auto* c = controllers[i];
                LOG_INFO(Input, "Gamepad registered for slot {}! Handle: {}", i,
                         SDL_GetGamepadID(pad));
                slot_taken[i] = true;
                c->user_id = u->user_id;
                UserManagement.LoginUser(u, i + 1);
                c->ConnectController(pad, publication);
                if (EmulatorSettings.IsMotionControlsEnabled()) {
                    if (SDL_SetGamepadSensorEnabled(c->m_sdl_gamepad, SDL_SENSOR_GYRO, true)) {
                        const float poll_rate =
                            SDL_GetGamepadSensorDataRate(c->m_sdl_gamepad, SDL_SENSOR_GYRO);
                        LOG_INFO(Input, "Gyro initialized, poll rate: {}", poll_rate);
                    } else {
                        LOG_ERROR(Input, "Failed to initialize gyro controls for gamepad {}",
                                  c->user_id);
                    }
                    if (SDL_SetGamepadSensorEnabled(c->m_sdl_gamepad, SDL_SENSOR_ACCEL, true)) {
                        const float poll_rate =
                            SDL_GetGamepadSensorDataRate(c->m_sdl_gamepad, SDL_SENSOR_ACCEL);
                        LOG_INFO(Input, "Accel initialized, poll rate: {}", poll_rate);
                    } else {
                        LOG_ERROR(Input, "Failed to initialize accel controls for gamepad {}",
                                  c->user_id);
                    }
                }
                break;
            }
        }
    }
    if (is_first_check) [[unlikely]] {
        is_first_check = false;
        if (controller_count == 0) {
            auto u = UserManagement.GetUserByPlayerIndex(1);
            controllers[0]->user_id = u->user_id;
            controllers[0]->ConnectController(nullptr, publication);
            UserManagement.LoginUser(u, 1);
        }
    }
    SDL_free(new_joysticks);
}
u8 GameControllers::GetGamepadIndexFromJoystickId(SDL_JoystickID id) {
    auto g = SDL_GetGamepadFromID(id);
    ASSERT(g != nullptr);
    for (int i = 0; i < 5; i++) {
        if (controllers[i]->m_sdl_gamepad == g) {
            return i;
        }
    }
    // LOG_TRACE(Input, "Gamepad index: {}", index);
    return -1;
}

std::optional<u8> GameControllers::GetControllerIndexFromUserID(s32 user_id) {
    auto const u = UserManagement.GetUserByID(user_id);
    if (!u) {
        return std::nullopt;
    }
    return u->player_index - 1;
}

std::optional<u8> GameControllers::GetControllerIndexFromControllerID(s32 controller_id) {
    if (controller_id < 1 || controller_id > 5) {
        return std::nullopt;
    }
    return controller_id - 1;
}

} // namespace Input
