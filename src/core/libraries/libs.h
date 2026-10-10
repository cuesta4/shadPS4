// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/host_call.h"

#define LIB_FUNCTION(nid, lib, libversion, mod, function)                                          \
    do {                                                                                              \
        Libraries::RegisterHLEFunction(sym, nid, lib, libversion, mod,                             \
                                       reinterpret_cast<u64>(HOST_CALL(function)));                \
    } while (0)

#define LIB_OBJ(nid, lib, libversion, mod, obj)                                                    \
    do {                                                                                              \
        Libraries::RegisterHLEObject(sym, nid, lib, libversion, mod, reinterpret_cast<u64>(obj));  \
    } while (0)

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries {

void RegisterHLEFunction(Core::Loader::SymbolsResolver* sym, const char* nid, const char* library,
                         u16 library_version, const char* module, u64 address);
void RegisterHLEObject(Core::Loader::SymbolsResolver* sym, const char* nid, const char* library,
                       u16 library_version, const char* module, u64 address);

void InitHLELibs(Core::Loader::SymbolsResolver* sym);

} // namespace Libraries
