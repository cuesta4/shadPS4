// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>

namespace Common {

/// Returns true when both containers hold byte-for-byte identical elements.
/// Only valid for containers of trivially-copyable elements without padding.
template <typename Vector>
[[nodiscard]] bool EqualVectorBytes(const Vector& lhs, const Vector& rhs) {
    using Value = typename Vector::value_type;
    return lhs.size() == rhs.size() &&
           (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(Value)) == 0);
}

/// Copies all elements of src into dst, replacing its previous contents. Works across container
/// types (e.g. small_vector -> static_vector) as long as the value types match.
template <typename Dst, typename Src>
void CopyVector(Dst& dst, const Src& src) {
    dst.clear();
    dst.insert(dst.end(), src.begin(), src.end());
}

} // namespace Common
