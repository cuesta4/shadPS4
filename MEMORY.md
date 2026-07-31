<!-- SPDX-FileCopyrightText: Copyright 2026 Arthur -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

# Project Memory

## Branch integration

- Preserve the custom commits already carried by `tutz-emu` when transplanting fixes from
  upstream-oriented PR branches.
- Personal build scripts, archives, and this file are internal and must remain outside commits.
- Use the project-local `build-fast.ps1`, which imports
  `D:\CODING\SDKs\EWDK\EWDK-LLVM.env`, for incremental C++ builds.

## Host shader cache generations

- Cache runtime-compiled host shaders globally rather than duplicating them in each title cache.
- Identify a generation from the fully expanded source, shader stage, glslang version, and an
  explicit compiler revision. Store permutations separately using the ordered define set.
- Activating a generation removes older generations of that named shader; adding a permutation
  preserves the other permutations in the active generation.
- Validate cached SPIR-V before module creation and publish new permutations through temporary
  files so incomplete writes are never consumed.
