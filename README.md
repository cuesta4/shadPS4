<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# SHADPS4 - TUTZ-GOW/TUTZ-EMU BUILDS

Unofficial Windows builds of [shadPS4](https://github.com/shadps4-emu/shadPS4),
focused on making **God of War III** and **Dead Nation** more playable while the
upstream project continues to evolve.

These builds are experimental and are not intended to replace upstream shadPS4.

> [!CAUTION]
> ## Unofficial builds — please read
>
> These binaries are **not official shadPS4 releases** and are not endorsed by the
> upstream developers.
>
> - Do **not** contact upstream developers about these builds.
> - Do **not** report bugs caused by these builds in the official shadPS4 repository.
> - Do **not** assume that a problem seen here exists in upstream shadPS4.
> - Reproduce issues with an official upstream build before opening an upstream report.
> - Build-specific feedback belongs in this fork, together with the exact build and game.

## Downloads

- [**TUTZ-EMU**](https://github.com/cuesta4/shadPS4/releases/tag/tutz-emu-2026-08-23) —
  the general-purpose build. Use it for Dead Nation and most other games.
- [**TUTZ-GOW**](https://github.com/cuesta4/shadPS4/releases/tag/tutz-gow-2026-08-23) —
  the God of War III build. It contains a game-focused GPU synchronization path and
  has **not** been tested as a general-purpose build.
- [**TUTZ-UI**](https://github.com/cuesta4/shadPS4/releases/tag/tutz-ui-2026-08-23) —
  the custom launcher executable. This release contains the executable only; use it
  with the official QtLauncher package.

The releases contain the compiled binaries for easier access. Feel free to fork, build and hack around at will.

## Features common to both builds

- Coherent aliased-resource tracking across the buffer and texture caches.
- Guest VBlank timing decoupled from Vulkan presentation timing.
- Semantically inactive depth/stencil attachments omitted from graphics pipelines.
- Asynchronous graphics shader and pipeline compilation, configurable per game,
  using six compiler workers.
- More persistent shader-cache storage and background cache I/O.
- Linear-readback, buffer-cache, PM4, draw, and rasterizer hot-path optimizations.
- Reduced redundant synchronization, logging, lookups, and allocations in frequently
  executed CPU/GPU paths; the resulting code is also easier for Clang to optimize.
- Configurable `app0`/HDD read bandwidth and fixed-time loading via **Disable Time Dilation**.
- Compatibility fixes for shader interfaces and upstream clip-plane changes.

## TUTZ-GOW-only features

- GPU-side virtual fencing for eligible GPU event signals.
- Deferred GPU completion labels/writebacks, avoiding unnecessary CPU waits and fences.
- GPU authority tracking and lazy materialization optimized for linear readbacks.
- Additional God of War III–focused synchronization and readback fast paths.

The TUTZ-GOW path is deliberately isolated from TUTZ-EMU because it changes the
GPU/CPU synchronization contract and has only been validated with God of War III.

## Recommended launcher setup

Download the [official QtLauncher package](https://github.com/shadps4-emu/shadPS4-qtlauncher/releases/latest),
replace its `shadPS4QtLauncher.exe` with the executable from the
[TUTZ-UI release](https://github.com/cuesta4/shadPS4/releases/tag/tutz-ui-2026-08-23),
and keep the emulator executables in the launcher version directory, for example:

```text
versions/
├── tutz-emu/
│   └── tutz-emu.exe
└── tutz-gow/
    └── tutz-gow.exe
```

God of War 3 Remastered settings:

1. Right-click the game in the launcher.
2. Open **Game-specific Settings... → Configure Game-specific Settings**.
3. Open the **Experimental** tab.
4. Under **HDD Read Speed**, set the bandwidth to **at least 75 MiB/s**.
5. Enable **Async Shader Recompiling** under 
   **Shader Cache**. Restart the game after changing it.
6. Keep **Readbacks Mode** disabled and enable
   **Readback Linear Images**. 

Other info:
The HDD speed might help other games, but at cost of loading speeds. 
The `Disable Time Dilation` mitigates that, but it might regress games.

Async shader compilation can reduce shader-compilation stutter, but it may expose
game-specific visual or stability problems. Disable it if it bother you too much.

## Scope and limitations

- These are Windows x64 builds compiled with Clang, Release optimizations, and ThinLTO. Feel free to clone the code and build yourself.
- TUTZ-GOW is a build targeting maximum performance for God of War III. You may try it with other games, but they might break.
- Neither build is a promise of perfect performance, accuracy, or visual correctness.

## Showcase

https://www.youtube.com/watch?v=tWWtB59fE7o

## Source and license

This fork is based on the open-source [shadPS4 project](https://github.com/shadps4-emu/shadPS4)
and remains available under the [GPL-2.0-or-later license](LICENSE).
