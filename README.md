<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# TUTZ builds for shadPS4

Unofficial Windows x64 builds of [shadPS4](https://github.com/shadps4-emu/shadPS4).
TUTZ-EMU is the general-purpose build; TUTZ-GOW focuses on God of War III. TUTZ-UI
is the launcher used to manage either emulator build.

> [!IMPORTANT]
> These are experimental fork builds, not official shadPS4 releases. For problems
> with these builds, open an issue in this repository with the game, build and
> settings you used. Reproduce a problem on an official build before reporting it
> upstream.

## Downloads

| Download | Use it for |
| --- | --- |
| [TUTZ-EMU](https://github.com/cuesta4/shadPS4/releases/tag/tutz-emu-2026-08-23) | Dead Nation and other games; start here unless you need the God of War III path. |
| [TUTZ-GOW](https://github.com/cuesta4/shadPS4/releases/tag/tutz-gow-2026-08-23) | God of War III, with additional GPU synchronization and upload optimizations. |
| [TUTZ-UI](https://github.com/cuesta4/shadPS4/releases/tag/tutz-ui-2026-08-23) | The custom QtLauncher executable used with either build. |

Each emulator release contains its `.exe` file. TUTZ-UI contains the launcher
`.exe`, not the Qt libraries or other files in the official launcher package.

## Install and select a build

1. Download and extract the [official QtLauncher package](https://github.com/shadps4-emu/shadPS4-qtlauncher/releases/latest).
   Keep all files from that package together.
2. Download `shadPS4QtLauncher.exe` from TUTZ-UI and replace the executable in
   the extracted launcher folder. Keep the Qt libraries from the official package.
3. Download `tutz-emu.exe` and `tutz-gow.exe`. Save each in a permanent folder;
   the launcher can use either executable from its current location.
4. Run `shadPS4QtLauncher.exe`. Open **Version Manager → Add Custom**, choose
   `tutz-emu.exe`, and name it **TUTZ-EMU**. Repeat for `tutz-gow.exe` and name
   it **TUTZ-GOW**.
5. Select the emulator you want in the launcher's version selector. Use TUTZ-EMU
   for other games and TUTZ-GOW for God of War III. You can switch at any time;
   the builds stay separate.

Game files are not included. Add your own dumped game through the launcher as
you would with official shadPS4.

## God of War III settings

Select **TUTZ-GOW**. Right-click the game, open
**Game-specific Settings... → Configure Game-specific Settings**, then the
**Experimental** tab:

1. Enable **Async Shader Recompiling** under **Shader Cache**.
2. Set **Readbacks Mode** to **Disabled** and turn on **Enable Readback Linear Images**.
3. If you encounter texture corruption, use one of the options below.

Restart the game after changing these settings. Async compilation may reduce
shader stutter, but disable it for this game if it causes visual or stability
issues.

### Texture corruption options

- **HDD setting:** Set **HDD Read Speed** to **75 MiB/s**. This can affect
  loading times. **Disable Time Dilation** keeps the simulated delay tied to
  real time when emulation slows down; use it only if that behavior suits the
  game.
- **Patch:** For CUSA01623 or CUSA01715, app version 01.02, use the
  [God of War III Remastered XML](https://github.com/cuesta4/shadPS4/releases/download/tutz-gow-2026-08-23/God_of_War_III_Remastered.xml)
  instead of the HDD setting. In **Cheats / Patches → Patches**, choose the
  **shadPS4** repository and run **Download Patches** once. Then right-click the
  game, choose **Open Folder... → Open Patches Folder**, open `shadPS4`, and
  replace `God_of_War_III_Remastered.xml` with the downloaded XML. Reopen
  **Cheats / Patches**, enable **Bug Fix - Texture Corruption Fix**, and save.
  Do not combine this fix with a resolution patch. Downloading patches again
  may replace the XML, so restore this file afterward if needed.

## What these builds change

### Shared by TUTZ-EMU and TUTZ-GOW

- Graphics shaders and pipelines can compile in the background. Shader-cache
  persistence and background I/O reduce repeated compilation and cache stalls.
- Buffer and texture caches track aliased resources coherently, including
  GPU-modified images and linear readbacks.
- Guest VBlank timing is separated from Vulkan presentation timing, and
  inactive depth/stencil attachments are omitted from graphics pipelines.
- PM4 processing, draw setup, rasterization, buffer-cache lookups and other
  frequent CPU/GPU paths avoid redundant work.
- Per-game `app0`/HDD read bandwidth and time-dilation controls provide
  options for games sensitive to storage timing.
- Shader interface and clip-plane compatibility fixes are included.

### Additional TUTZ-GOW work

- Eligible GPU events use virtual fences and deferred completion labels or
  writebacks, reducing waits on the CPU command path.
- GPU-owned guest memory can be served from its GPU shadow during uploads.
  Authority tracking and linear-readback paths avoid downloading unrelated
  image data or waiting for a whole resource when only part is needed.
- Guest-to-staging copies run on workers when safe; protected memory remains
  ordered with the command processor. Vulkan commands are recorded on a
  dedicated thread.
- Reuse of shader specialization, descriptor state and texture lookups reduces
  repeated work across draws. Render-target updates and command submission
  paths also avoid unnecessary operations.

## Limits and feedback

TUTZ-GOW has been tuned and validated primarily with God of War III; behavior
in other games is uncertain. Shader compilation may still stutter on a fresh
cache, and neither build guarantees a particular frame rate or visual result.
If you report a regression, include the game serial and version, selected
build, GPU and driver, relevant settings, and steps to reproduce it.

## Showcase

[![Watch the God of War III demonstration](https://img.youtube.com/vi/tWWtB59fE7o/maxresdefault.jpg)](https://www.youtube.com/watch?v=tWWtB59fE7o)

## Source and license

The emulator sources are available on the [TUTZ-EMU](https://github.com/cuesta4/shadPS4/tree/tutz-emu)
and [TUTZ-GOW](https://github.com/cuesta4/shadPS4/tree/codex/eltutz-gow) branches.
[TUTZ-UI source](https://github.com/cuesta4/shadps4-qtlauncher) is maintained
separately. Thanks to the upstream shadPS4 contributors. This fork is licensed
under [GPL-2.0-or-later](LICENSE).
