<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# shadPS4-eltutz

**shadPS4-eltutz** is an unofficial Windows build of the
[shadPS4](https://github.com/shadps4-emu/shadPS4) PlayStation 4 emulator. It aims to make
games run faster, feel more responsive and look smoother on your screen. Its main test
games are **God of War III Remastered** and **Bloodborne**.

There is now a single build for every game. The older TUTZ-EMU and TUTZ-GOW builds are
retired: everything they did lives in shadPS4-eltutz.

Massive thanks to the shadPS4 developers. None of this would exist without their work.

> [!CAUTION]
> ## Unofficial build: please read
>
> This is **not an official shadPS4 release** and is not endorsed by the upstream
> developers.
>
> - Do **not** contact the upstream developers about this build.
> - Do **not** report problems with this build in the official shadPS4 repository.
> - Do **not** assume that a problem you see here also exists in official shadPS4.
>   Try an official build first before opening an upstream report.
> - Feedback about this build belongs in this fork. Include the game and the build
>   you used.

- [What this build does better than official shadPS4](#what-this-build-does-better-than-official-shadps4)
- [What you need](#what-you-need)
- [Downloads](#downloads)
- [Installation, step by step](#installation-step-by-step)
- [Recommended settings](#recommended-settings)
- [Troubleshooting](#troubleshooting)
- [Building it yourself](#building-it-yourself)

## What this build does better than official shadPS4

On official shadPS4, God of War III was hard to play:

- It stuttered all the time.
- It froze for seconds whenever it had to compile shaders.
- Many scenes ran below 50 FPS.

With this build, it often stays locked at 120 FPS (with the 120 FPS patch). Bloodborne
and other games benefit from most of the same work.

### No more freezes while shaders compile

A shader is a small program that draws one kind of effect on your graphics card. It must
be compiled the first time the effect appears.

- **Compilation runs in the background.** Official builds stop the game until the
  shader is ready. In God of War III, that meant freezes of several seconds. With
  **Async Shader Recompiling**, six worker threads compile in the background and the
  game keeps running.
- **Shorter waits for new shaders.** The game still waits while the emulator reads a new
  shader and its data from the game's memory. Two high-priority threads now do only that
  step and pass the rest to the other workers, so the game waits about 1 ms per new
  shader instead of queuing behind other compile work.
- **Compiled shaders stay compiled.**
  - Shaders compiled once are kept between sessions, so they are not compiled again.
  - The cache is written in the background, without blocking the game.
  - A damaged cache archive is repaired instead of being lost.
  - Loading the cache no longer hangs the game.
- **One cache to delete.** Deleting a game's shader cache now also clears its Vulkan
  pipeline cache.

### Much higher frame rates

Every frame, the emulator turns thousands of PS4 graphics commands into commands your
graphics card understands. This build does that job with much less overhead.

- **Less work for every object drawn.** Command processing, draw setup, and buffer and
  texture lookups were rewritten to skip repeated work and execute far fewer
  instructions. The latest round alone took our God of War III test scene from about
  100 FPS to a locked 120 FPS.
- **The work is shared across your CPU cores.** Helper threads copy game data for the
  graphics card and record graphics commands. That work no longer piles up on one busy
  thread.
- **Less waiting between the CPU and the graphics card.** Games constantly wait for
  results from the GPU: completion signals, images read back to memory, memory shared
  by several images. This build handles those without stopping the emulator whenever it
  can. That also removed the frame drops when God of War III hits certain enemies.
- **Fewer interruptions from memory tracking.** Bloodborne streams data through a large
  block of memory, and the emulator stopped at almost every 4 KB of it to track what
  changed. It now spots data written in sequence and handles it in bulk. That change
  took our Bloodborne benchmark scene from about 67 to about 80 FPS.
- **Bloodborne frees a CPU core when playing offline.** Its network thread no longer
  spins a whole core when there is no network to talk to.

### A correct picture

- **God of War III exposure.** The game measures how bright the scene is and adjusts the
  exposure, like a camera. It writes that measurement into a tiny image and reads it
  back through another image that shares the same memory. Official builds did not pass
  the data between the two, so the picture was blown out. This build keeps images that
  share memory in sync, and the exposure adapts as it should. The same fix brings back
  the hidden effects of the chests.
- **God of War III texture corruption.** The included patch file gives the game more
  video memory, which fixes the corrupted textures.
- **Other effects that read other images.** Post-processing effects that read another
  image's memory get the right data too, instead of missing or stale content.
- **No more cropped passes.** On the PS4, the color and depth buffers of a pass are
  independent. On the PC, an unused, smaller depth buffer could shrink the drawn area of
  a full-screen pass and leave old pixels around it. Depth buffers that have no effect
  are now left out.
- **Tessellation shaders.** Data passed between the tessellation stages of a shader,
  which add geometric detail, is now declared correctly.
- **Blending that keeps the smaller or the larger value.**
  - To draw some effects, the graphics card compares each new pixel with the one already
    in the image and keeps the smaller or the larger of the two. The PS4 scales both
    values first; PC graphics cards skip that step.
  - God of War III uses this for a shadow pass and for some transparent effects. Its
    main image keeps extra brightness in a hidden channel, so without the scaling those
    scenes came out brighter, with fainter shadows, than on the PS4.
  - This build computes the same result as the PS4. When one of the values is scaled to
    zero, a simpler blend gives it exactly. When both values are scaled by themselves,
    the effect is drawn a second time to finish the calculation.

### Motion looks smoother

Frame pacing is about *when* each frame reaches your screen. Even a high frame rate
looks choppy when frames arrive unevenly.

- **The game's clock runs on its own.**
  - A PS4 game paces itself on the console's display refresh, the "vblank".
  - On official builds, a slow moment when showing a frame, such as switching to
    fullscreen or a slow launch, could leave a game stuttering at a fraction of its
    speed until you restarted it.
  - In this build, that clock no longer depends on when Windows shows a frame.
- **Frames are timed to your monitor.**
  - The emulator measures when your monitor shows each frame and learns its exact
    refresh rate. It then releases each new frame just in time for the next refresh.
  - A frame that cannot be shown sooner waits for the next refresh. It is never thrown
    away, so your graphics card never renders frames that no one sees.

### Controls feel more responsive

Input lag is the time between pressing a button and seeing the result. It grows when
finished frames wait in line to be shown.

- **Short queues.** The frame timing described above keeps that line short: frames reach
  the monitor about as soon as it can show them.
- **GPU Frames Ahead.** This setting (0, 1 or 2) limits how far the emulated console may
  run ahead of your graphics card. Lower values cut lag when the graphics card is the
  bottleneck.
- **NVIDIA Reflex.** On NVIDIA graphics cards, Reflex holds the emulated console back
  until your graphics card is about to need the next frame, so frames do not wait in a
  queue.

### Sound keeps up after a hitch

A PS4 plays a small block of sound every few milliseconds, even when the game is busy.

- When the emulator stalls, the PS4 would still have played the blocks due in the
  meantime. This build catches up on at most four of them.
- It waits for the game to deliver each of those blocks instead of letting it pass as
  silence.
- Sound stays in step with the game, and a long stall cannot pile up extra delay.

### Extra options

- **HDD Read Speed** and **Disable Time Dilation** let you slow down how fast a game
  reads its data, per game. Some games expect a real PS4 hard drive and can glitch when
  data arrives too fast. See [Recommended settings](#recommended-settings).

### The God of War III fast path

God of War III reads results back from the graphics card every frame; that is how its
exposure works, for example. For this game (CUSA01715), and only with **Enable Readback
Linear Images** on, this build uses a special path:

- The GPU signals its own progress.
- The results stay on the graphics card until the game's own code actually needs them in
  memory.

Every other game uses the regular path. That is why one build can now serve all games.

## What you need

- **Windows 10 or 11, 64-bit.**
- **A CPU with AVX2**: Intel Core 4th generation (Haswell) or newer, or any AMD Ryzen.
  The release is compiled for the processor of the machine that built it, an AMD Ryzen
  5000 (Zen 3). It should run on other AVX2 processors. If the emulator closes the moment
  a game starts, [build it yourself](#building-it-yourself).
- **A graphics card with up-to-date Vulkan drivers.** Reflex needs an NVIDIA card.
- **Your own PS4 games**, dumped from your own console. No games are provided here.
- Optionally, **PS4 firmware modules**, also dumped from your own console. Some games
  need them. See [firmware modules](#firmware-modules).

## Downloads

| File | Where to get it | What it is |
|---|---|---|
| `shadPS4QtLauncher-win64-qt-<date>.zip` | [Official QtLauncher releases](https://github.com/shadps4-emu/shadPS4-qtlauncher/releases) | The official launcher package, with the Qt files the launcher needs |
| `shadPS4QtLauncher.exe` | [Latest shadPS4-eltutz release](https://github.com/cuesta4/shadPS4/releases/latest) | The eltutz launcher. It replaces the official launcher executable and adds this build's settings |
| `shadps4-eltutz.exe` | [Latest shadPS4-eltutz release](https://github.com/cuesta4/shadPS4/releases/latest) | The emulator |
| `God_of_War_III_Remastered.xml` | [Latest shadPS4-eltutz release](https://github.com/cuesta4/shadPS4/releases/latest) | God of War III patches: texture fix, 120 FPS, skip intro, skip videos |

## Installation, step by step

The steps use `C:\Games\shadPS4` as the install folder. Any folder you own works.

### 1. Install the launcher

> [!TIP]
> **Already use the QtLauncher?** You can keep your current install:
>
> 1. Close the launcher.
> 2. Replace its `shadPS4QtLauncher.exe` with the one from the
>    [latest shadPS4-eltutz release](https://github.com/cuesta4/shadPS4/releases/latest).
> 3. Put `shadps4-eltutz.exe` in your `versions` folder.
> 4. Add the emulator in the launcher, as in [step 2](#2-add-the-emulator).
>
> Your games, settings and saves stay as they are, and this build's new settings are
> added automatically. After step 2, skip to
> [step 4](#4-install-the-god-of-war-iii-patches) if you play God of War III.

For a fresh install:

1. Open the [official QtLauncher releases](https://github.com/shadps4-emu/shadPS4-qtlauncher/releases)
   and download the newest `shadPS4QtLauncher-win64-qt-<date>.zip`. All of those
   releases are marked "Pre-release"; that is expected.
2. Extract the zip into `C:\Games\shadPS4`. Avoid `Program Files`: Windows blocks
   programs from writing there.
3. **Recommended:** inside `C:\Games\shadPS4`, create an empty folder named `user`.
   The launcher and the emulator then keep your settings, saves, shader caches and
   patches in `C:\Games\shadPS4\user`, not in your Windows profile. The whole install
   becomes one folder that is easy to back up or move.
4. Download `shadPS4QtLauncher.exe` from the
   [latest shadPS4-eltutz release](https://github.com/cuesta4/shadPS4/releases/latest)
   and replace the file with the same name in `C:\Games\shadPS4`. Keep all the other
   files from the zip: the launcher needs its Qt DLLs and folders.

### 2. Add the emulator

1. Create the folder `C:\Games\shadPS4\versions\shadps4-eltutz` and put
   `shadps4-eltutz.exe` in it.
2. Start `shadPS4QtLauncher.exe` and click **Version Manager**, at the top right.
3. Click **Add Custom**, select `shadps4-eltutz.exe` and name the version
   `shadps4-eltutz`. The launcher selects it right away: its box in the **Selected**
   column is ticked. You can switch versions later from the same list.

### 3. Add your games

1. Open **Settings**, go to the **Paths** tab and, under **Game Folders**, click
   **Add...**. Pick the folder that holds your games. Each game sits in its own folder,
   for example `CUSA01715`.
2. Game updates go in a folder next to the game, named after it with `-UPDATE` at the
   end, for example `CUSA01715-UPDATE`. The God of War III patches need update 01.02.

### 4. Install the God of War III patches

1. In the launcher menu, click **Utils → Download Cheats/Patches**. This downloads the
   community patches for many games, such as Bloodborne.
2. Right-click God of War III and choose **Open Folder... → Open Patches Folder**. Open
   the `shadPS4` folder inside it and replace `God_of_War_III_Remastered.xml` with the
   one from the eltutz release.
   > [!NOTE]
   > Downloading the patches again overwrites this file. Copy the eltutz file back
   > after every download.
3. Right-click God of War III, choose **Cheats / Patches**, open the **Patches** tab and
   make sure these are enabled, then save:
   - **Bug Fix - Texture Corruption Fix**
   - **Frame Rate Patch - 120 FPS**. It needs **Vblank Frequency** set to 120; see the
     next step.
   - **Skip Intro** and **Skip Any Video With X Button**, if you want them.

   Enable either the texture fix or one resolution patch, never both. The resolution
   patches already include more video memory.

   > [!IMPORTANT]
   > The texture fix gives the game more video memory, so the emulator must reserve
   > more memory too: set **Additional DMem Allocation** to 2048 or more (next step).
   > Without it, the game crashes. Each resolution patch states its own value in its
   > description.

### 5. Set up God of War III

Right-click the game and choose **Game-specific Settings... → Configure Game-specific
Settings**. Settings made there apply to this game only.

| Tab | Setting | Value | Why |
|---|---|---|---|
| Graphics | Present Mode | Mailbox | Lowest lag at 120 FPS |
| Graphics | Enable NVIDIA Reflex | On (NVIDIA only) | Shorter frame queue |
| Experimental | Additional DMem Allocation | 2048 | Required by the texture fix; without it the game crashes |
| Experimental | Vblank Frequency | 120 | Required by the 120 FPS patch |
| Experimental | Readbacks Mode | Disabled | The fast path replaces it |
| Experimental | Enable Readback Linear Images | On | Turns on the God of War III fast path |
| Experimental | Async Shader Recompiling | On | Less shader stutter |
| Experimental | GPU Frames Ahead | 2 | The default |

With the texture fix patch on, leave **HDD Read Speed** at its default.

### 6. Play

Double-click the game. Useful keys while playing:

| Key | Action |
|---|---|
| F10 | FPS counter |
| Ctrl+F10 | Video debug info |
| F11 | Fullscreen |

Xbox and DualShock controllers work out of the box. Keyboard and mouse controls can be
changed from the **Controllers** and **Keyboard** buttons in the launcher toolbar.

### Firmware modules

Some games need PS4 system modules, such as fonts and audio decoders. Dump them from
your own console and copy the `.sprx` files into the `sys_modules` folder of your user
folder (`C:\Games\shadPS4\user\sys_modules` if you created the `user` folder in step 1).
The official README has the
[list of supported modules](https://github.com/shadps4-emu/shadPS4#firmware-files).

## Recommended settings

These tips apply to every game. Set them per game, as in step 5, so each game keeps its
own.

- **Present Mode**
  - **Fifo** (V-Sync) shows every frame, in order: the smoothest option. This build
    keeps its lag low.
  - **Mailbox** replaces a waiting frame with a newer one. It can cut a little more lag,
    but pacing may be less even.
  - With a G-Sync or FreeSync monitor, Fifo lets the monitor follow the game's frame
    rate.
- **Vblank Frequency.** Keep 60 unless a patch asks for more, like the God of War III
  120 FPS patch.
- **NVIDIA Reflex.** Try it on. If motion looks less even, turn it off for that game.
  In our tests, Bloodborne at 120 Hz looked smoother without it.
- **GPU Frames Ahead.** 2 is the default. Try 1 for less lag when your graphics card is
  the bottleneck. 0 removes the limit and is not recommended.
- **Async Shader Recompiling.** Keep it on to avoid shader stutter. If a game shows
  visual glitches or crashes, turn it off for that game.
- **HDD Read Speed** and **Disable Time Dilation**, in the Experimental tab.
  - They are only an option for other games that glitch when their data arrives faster
    than from a real PS4 hard drive.
  - A lower read speed means longer loads.
  - Disable Time Dilation keeps the simulated delays tied to real time when the
    emulator slows down, but it may cause issues in some games.
  - For God of War III, leave them alone: the texture fix in the XML patch is the
    preferred fix.

## Troubleshooting

- **"No emulator version was selected" or "Could not find the emulator executable".**
  Open **Version Manager** and select `shadps4-eltutz` again. If you moved the `.exe`,
  add it again with **Add Custom**.
- **The emulator closes as soon as a game starts.**
  - Update your graphics drivers.
  - Check the log: right-click the game, then **Open Folder... → Open Log Folder**.
  - An error about an illegal instruction means your processor lacks an instruction the
    release uses. [Build it yourself](#building-it-yourself) for your own CPU.
- **God of War III crashes during gameplay.**
  - Check that **Additional DMem Allocation** is 2048 or more with the texture fix, or
    the value in the description of your resolution patch.
- **God of War III textures are still corrupted.**
  - Check that **Bug Fix - Texture Corruption Fix** is enabled.
  - Check that your game is CUSA01715 with update 01.02.
- **A short hitch the first time an effect appears.** That is the shader being compiled.
  It is cached, so it does not happen again.

## Building it yourself

Follow the official
[Windows build instructions](https://github.com/shadps4-emu/shadPS4/blob/main/documents/building-windows.md)
with Clang, and check out the `eltutz` branch. The release is built as Release with
ThinLTO, tuned for the CPU of the build machine and without stack buffer checks:

```powershell
cmake --preset x64-Clang-Release -B Build/Release -DX86_64_MARCH=native `
  "-DCMAKE_C_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /GS- -flto=thin" `
  "-DCMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /GS- -flto=thin"
cmake --build Build/Release
```

ThinLTO needs `lld-link` as the linker. Leave out `-DX86_64_MARCH=native` for a build
that runs on any AVX2 processor.

## Showcase

<p align="center">
  <a href="https://www.youtube.com/watch?v=tWWtB59fE7o">
    <img
      src="https://img.youtube.com/vi/tWWtB59fE7o/maxresdefault.jpg"
      width="720"
      alt="Watch demonstration video on YouTube"
    >
  </a>
  <br>
  <a href="https://www.youtube.com/watch?v=tWWtB59fE7o">
    ▶️ <strong>Watch on YouTube</strong>
  </a>
</p>

## Source and license

This fork is based on the open-source [shadPS4 project](https://github.com/shadps4-emu/shadPS4)
and remains available under the [GPL-2.0-or-later license](LICENSE).
