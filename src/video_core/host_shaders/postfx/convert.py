# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def run(args):
    subprocess.run([str(arg) for arg in args], check=True, capture_output=True, text=True)


def convert(tools, name, source, entry, defines, credit, licence):
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        spv = Path(temporary) / "shader.spv"
        glsl = Path(temporary) / "shader.comp"
        run([tools / "dxc.exe", "-spirv", "-T", "cs_6_0", "-E", entry, "-O3",
             "-fspv-target-env=vulkan1.1", "-Fo", spv, root / source,
             *["-D" + define for define in defines]])
        run([tools / "spirv-cross.exe", spv, "--vulkan-semantics", "--version", "450",
             "--output", glsl])
        text = glsl.read_text().replace("rgba32f)", "rgba16f)")
        if name.startswith("cmaa"):
            text = re.sub(r"(binding = 1, )r32ui", r"\1CMAA_EDGE_FORMAT", text)
            helpers = ""
            if "texelFetch(" in text:
                helpers += """vec4 SafeFetch(texture2D t, ivec2 p, int lod) {
    ivec2 size = textureSize(t, lod);
    return any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))
        ? vec4(0) : texelFetch(t, p, lod);
}
#define texelFetch SafeFetch
"""
            if "imageLoad(" in text:
                helpers += """#define imageLoad(t,p) (any(lessThan((p),ivec2(0))) || any(greaterThanEqual((p),imageSize(t))) ? uvec4(0) : imageLoad(t,p))
"""
            text = text.replace("void main()", helpers + "\nvoid main()")
        formats = {"psmaa0": {10: "PSMAA_LUMA_FORMAT", 11: "PSMAA_LUMA_FORMAT",
                              12: "PSMAA_VECTOR_FORMAT"},
                   "psmaa2": {10: "PSMAA_VECTOR_FORMAT"},
                   "psmaa3": {10: "PSMAA_VECTOR_FORMAT"}, "psmaa4": {10: "rgba8"}}
        for binding, fmt in formats.get(name, {}).items():
            text = text.replace(f"binding = {binding}, rgba16f)", f"binding = {binding}, {fmt})")
        defaults = """#ifndef PSMAA_LUMA_FORMAT
#define PSMAA_LUMA_FORMAT r16f
#endif
#ifndef PSMAA_VECTOR_FORMAT
#define PSMAA_VECTOR_FORMAT rg16f
#endif
#ifndef CMAA_EDGE_FORMAT
#define CMAA_EDGE_FORMAT r8ui
#endif
"""
        text = text.replace("layout(local_size_x", defaults + "\nlayout(local_size_x", 1)
        text = f"// SPDX-FileCopyrightText: {credit}\n// SPDX-License-Identifier: {licence}\n\n" + text
        text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"
        glsl.write_text(text, encoding="utf-8", newline="\n")
        for options in [[], ["-DPSMAA_LUMA_FORMAT=rgba16f", "-DPSMAA_VECTOR_FORMAT=rgba16f",
                             "-DCMAA_EDGE_FORMAT=r32ui"]]:
            run([tools / "glslangValidator.exe", "-V", "--target-env", "vulkan1.1",
                 *options, "-o", spv, glsl])
            run([tools / "spirv-val.exe", "--target-env", "vulkan1.1", spv])
        (root.parent / (name + ".comp")).write_text(text, encoding="utf-8", newline="\n")
        print(name + " validated", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Regenerate the GLSL post-processing shaders.")
    parser.add_argument("--tools", type=Path, required=True,
                        help="Vulkan SDK Bin directory containing DXC, SPIRV-Cross and SPIRV-Tools")
    args = parser.parse_args()
    for stage in range(7):
        convert(args.tools, f"psmaa{stage}", "psmaa.hlsl", "main", [f"STAGE={stage}"],
                "Copyright 2025 RdenBlaauwen; Jorge Jimenez et al.; NVIDIA Corporation; Derek Brush",
                "LicenseRef-PSMAA-ThirdParty")
    for stage, entry in enumerate(["EdgesColor2x2CS", "ComputeDispatchArgsCS",
                                   "ProcessCandidatesCS", "DeferredColorApply2x2CS"]):
        convert(args.tools, f"cmaa2_{stage}", "cmaa2.hlsl", entry,
                ["CMAA2_UAV_STORE_TYPED=1", "CMAA2_UAV_STORE_TYPED_UNORM_FLOAT=0",
                 "CMAA2_UAV_STORE_CONVERT_TO_SRGB=0"], "Copyright Intel Corporation", "Apache-2.0")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stdout + error.stderr)
