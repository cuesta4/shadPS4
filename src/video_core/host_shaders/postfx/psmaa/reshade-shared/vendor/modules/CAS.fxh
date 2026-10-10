#ifndef _CAS_FHX // include guard
#define _CAS_FHX
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//
//                                 [CAS] FIDELITY FX - CONSTRAST ADAPTIVE SHARPENING 1.20190610
//
//==============================================================================================================================
// LICENSE
// =======
// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
// -------
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// -------
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
// -------
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//==============================================================================================================================
//
// Modified by Robert den Blaauwen (aka RdenBlaauwen) on 2025 to port from HLSL to the ReShade graphics language.

// MACROS
// The following preprocessor variables should be defined in the main file.
// The values are defaults and can be changed as needed:
// #define CAS_BETTER_DIAGONALS 1

#include "../../libraries/functions.fxh"

namespace CAS {
  float3 CasLoad(sampler colorLinearSampler, float2 texcoord)
  {
    return tex2D(colorLinearSampler, texcoord).rgb;
  }

  float3 CasLoad(sampler colorLinearSampler, float2 texcoord, float2 offset)
  {
    return tex2Doffset(colorLinearSampler, texcoord, offset).rgb;
  }

  // Function for precalculating any constants needed for the filter.
  // Currently only calculates the sharpness value.
  // Call this function before calling the wrapper or CasCalculations().
  void CasSetup(
      // out uint const1,
      out float const1,
      float sharpness // 0 := default (lower ringing), 1 := maximum (higest ringing)
  )
  {
    // Sharpness value
    const1 = -rcp(8.0f - 3.0f * sharpness);
  }

  // The calculations of the CAS function, separated from the tex sampling.
  // If you already have a 9 tap pattern of samples in the context you're using this in,
  // you can just call `CasCalculations()` and pass the color values to skip having to sample again.
  float3 CasCalculations(
    float3 a, float3 b, float3 c,
    float3 d, float3 e, float3 f,
    float3 g, float3 h, float3 i,
    float const1
  )
  {

    // Soft min and max.
    //  a b c             b
    //  d e f * 0.5  +  d e f * 0.5
    //  g h i             h
    // These are 2.0x bigger (factored out the extra multiply)
    float3 mn = Functions::min(d, e, f, b, h);

    #if CAS_BETTER_DIAGONALS
      float3 mn2 = Functions::min(mn, a, c, g, i);
      mn = mn + mn2;
    #endif

    float3 mx = Functions::max(d, e, f, b, h);

    #if CAS_BETTER_DIAGONALS
      float3 mx2 = Functions::max(mx, a, c, g, i);
      mx = mx + mx2;
    #endif

    // Smooth minimum distance to signal limit divided by smooth max
    float3 rcpM = rcp(mx);

    #if CAS_BETTER_DIAGONALS
      float3 amp = saturate(min(mn, 2.0f - mx) * rcpM);
    #else
      float3 amp = saturate(min(mn, 1.0f - mx) * rcpM);
    #endif

    // Shaping amount of sharpening
    amp = sqrt(amp);

    // Filter shape.
    //  0 w 0
    //  w 1 w
    //  0 w 0
    float peak = const1;
    float3 w = amp * peak;
    // Filter
    float3 rcpWeight = rcp(1.0f + 4.0f * w);
    return saturate(((b + d + f + h) * w + e) * rcpWeight);
  }

  // This is the function you usually want to call.
  void CasWrapper(
      float2 texcoord,
      float const1,
      sampler colorLinearSampler,
      out float3 original,
      out float3 processed
  )
  {
    // a b c
    // d e f
    // g h i
    float3 a = CasLoad(colorLinearSampler, texcoord, float2(-1, -1));
    float3 b = CasLoad(colorLinearSampler, texcoord, float2(0, -1));
    float3 c = CasLoad(colorLinearSampler, texcoord, float2(1, -1));
    float3 d = CasLoad(colorLinearSampler, texcoord, float2(-1, 0));
    original = CasLoad(colorLinearSampler, texcoord);
    float3 f = CasLoad(colorLinearSampler, texcoord, float2(1, 0));
    float3 g = CasLoad(colorLinearSampler, texcoord, float2(-1, 1));
    float3 h = CasLoad(colorLinearSampler, texcoord, float2(0, 1));
    float3 i = CasLoad(colorLinearSampler, texcoord, float2(1, 1));

    processed = CasCalculations(a, b, c, d, original, f, g, h, i, const1);
  }

  void CasFilter(
      float2 texcoord,
      float const1,
      sampler colorLinearSampler,
      out float3 pix
  )
  {
    float3 original; // goes unused, just to satisfy the wrapper params
    CasWrapper(texcoord, const1, colorLinearSampler, original, pix);
  }

  // custom overload for blending strength
  void CasFilter(
      float2 texcoord,
      float const1,
      float blendingStrength,
      sampler colorLinearSampler,
      out float3 pix
  )
  {
    float3 original;
    float3 result;
    CasWrapper(texcoord, const1, colorLinearSampler, original, result);
    pix = lerp(original, result, blendingStrength);
  }
}
#endif // CAS_FHX include guard
