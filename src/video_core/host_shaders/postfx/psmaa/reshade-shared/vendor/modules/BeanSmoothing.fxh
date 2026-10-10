#ifndef _BEAN_SMOOTHING_FHX  // include guard
#define _BEAN_SMOOTHING_FHX
/////////////////////////////////// CREDITS & LICENSES ///////////////////////////////////
/**
 * This shader contains components taken and/or adapted from Lordbean's TSMAA.
 * https://github.com/lordbean-git/reshade-shaders/blob/main/Shaders/TSMAA.fx
 *
 * The 'smooth()' function in this module is a modified version of FXAA 3.11,
 * Copyright (c) 2010, 2011 NVIDIA Corporation.
 *
 * Code originating from SMAA is Copyright (c) 2013 Jorge Jimenez et al.
 *
 * TSMAA code is Copyright (c) "Lordbean" Derek Brush (derekbrush@gmail.com).
 *
 * Modifications and all other original code are Copyright (c) 2025 RdenBlaauwen.
 *
 * This file is subject to the licenses and disclaimers of its respective components,
 * the full texts of which are included below.
 */
/*============================================================================


                    NVIDIA FXAA 3.11 by TIMOTHY LOTTES


------------------------------------------------------------------------------
COPYRIGHT (C) 2010, 2011 NVIDIA CORPORATION. ALL RIGHTS RESERVED.
------------------------------------------------------------------------------
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL NVIDIA
OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION,
OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE
THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
DAMAGES.*/
/*               TSMAA for ReShade 3.1.1+
 *
 *    (Temporal Subpixel Morphological Anti-Aliasing)
 *
 *
 *     Experimental multi-frame SMAA implementation
 *
 *                     by lordbean
 *
 */
/*------------------------------------------------------------------------------
 * THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *-------------------------------------------------------------------------------*/
/*
 * Modified by Robert den Blaauwen (aka RdenBlaauwen).
*/
/**
 *                  _______  ___  ___       ___           ___
 *                 /       ||   \/   |     /   \         /   \
 *                |   (---- |  \  /  |    /  ^  \       /  ^  \
 *                 \   \    |  |\/|  |   /  /_\  \     /  /_\  \
 *              ----)   |   |  |  |  |  /  _____  \   /  _____  \
 *             |_______/    |__|  |__| /__/     \__\ /__/     \__\
 *
 *                               E N H A N C E D
 *       S U B P I X E L   M O R P H O L O G I C A L   A N T I A L I A S I N G
 *
 *                         http://www.iryoku.com/smaa/
 *
 * Copyright (C) 2013 Jorge Jimenez (jorge@iryoku.com)
 * Copyright (C) 2013 Jose I. Echevarria (joseignacioechevarria@gmail.com)
 * Copyright (C) 2013 Belen Masia (bmasia@unizar.es)
 * Copyright (C) 2013 Fernando Navarro (fernandn@microsoft.com)
 * Copyright (C) 2013 Diego Gutierrez (diegog@unizar.es)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to
 * do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software. As clarification, there
 * is no requirement that the copyright notice and permission be included in
 * binary distributions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// MACROS
// The following preprocessor variables should be defined in the main file.
// The values are defaults and can be changed as needed:
// #define SMOOTHING_SATURATION_DIVISOR_FLOOR 0.01
// #define SMOOTHING_BUFFER_RCP_HEIGHT BUFFER_RCP_HEIGHT
// #define SMOOTHING_BUFFER_RCP_WIDTH BUFFER_RCP_WIDTH
// #define SMOOTHING_DEBUG false
// #define SMOOTHING_THRESHOLD_DEPTH_GROWTH_START .5
// #define SMOOTHING_THRESHOLD_DEPTH_GROWTH_FACTOR 2.5
// #define SMOOTHING_MIN_ITERATIONS 3.0f
// #define SMOOTHING_MAX_ITERATIONS 15.0f

// Shorthands for sampling
// #define SmoothingSampleLevelZero(tex, coord) tex2Dlod(tex, float4(coord, 0.0, 0.0))
// #define SmoothingSampleLevelZeroOffset(tex, coord, offset) tex2Dlodoffset(tex, float4(coord, coord), offset)
// #define SmoothingGatherLeftDeltas(tex, coord) tex2Dgather(tex, texcoord, 0);
// #define SmoothingGatherTopDeltas(tex, coord) tex2Dgather(tex, texcoord, 1);

#include "../../libraries/color.fxh"
#include "../../libraries/functions.fxh"

namespace BeanSmoothing
{
  namespace SMAA
  {
    // All code in the SMAA namespace originates from the original SMAA shader,
    // and is subject to the SMAA license included in the beginning of this file.
    /**
     * Conditional move:
     */
    void Movc(bool2 cond, inout float2 variable, float2 value)
    {
      [flatten] if (cond.x) variable.x = value.x;
      [flatten] if (cond.y) variable.y = value.y;
    }

    void Movc(bool4 cond, inout float4 variable, float4 value)
    {
      Movc(cond.xy, variable.xy, value.xy);
      Movc(cond.zw, variable.zw, value.zw);
    }
  }

  float dotweight(float3 middle, float3 neighbor, bool useluma)
  {
    if (useluma)
      return Color::luma(neighbor);
    else
      return Color::luma(abs(middle - neighbor));
  }

  float saturation(float3 rgb)
  {
    float maxComp = max(Functions::max(rgb), SMOOTHING_SATURATION_DIVISOR_FLOOR);
    return Functions::min(rgb) / maxComp;
  }

  float dotsat(float3 rgb, float L)
  {
    return ((Functions::max(rgb) - Functions::min(rgb)) / (1.0 - (2.0 * L - 1.0) + trunc(L)));
  }

  // Function which takes a depth value and produces a curve which is 0.0f until the depth equals a set start value,
  // then grows until it reaches a set maximum value at depth == 1.0f. Finally adds the result to 1.0f to produce a value which
  // can be multiplied with thresholds to grow them based on depth.
  //
  // This code is written by RdenBlaauwen and is fully FOSS, not copyrighted.
  float calcDepthGrowthFactor(float depth)
  {
    float flooredCurve = max(depth - SMOOTHING_THRESHOLD_DEPTH_GROWTH_START, 0.0f);
    float rcpGrowthFactor = 1.0f - SMOOTHING_THRESHOLD_DEPTH_GROWTH_START;
    float curve = saturate(flooredCurve / rcpGrowthFactor);
    return mad(curve, SMOOTHING_THRESHOLD_DEPTH_GROWTH_FACTOR, 1.0f);
  }

  // Calculate the maximum number of iterations based on the mod value that the smoothing algorithm may perform
  uint calcMaxSmoothingIterations(float mod)
  {
    return (uint)(lerp(SMOOTHING_MIN_ITERATIONS, SMOOTHING_MAX_ITERATIONS, mod) + .5);
  }

  // This function provides the smoothest results when working in gamma space, but this also causes the output to darken.
  // Running in linear space is better for more clarity, but the results are less smooth and have some artifacts.
  float3 smooth(float2 texcoord, float4 offset, sampler colorTex, sampler blendSampler, float threshold, uint maxIterations) : SV_Target
  {
    const float3 debugColorNoHits = float3(0.0,0.0,0.0);
    const float3 debugColorSmallHit = float3(0.0,0.0,1.0);
    const float3 debugColorBigHit = float3(1.0,0.0,0.0);

    float3 mid = SmoothingSampleLevelZero(colorTex, texcoord).rgb;
    float3 original = mid;

    float lumaM = Color::luma(mid);
    float chromaM = dotsat(mid, lumaM);
    bool useluma = lumaM > chromaM;
    if (!useluma) lumaM = 0.0;

    float lumaS = dotweight(mid, SmoothingSampleLevelZeroOffset(colorTex, texcoord, int2( 0, 1)).rgb, useluma);
    float lumaE = dotweight(mid, SmoothingSampleLevelZeroOffset(colorTex, texcoord, int2( 1, 0)).rgb, useluma);
    float lumaN = dotweight(mid, SmoothingSampleLevelZeroOffset(colorTex, texcoord, int2( 0,-1)).rgb, useluma);
    float lumaW = dotweight(mid, SmoothingSampleLevelZeroOffset(colorTex, texcoord, int2(-1, 0)).rgb, useluma);

    float rangeMax = Functions::max(lumaS, lumaE, lumaN, lumaW, lumaM);
    float rangeMin = Functions::min(lumaS, lumaE, lumaN, lumaW, lumaM);

    float range = rangeMax - rangeMin;

    // early exit check
    bool earlyExit = (range < threshold);
    if (earlyExit) {
      // If debug, return no hits color to signify no smoothing took place.
      if(SMOOTHING_DEBUG){
        return debugColorNoHits;
      }
      return original;
    }
    // If debug, early return. Return hit colors to signify that smoothing takes place here
    if(SMOOTHING_DEBUG) {
      // The further the range is above the threshold, the bigger the "hit"
      float strength = smoothstep(threshold, 1.0, range);
      return lerp(debugColorSmallHit, debugColorBigHit, strength);
    }

    float lumaNW = dotweight(mid, SmoothingSampleLevelZeroOffset(colorTex, texcoord, int2(-1,-1)).rgb, useluma);
    float lumaSE = dotweight(mid, SmoothingSampleLevelZeroOffset(colorTex, texcoord, int2( 1, 1)).rgb, useluma);
    float lumaNE = dotweight(mid, SmoothingSampleLevelZeroOffset(colorTex, texcoord, int2( 1,-1)).rgb, useluma);
    float lumaSW = dotweight(mid, SmoothingSampleLevelZeroOffset(colorTex, texcoord, int2(-1, 1)).rgb, useluma);

    // These vals serve as caches, so they can be used later without having to redo them
    // It's just an optimisation thing, though the difference it makes is so small it could just be statistical noise.
    float lumaNWSW = lumaNW + lumaSW;
    float lumaNS = lumaN + lumaS;
    float lumaNESE = lumaNE + lumaSE;
    float lumaSWSE = lumaSW + lumaSE;
    float lumaWE = lumaW + lumaE;
    float lumaNWNE = lumaNW + lumaNE;

      bool horzSpan = (abs(mad(-2.0, lumaW, lumaNWSW)) + mad(2.0, abs(mad(-2.0, lumaM, lumaNS)), abs(mad(-2.0, lumaE, lumaNESE)))) >= (abs(mad(-2.0, lumaS, lumaSWSE)) + mad(2.0, abs(mad(-2.0, lumaM, lumaWE)), abs(mad(-2.0, lumaN, lumaNWNE))));
      float lengthSign = horzSpan ? SMOOTHING_BUFFER_RCP_HEIGHT : SMOOTHING_BUFFER_RCP_WIDTH;

    float4 midWeights = float4(
      SmoothingSampleLevelZero(blendSampler, offset.xy).a,
      SmoothingSampleLevelZero(blendSampler, offset.zw).g,
      SmoothingSampleLevelZero(blendSampler, texcoord).zx
    );

    bool smaahoriz = max(midWeights.x, midWeights.z) > max(midWeights.y, midWeights.w);
    bool smaadata = any(midWeights);
    float maxWeight = Functions::max(midWeights.r, midWeights.g, midWeights.b, midWeights.a);
    float maxblending = 0.5 + (0.5 * maxWeight);

    if ((horzSpan && smaahoriz && smaadata) || (!horzSpan && !smaahoriz && smaadata)) {
      maxblending *= 1.0 - maxWeight / 2.0;
    } else {
      maxblending = min(maxblending * 1.5, 1.0);
    };

    float2 lumaNP = float2(lumaN, lumaS);
    SMAA::Movc(bool(!horzSpan).xx, lumaNP, float2(lumaW, lumaE));

    float gradientN = lumaNP.x - lumaM;
    float gradientS = lumaNP.y - lumaM;
    float lumaNN = lumaNP.x + lumaM;

    if (abs(gradientN) >= abs(gradientS)) lengthSign = -lengthSign;
    else lumaNN = lumaNP.y + lumaM;

    float2 posB = texcoord;

    const float texelsize = 0.5; // TODO: Macro?
    const float lengthSignDivided = lengthSign / 2.0f;

    float2 offNP = float2(0.0, SMOOTHING_BUFFER_RCP_HEIGHT * texelsize);
    SMAA::Movc(bool(horzSpan).xx, offNP, float2(SMOOTHING_BUFFER_RCP_WIDTH * texelsize, 0.0));
    SMAA::Movc(bool2(!horzSpan, horzSpan), posB, float2(posB.x + lengthSignDivided, posB.y + lengthSignDivided));

    float2 posN = posB - offNP;
    float2 posP = posB + offNP;

    float lumaEndN = dotweight(mid, SmoothingSampleLevelZero(colorTex, posN).rgb, useluma);
    float lumaEndP = dotweight(mid, SmoothingSampleLevelZero(colorTex, posP).rgb, useluma);

    float gradientScaled = max(abs(gradientN), abs(gradientS)) * .25;
    bool lumaMLTZero = mad(.5, -lumaNN, lumaM) < 0.0f;

    lumaNN *= .5;

    lumaEndN -= lumaNN;
    lumaEndP -= lumaNN;

    bool doneN = abs(lumaEndN) >= gradientScaled;
    bool doneP = abs(lumaEndP) >= gradientScaled;
    bool doneNP = doneN && doneP;

    if(!doneNP){
      uint iterations = 0;
      [loop] while (iterations < maxIterations)
      {
        doneNP = doneN && doneP;
        if (doneNP) break;
        if (!doneN)
        {
          posN -= offNP;
          lumaEndN = dotweight(mid, SmoothingSampleLevelZero(colorTex, posN).rgb, useluma);
          lumaEndN -= lumaNN;
          doneN = abs(lumaEndN) >= gradientScaled;
        }
        if (!doneP)
        {
          posP += offNP;
          lumaEndP = dotweight(mid, SmoothingSampleLevelZero(colorTex, posP).rgb, useluma);
          lumaEndP -= lumaNN;
          doneP = abs(lumaEndP) >= gradientScaled;
        }
        iterations++;
      }
    }

    float2 dstNP = float2(texcoord.y - posN.y, posP.y - texcoord.y);
    SMAA::Movc(bool(horzSpan).xx, dstNP, float2(texcoord.x - posN.x, posP.x - texcoord.x));

    bool goodSpan = (dstNP.x < dstNP.y) ? ((lumaEndN < 0.0) != lumaMLTZero) : ((lumaEndP < 0.0) != lumaMLTZero);
    float pixelOffset = mad(-rcp(dstNP.y + dstNP.x), min(dstNP.x, dstNP.y), 0.5);
    float subpixOut = pixelOffset * maxblending;

    [branch] if (!goodSpan)
    {
      subpixOut = mad(mad(2.0, lumaNS + lumaWE, lumaNWSW + lumaNESE), 0.083333, -lumaM) * rcp(range); //ABC
      subpixOut = pow(saturate(mad(-2.0, subpixOut, 3.0) * (subpixOut * subpixOut)), 2.0) * maxblending * pixelOffset; // DEFGH
    }

    float2 posM = texcoord;
    SMAA::Movc(bool2(!horzSpan, horzSpan), posM, mad(lengthSign, subpixOut, posM));

    return SmoothingSampleLevelZero(colorTex, posM).rgb;
  }
}
#endif // BEAN_SMOOTHING_FHX
