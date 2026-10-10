#ifndef _ANOMALOUS_PIXEL_BLENDING_FHX  // include guard
#define _ANOMALOUS_PIXEL_BLENDING_FHX

// MACROS
// The following preprocessor variables should be defined in the main file.
// The values are defaults and can be changed as needed:
// #define APB_LUMA_PRESERVATION_BIAS .5
// #define APB_LUMA_PRESERVATION_STRENGTH 1.0f
// #define APB_MIN_FILTER_STRENGTH .15

// DEPENDENCIES
#include "../libraries/functions.fxh"
#include "../libraries/color.fxh"

namespace AnomalousPixelBlending
{
  /**
   * Apply some local contrast adaptation to the deltas which are going to be used to determine blending strength.
   * Assumes deltas are in the order RGBA.
   */
  float4 applyLCA(float4 deltas, float lcaFactor)
  {
    float4 maxLocalDeltas = Functions::max(deltas.gbar, deltas.barg, deltas.argb);

    return mad(maxLocalDeltas, -lcaFactor, deltas);
  }

  /**
   * Diminishes the deltas that aren't part of the greatest corner.
   * Can be used to detect if a pixel is part of a corner without significantly smaller yet relevant deltas
   * interfering with any deltas making up the actual corner.
   * Assumes deltas are in the order RGBA.
   */
  float4 applyCornerCorrection(float4 deltas)
  {
    float2 greatestCornerDeltas = max(deltas.rg, deltas.ba);
    float avgGreatestCornerDelta = (greatestCornerDeltas.x + greatestCornerDeltas.y) / 2.0f;
    // taking the square, then dividing by the average greatest corner delta diminishes smaller deltas
    // and preserves the deltas of the largest corner
    return (deltas * deltas) / avgGreatestCornerDelta;
  }

  /**
   * Checks if the deltas correspond to a corner shape. Useful for filtering images before applying
   * an AA technique which needs to detect corners to work properly.
   * Assumes deltas are in the order RGBA.
   *
   * @param deltas: float4 containing the deltas for each edge (R: left, G: top, B: right, A: bottom)
   * @param cornerCorrectionStrength: how much the corner correction is applied to the detlas before checking
   *                                  how many corners there are.
   * @param edgeThreshold: threshold above which a delta is considered an edge.
   * @returns true if the deltas correspond to a corner shape, false otherwise.
   */
  bool checkIfCorner(float4 deltas, float cornerCorrectionStrength, float edgeThreshold)
  {
    float4 correctedDeltas = applyCornerCorrection(deltas);
    correctedDeltas = lerp(deltas, correctedDeltas, cornerCorrectionStrength);

    float4 correctedEdges = step(edgeThreshold, correctedDeltas);
    float cornerNumber = (correctedEdges.r + correctedEdges.b) * (correctedEdges.g + correctedEdges.a);
    return cornerNumber == 1.0f;
  }

  /**
   * Calculates the blending strength based on the deltas provided.
   * Uses a soft threshold to determine the weight of each delta.
   * Assumes deltas are in the order RGBA.
   *
   * @param deltas: float4 containing the deltas for each edge (R: left, G: top, B: right, A: bottom)
   * @param threshold: determines the approximate center of the range above which a delta is considered a full edge.
   * @param marginFactor: Factor to expand the threshold range for smoothstep.
   * @returns blending strength in the range [APB_MIN_FILTER_STRENGTH, 1.0f]
   */
  float calcBlendingStrength(float4 deltas, float threshold, float marginFactor)
  {
    float4 edges = smoothstep(threshold / marginFactor, threshold * marginFactor, deltas);
    // redo to get normal deltas, use that to calc filter strength
    float cornerAmount = (edges.r + edges.b) * (edges.g + edges.a);
    // Determine filter strength based on the number of corners detected
    return max(cornerAmount / 4.0f, APB_MIN_FILTER_STRENGTH);
  }

  /**
   * Calculates a weighted average of a 9 tap pattern of pixels.
   *
   * @param float strength: strength of the effect, how much the calculated local average is applied to the final result.
   * @returns float3 localavg
   */
  float3 CalcLocalAvg(
      float3 NW, float3 N, float3 NE,
      float3 W, float3 C, float3 E,
      float3 SW, float3 S, float3 SE,
      float strength)
  {
    // idea behind this algo is to arrange neighbouring pixels into "patterns" of pixels which represent certain
    // shapes (local morphology) which we want to reinforce or weaken. Then by taking the max and min of these patterns,
    // we tend to get the shapes which match the actual morhpology of the image the most. We can use these to get
    // a local average which respects the local morphology more than a simple blur would.
    //
    // patterns are made by taking the average of a bunch of pixels: (sum of pixels) / n. But since most patterns have n = 6,
    // we can save perfomance by only taking the sums, multiplying the patterns with fewer than 6 pixels so they have the same weight,
    // and then dividing only the final localavg and any other patterns you want to use by 6 at the end ("normalizing" them).

    // pattern:
    //  e f g
    //  h a b
    //  i c d

    // these line patterns are in other patterns too, so their sums are not multiplied by anything yet.
    float3 diag1 = NW + C + SE;
    float3 diag2 = SW + C + NE;

    float3 horz = W + C + E;
    float3 vert = N + C + S;

    // Reinforced patterns (n = 6)
    float3 bottomHalf = horz + SW + S + SE;
    float3 topHalf = horz + N + NW + NE;
    float3 leftHalf = NW + W + SW + vert;
    float3 rightHalf = vert + NE + E + SE;

    float3 diagHalfNW = diag2 + N + W + NW;
    float3 diagHalfSE = diag2 + E + SE + S;
    float3 diagHalfNE = diag1 + NE + E + N;
    float3 diagHalfSW = diag1 + W + S + SW;

    // Weakened patterns (n = 5, 6/5 = 1.2)
    float3 surround = (N + S + horz) * 1.2;
    float3 diagSurround = (diag1 + NE + SW) * 1.2;

    // Line patterns (subtype of reinforced. Helps to preserve simple lines. n = 3, 6/3 = 2.0f)
    diag1 *= 2.0f;
    diag2 *= 2.0f;

    horz *= 2.0f;
    vert *= 2.0f;

    float3 maxDesired = Functions::max(leftHalf, bottomHalf, diag1, diag2, topHalf, rightHalf, diagHalfNE, diagHalfNW, diagHalfSE, diagHalfSW);
    float3 minDesired = Functions::min(leftHalf, bottomHalf, diag1, diag2, topHalf, rightHalf, diagHalfNE, diagHalfNW, diagHalfSE, diagHalfSW);

    float3 maxLine = Functions::max(horz, vert, maxDesired);
    float3 minLine = Functions::min(horz, vert, minDesired);

    float3 maxUndesired = max(surround, diagSurround);
    float3 minUndesired = min(surround, diagSurround);

    // Constants for local average calculation
    static const float undesiredAmount = 2.0f;
    static const float DesiredPatternsWeight = 2.0f;
    static const float LineWeight = 1.3f;
    // Multiply by 2.0f, because each sum is from a pair of values
    static const float LocalAvgDenominator = mad(DesiredPatternsWeight + LineWeight, 2.0f, -undesiredAmount);

    float3 undesiredSum = -maxUndesired - minUndesired;
    float3 lineSum = maxLine + minLine;
    float3 desiredSum = maxDesired + minDesired;

    lineSum = mad(lineSum, LineWeight, undesiredSum);
    desiredSum = mad(desiredSum, DesiredPatternsWeight, lineSum);
    // Also divide by 6.0f to normalise the end result. this finishes the actual calculation of the averages (patterns) earlier.
    float3 localavg = (desiredSum / LocalAvgDenominator) / 6.0f;

    // If the new target pixel value is less bright than the max desired shape, boost it's value accordingly
    float2 patternLumas = float2(Color::luma(maxLine), Color::luma(minLine)) / 6.0f;
    float localLuma = Color::luma(localavg);
    // TODO: try using delta between origLuma and localLuma to determine strength and direction of the boost/weakening
    // if new value is brighter than max desired shape, boost strength is 0.0f and localavg should be multiplied by 1.0f. Else, boost it.
    float boost = saturate(patternLumas.x - localLuma);
    float weaken = patternLumas.y - localLuma;
    float origLuma = Color::luma(C);
    float direction = APB_LUMA_PRESERVATION_BIAS + origLuma - localLuma;
    direction = saturate(mad(direction, APB_LUMA_PRESERVATION_STRENGTH, .5));
    float change = lerp(weaken, boost, direction);
    localavg *= 1.0f + change; // add to 1, because the operation must nudge the local avg, not take a fraction of it

    return lerp(C, localavg, strength);
  }
}
#endif // ANOMALOUS_PIXEL_BLENDING_FHX
