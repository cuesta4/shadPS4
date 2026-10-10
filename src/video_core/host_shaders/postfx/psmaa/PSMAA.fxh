#ifndef _PSMAA_FXH // include guard
#define _PSMAA_FXH

#include "./reshade-shared/libraries/color.fxh"
#include "./reshade-shared/libraries/functions.fxh"

//// IMPLEMENTATION
// MACROS with example values for the ReShade language:
// The following preprocessor variables should be defined in the main file.
// The values are defaults and can be changed as needed:

//// PSMAA MACROS START

// #define PSMAA_USE_SIMPLIFIED_DELTA_CALCULATION 0
#define PSMAA_BUFFER_METRICS float4(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT, BUFFER_WIDTH, BUFFER_HEIGHT)
#define PSMAA_PIXEL_SIZE BUFFER_PIXEL_SIZE
#define PSMAATexture2D(tex) sampler tex
#define PSMAASamplePoint(tex, coord) tex2D(tex, coord)
#define PSMAASamplePointOffset(tex, coord, offset) tex2Doffset(tex, coord, offset)
#define PSMAASampleLevelZero(tex, coord) tex2Dlod(tex, float4(coord, 0.0f, 0.0f))
#define PSMAASampleLevelZeroOffset(tex, coord, offset) tex2Dlodoffset(tex, float4(coord, coord), offset)
#define PSMAAGatherLeftEdges(tex, coord) tex2Dgather(tex, coord, 0); // TODO: test if this works in DX9
#define PSMAAGatherTopEdges(tex, coord) tex2Dgather(tex, coord, 1);
// #define PSMAA_PRE_PROCESSING_THRESHOLD_MULTIPLIER 1.0f
// #define PSMAA_PRE_PROCESSING_CMAA_LCA_FACTOR_MULTIPLIER 1.0f
// #define PSMAA_PRE_PROCESSING_STRENGTH 1.0f
// #define PSMAA_PRE_PROCESSING_STRENGTH_THRESH .15
// #define PSMAA_PRE_PROCESSING_EXPLICIT_ZERO true
// #define PSMAA_PRE_PROCESSING_THRESHOLD_MARGIN_FACTOR 1.5
// #define PSMAA_THRESHOLD_FLOOR 0.018
//
// #define PSMAA_EDGE_DETECTION_FACTORS_HIGH_LUMA float4(threshold, CMAALCAFactor, SMAALCAFactor, SMAALCAAdjustmentBiasByCMAALocalContrast)
// #define PSMAA_EDGE_DETECTION_FACTORS_LOW_LUMA float4(threshold, CMAALCAFactor, SMAALCAFactor, SMAALCAAdjustmentBiasByCMAALocalContrast)
//   These are float4's with the following values:
//     x: threshold
//     y: CMAA LCA factor
//     z: SMAA LCA factor
//     w: SMAA LCA adjustment bias by CMAA local contrast
// #define PSMAA_SMOOTHING_DELTA_WEIGHT_PREDICATION_FACTOR .8
#define PSMAA_SMOOTHING_DELTA_WEIGHT_FLOOR .06
#define PSMAA_SMOOTHING_THRESHOLD .05
// #define PSMAA_SMOOTHING_DELTA_WEIGHTS float2(.02, .25) // min and max delta
// #define PSMAA_SMOOTHING_DELTA_WEIGHT_DEBUG false
// #define PSMAA_SMOOTHING_THRESHOLDS float2(.025, .075) //threshold at min and max luma
// #define PSMAA_SHARPENING_COMPENSATION_STRENGTH .65f
// #define PSMAA_SHARPENING_COMPENSATION_CUTOFF .5f
// #define PSMAA_SHARPENING_EDGE_BIAS -1.0f
#define PSMAA_SHARPENING_EDGE_BIAS_WEIGHTS float4(.4, .8, 1.2, 1.6)
// #define PSMAA_SHARPENING_SHARPNESS 0.0f
// #define PSMAA_SHARPENING_BLENDING_STRENGTH 0.0f
// #define PSMAA_SHARPENING_DEBUG false

//// PSMAA MACROS END


//// APB MACROS START

// #define APB_LUMA_PRESERVATION_BIAS .5
// #define APB_LUMA_PRESERVATION_STRENGTH 1.0f
#define APB_MIN_FILTER_STRENGTH .15

#include "./reshade-shared/modules/AnomalousPixelBlending.fxh"

//// APB MACROS END


//// BEANSMOOTHING MACROS START

#define SMOOTHING_SATURATION_DIVISOR_FLOOR 0.01
#define SMOOTHING_DEBUG false
#define SMOOTHING_BUFFER_RCP_HEIGHT PSMAA_BUFFER_METRICS.y
#define SMOOTHING_BUFFER_RCP_WIDTH PSMAA_BUFFER_METRICS.x
#define SMOOTHING_MIN_ITERATIONS 5.0f
#define SMOOTHING_MAX_ITERATIONS 20.0f
// #define SMOOTHING_THRESHOLD_DEPTH_GROWTH_START .5
// #define SMOOTHING_THRESHOLD_DEPTH_GROWTH_FACTOR 2.5 // threshold multiplier based on depth

// Shorthands for sampling
#define SmoothingSampleLevelZero(tex, coord) PSMAASampleLevelZero(tex, coord)
#define SmoothingSampleLevelZeroOffset(tex, coord, offset) PSMAASampleLevelZeroOffset(tex, coord, offset)
#define SmoothingGatherLeftDeltas(tex, coord) PSMAAGatherLeftEdges(tex, coord)
#define SmoothingGatherTopDeltas(tex, coord) PSMAAGatherTopEdges(tex, coord)

#include "./reshade-shared/vendor/modules/BeanSmoothing.fxh"

//// BEANSMOOTHING MACROS END


//// CAS MACROS START

#define CAS_BETTER_DIAGONALS 1

#include "./reshade-shared/vendor/modules/CAS.fxh"

//// CAS MACROS END


//// SMAA MACROS START

// #ifdef SMAA_PRESET_CUSTOM
// 	#define SMAA_MAX_SEARCH_STEPS 32
// 	#define SMAA_MAX_SEARCH_STEPS_DIAG 19
// 	#define SMAA_CORNER_ROUNDING 10
// #endif

#define SMAA_RT_METRICS PSMAA_BUFFER_METRICS
#define SMAATexture2D(tex) PSMAATexture2D(tex)
#define SMAATexturePass2D(tex) tex
#define SMAASampleLevelZero(tex, coord) PSMAASampleLevelZero(tex, coord)
#define SMAASampleLevelZeroPoint(tex, coord) tex2Dlod(tex, float4(coord, 0.0f, 0.0f))
#define SMAASampleLevelZeroOffset(tex, coord, offset) PSMAASampleLevelZeroOffset(tex, coord, offset)
#define SMAASample(tex, coord) tex2D(tex, coord)
#define SMAASamplePoint(tex, coord) PSMAASamplePoint(tex, coord)
#define SMAASampleOffset(tex, coord, offset) tex2D(tex, coord + offset * SMAA_RT_METRICS.xy)
#define SMAA_FLATTEN [flatten]
#define SMAA_BRANCH [branch]

#include "./reshade-shared/vendor/SMAA.fxh"

//// SMAA MACROS START


namespace PSMAA
{
  /**
   * Get the luma weighted delta between two vectors
   */
  float GetDelta(float3 cA, float3 cB, float rangeA, float rangeB)
  {
    float3 cDelta = abs(cA - cB);
    float deltaLuma = Color::luma(cDelta);

#if PSMAA_USE_SIMPLIFIED_DELTA_CALCULATION == 1

    return deltaLuma;

#else

    float colorDelta = Functions::max(cDelta);
    float colorfulness = max(rangeA, rangeB);
    return lerp(deltaLuma, colorDelta, colorfulness);

#endif
  }

  float GetDelta(float3 cA, float3 cB)
  {
    float rangeA = Functions::max(cA) - Functions::min(cA);
    float rangeB = Functions::max(cB) - Functions::min(cB);
    return GetDelta(cA, cB, rangeA, rangeB);
  }

  /**
   * Get the deltas for the current pixel. Combines euclidian luma detection with color detection dynamically based on the colorfulness of the pixels.
   */
  float2 GetDeltas(float3 cLeft, float3 cTop, float3 cCurrent, float rangeLeft, float rangeTop, float rangeCurrent)
  {
    float2 deltas;
    deltas.x = GetDelta(cLeft, cCurrent, rangeLeft, rangeCurrent);
    deltas.y = GetDelta(cTop, cCurrent, rangeTop, rangeCurrent);

    return deltas;
  }

  // A number between 0 - 1 indicating whether the min or amx amount of smoothing iterations should be used.
  // What number of iterations is considered min or max is out of scope for this function and
  // should be decided elsewhere, based on this mod's value.
  float getSmoothingIterationsMod(float4 deltas, float maxLocalLuma)
  {
    float2 maxDeltaCorner = max(deltas.rb, deltas.ga);
    // Use pythagorean theorem to calculate the "weight" of the contrast of the biggest corner
    float deltaWeight = sqrt(Functions::sum(maxDeltaCorner * maxDeltaCorner));

    float2 thresholds = float2(PSMAA_SMOOTHING_DELTA_WEIGHTS.x, PSMAA_SMOOTHING_DELTA_WEIGHTS.y);
    thresholds *= mad(1.0f - maxLocalLuma, -PSMAA_SMOOTHING_DELTA_WEIGHT_PREDICATION_FACTOR, 1.0f);
    thresholds = max(thresholds, PSMAA_SMOOTHING_DELTA_WEIGHT_FLOOR);
    return smoothstep(thresholds.x, thresholds.y, deltaWeight);
  }

  void GatherNeighborDeltas(
      PSMAATexture2D(deltaTex),
      float4 gatherOffset,
      out float4 horzDeltas,
      out float4 vertDeltas)
  {
    #if __RENDERER__ >= 0xa000 // If DX10 or higher

      horzDeltas = PSMAAGatherTopEdges(deltaTex, gatherOffset.xy);
      horzDeltas = horzDeltas.wzxy;
      // [  ]  [  ]   [  ]
      // [hx]  [hy]   [  ]
      // [hz]  [hw]   [  ]
      vertDeltas = PSMAAGatherLeftEdges(deltaTex, gatherOffset.zw);
      vertDeltas = vertDeltas.wzxy;
      // [  ]  [vx]   [vy]
      // [  ]  [vz]   [vw]
      // [  ]  [  ]   [  ]

    #else // If DX9

      horzDeltas.x = PSMAASamplePoint(deltaTex, gatherOffset.xy).g;
      horzDeltas.z = PSMAASamplePointOffset(deltaTex, gatherOffset.xy, int2(0,1)).g;
      horzDeltas.w = PSMAASamplePointOffset(deltaTex, gatherOffset.xy, int2(1,1)).g;

      vertDeltas.x = PSMAASamplePoint(deltaTex, gatherOffset.zw).r;
      vertDeltas.y = PSMAASamplePointOffset(deltaTex, gatherOffset.zw, int2(1,0)).r;
      vertDeltas.w = PSMAASamplePointOffset(deltaTex, gatherOffset.zw, int2(1,1)).r;

      float2 center = PSMAASamplePointOffset(deltaTex, gatherOffset.xy, int2(1,0)).rg;
      horzDeltas.y = center.g;
      vertDeltas.z = center.r;

    #endif
  }

  float2 CalculateCMAALocalContrast(float4 vertDeltas, float4 horzDeltas, float cmaaLCAFactor)
  {
    float2 cmaaLCA;
    cmaaLCA.r = Functions::max(vertDeltas.x, vertDeltas.y, vertDeltas.z, horzDeltas.w);
    cmaaLCA.g = Functions::max(horzDeltas.x, horzDeltas.y, horzDeltas.z, vertDeltas.w);
    return cmaaLCA * cmaaLCAFactor;
  }

  float2 DetectEdges(float2 deltas, float threshold, float2 cmaaLCA)
  {
    float2 currDeltas = deltas - cmaaLCA.rg;
    return step(threshold, currDeltas);
  }

  float2 GetSMAAExtremesDeltas(PSMAATexture2D(deltaTex), float4 offset)
  {
    return float2(
        PSMAASamplePoint(deltaTex, offset.xy).r, // leftLeftDelta
        PSMAASamplePoint(deltaTex, offset.zw).g  // topTopDelta
    );
  }

  float2 ApplySMAALCA(
      float2 edges,
      // x = bottom, y = top, z = toptop
      float3 horizontalDeltas,
      // x = right, y = left, z = leftleft
      float3 verticalDeltas,
      // x: SMAA's local contrast adaptation factor
      // y: SMAA LCA adjustment bias by CMAA local contrast
      float2 LCAFactors,
      float2 cmaaLocalContrast)
  {
    float2 maxDeltas = float2(Functions::max(verticalDeltas), Functions::max(horizontalDeltas));
    float finalDelta = max(maxDeltas.r, maxDeltas.g);

    float2 currDeltas = mad(cmaaLocalContrast, LCAFactors.y, float2(verticalDeltas.y, horizontalDeltas.y));
    edges.rg *= step(finalDelta, LCAFactors.x * currDeltas);

    return edges;
  }

  /**
   * Maps (scales) an input to an output such that the speed at wich the output grows, the max value it reaches
   * and when that max value is reached can be controlled. When the max value is reached, the output just "flatlines".
   *
   * @param ceiling the maximum value the output can reach.
   * @param cutoffValue the value of the input at which the ceiling is reached
   */
  float scaleSignal(float input, float ceiling, float cutoffValue)
  {
    float growthspeed = ceiling / cutoffValue;
    return min(input * growthspeed, ceiling);
  }

  float calcEdgeBiasStrength(float4 deltas)
  {
    static const float4 Weights = PSMAA_SHARPENING_EDGE_BIAS_WEIGHTS;

    float2 transverseMax = max(deltas.rg, deltas.ba);
    float2 transverseMin = min(deltas.rg, deltas.ba);

    float cornerDelta = Functions::min(transverseMax);
    float lineDelta = Functions::max(transverseMin);

    float4 factors;
    // straight edge delta (largest delta)
    factors.r = Functions::max(transverseMax);
    // corner delta (2nd largest delta)
    factors.g = max(cornerDelta, lineDelta);
    // cup delta (3rd largest delta)
    factors.b = min(cornerDelta, lineDelta);
    // pix delta (smallest delta)
    factors.a = Functions::min(transverseMin);

    return dot(factors, Weights) * PSMAA_SHARPENING_EDGE_BIAS;
  }

  namespace Pass
  {
    void PreProcessingPS(
        float2 texcoord,
        PSMAATexture2D(colorGammaTex), // input color texture (C)
        out float maxLocalLuma,        // output maximum luma from all nine samples
        out float originalLuma,        // luma of the original color, for change detection
        out float2 filteringStrength   // strength at which FilteringPS is determned to run on this pixel
    )
    {
      // NW N NE
      // W  C  E
      // SW S SE
      float3 C, E, S, SE;
#if __RENDERER__ >= 0xa000 // if DX10 or above
      // get RGB values from the c, d, b, and a positions, in order.
      float4 cdbared = tex2Dgather(colorGammaTex, texcoord, 0);
      float4 cdbagreen = tex2Dgather(colorGammaTex, texcoord, 1);
      float4 cdbablue = tex2Dgather(colorGammaTex, texcoord, 2);

      C = float3(cdbared.w, cdbagreen.w, cdbablue.w);
      E = float3(cdbared.z, cdbagreen.z, cdbablue.z);
      S = float3(cdbared.x, cdbagreen.x, cdbablue.x);
      SE = float3(cdbared.y, cdbagreen.y, cdbablue.y);
#else // if DX9
      C = PSMAASampleLevelZero(colorGammaTex, texcoord).rgb;
      E = PSMAASampleLevelZeroOffset(colorGammaTex, texcoord, float2(1, 0)).rgb;
      S = PSMAASampleLevelZeroOffset(colorGammaTex, texcoord, float2(0, 1)).rgb;
      SE = PSMAASampleLevelZeroOffset(colorGammaTex, texcoord, float2(1, 1)).rgb;
#endif

      float3 NW = PSMAASampleLevelZeroOffset(colorGammaTex, texcoord, float2(-1, -1)).rgb;
      float3 W = PSMAASampleLevelZeroOffset(colorGammaTex, texcoord, float2(-1, 0)).rgb;
      float3 SW = PSMAASampleLevelZeroOffset(colorGammaTex, texcoord, float2(-1, 1)).rgb;
      float3 N = PSMAASampleLevelZeroOffset(colorGammaTex, texcoord, float2(0, -1)).rgb;
      float3 NE = PSMAASampleLevelZeroOffset(colorGammaTex, texcoord, float2(1, -1)).rgb;

      float3 maxLocalColor = Functions::max(NW, W, SW, N, S, NE, E, SE, C);
      // These make sure that Red and Blue don't count as much as Green,
      // without making all results darker when taking the greatest component
      static const float3 LumaCorrection = float3(.297, 1.0f, .101);

      // OUTPUT
      maxLocalLuma = Functions::max(maxLocalColor * LumaCorrection); // TODO; replace with proper luma function
      // OUTPUT
      originalLuma = Color::luma(C);

      float4 deltas;
      deltas.r = GetDelta(W, C);
      deltas.g = GetDelta(N, C);
      deltas.b = GetDelta(E, C);
      deltas.a = GetDelta(S, C);

      // Use detection factors for edge detection here too, so that the results of this pass scale proportionally to the needs of edge detection.
      float2 detectionFactor = lerp(PSMAA_EDGE_DETECTION_FACTORS_LOW_LUMA.x, PSMAA_EDGE_DETECTION_FACTORS_HIGH_LUMA.x, maxLocalLuma);
      // scale with multipliers specific to this pass
      detectionFactor *= PSMAA_PRE_PROCESSING_THRESHOLD_MULTIPLIER;
      // Minimum threshold to prevent blending in very dark areas
      float threshold = max(detectionFactor, PSMAA_THRESHOLD_FLOOR);

      float4 edges = step(threshold, deltas);
      if (Functions::sum(edges) < 2.0f) // Leave filter strength as 0.0f for straight lines, to prevent blur
      {
        // OUTPUT
        filteringStrength = float2(0.0f, 0.0f);
        return;
      }

      // greatest corner correction and corner check
      float isCorner = AnomalousPixelBlending::checkIfCorner(deltas, PSMAA_PRE_PROCESSING_GREATEST_CORNER_CORRECTION_STRENGTH, threshold) ? 1.0f : 0.0f;

      float strength = AnomalousPixelBlending::calcBlendingStrength(deltas, threshold, PSMAA_PRE_PROCESSING_THRESHOLD_MARGIN_FACTOR) * PSMAA_PRE_PROCESSING_STRENGTH;

      // OUTPUT
      filteringStrength = float2(strength, isCorner);
    }

    void FilteringPS(
        float2 texcoord,
        PSMAATexture2D(colorLinearTex),
        PSMAATexture2D(filterStrengthTex),
        out float4 filteredColor)
    {
      float2 strengthAndIsCorner = PSMAASamplePoint(filterStrengthTex, texcoord).rg;

      if (strengthAndIsCorner.y >= .9f || strengthAndIsCorner.x <= PSMAA_PRE_PROCESSING_STRENGTH_THRESH)
        { postfx_discard = true; return ; } // skip if corner or no filtering needed

      // NW N NE
      // W  C  E
      // SW S SE
      // Keep the alpha from the original texture
      float4 CRaw = PSMAASampleLevelZero(colorLinearTex, texcoord);
      float3 C = CRaw.rgb;
      float3 NW = PSMAASampleLevelZeroOffset(colorLinearTex, texcoord, float2(-1, -1)).rgb;
      float3 W = PSMAASampleLevelZeroOffset(colorLinearTex, texcoord, float2(-1, 0)).rgb;
      float3 SW = PSMAASampleLevelZeroOffset(colorLinearTex, texcoord, float2(-1, 1)).rgb;
      float3 N = PSMAASampleLevelZeroOffset(colorLinearTex, texcoord, float2(0, -1)).rgb;
      float3 S = PSMAASampleLevelZeroOffset(colorLinearTex, texcoord, float2(0, 1)).rgb;
      float3 NE = PSMAASampleLevelZeroOffset(colorLinearTex, texcoord, float2(1, -1)).rgb;
      float3 E = PSMAASampleLevelZeroOffset(colorLinearTex, texcoord, float2(1, 0)).rgb;
      float3 SE = PSMAASampleLevelZeroOffset(colorLinearTex, texcoord, float2(1, 1)).rgb;

      float3 filteredLocalAvg = AnomalousPixelBlending::CalcLocalAvg(
          NW, N, NE, W, C, E, SW, S, SE,
          strengthAndIsCorner.x);

      // OUTPUT with localavg and original alpha
      filteredColor = float4(filteredLocalAvg, CRaw.a);
    }

    /**
     * Calculates the offsets for the current pixel.
     * Decided to leave the offset as an array, because I'll likely need more values in the future.
     */
    void DeltaCalculationVS(float2 texcoord, out float4 offset[1])
    {
      offset[0] = mad(PSMAA_BUFFER_METRICS.xyxy, float4(-1.0, 0.0, 0.0, -1.0), texcoord.xyxy);
    }

    /**
     * Calculate the top and left deltas for the current pixel.
     */
    void DeltaCalculationPS(float2 texcoord, float4 offset[1], PSMAATexture2D(colorGammaTex), out float2 deltas)
    {
      float3 current = PSMAASamplePoint(colorGammaTex, texcoord).rgb;
      float3 left = PSMAASamplePoint(colorGammaTex, offset[0].xy).rgb;
      float3 top = PSMAASamplePoint(colorGammaTex, offset[0].zw).rgb;

      float rangeCurrent = Functions::max(current) - Functions::min(current);
      float rangeLeft = Functions::max(left) - Functions::min(left);
      float rangeTop = Functions::max(top) - Functions::min(top);

      deltas = GetDeltas(left, top, current, rangeLeft, rangeTop, rangeCurrent);
    }

    void EdgeDetectionVS(float2 texcoord, out float4 offset[2])
    {
      offset[0] = mad(PSMAA_PIXEL_SIZE.xyxy, float4(-.5, 0.0, 0.0, -.5), texcoord.xyxy);
      offset[1] = mad(PSMAA_BUFFER_METRICS.xyxy, float4(-1.0, 0.0, 0.0, -1.0), texcoord.xyxy);
    }
    /**
     * Calculate the edges for the current pixel.
     * Temporary implementation for testing purposes. Should not be used in production for now.
     */
    void EdgeDetectionPS(
        float2 texcoord,
        float4 offset[2],
        PSMAATexture2D(deltaTex),
        PSMAATexture2D(lumaTex),
        out float2 edgesOutput)
    {
      float4 horzDeltas, vertDeltas;
      GatherNeighborDeltas(deltaTex, offset[0], horzDeltas, vertDeltas);

      float localLuma = PSMAASamplePoint(lumaTex, texcoord).r;
      // Adjust threshold and LCA factors according to the max local luma
      float4 detectionFactors = lerp(PSMAA_EDGE_DETECTION_FACTORS_LOW_LUMA, PSMAA_EDGE_DETECTION_FACTORS_HIGH_LUMA, localLuma);
      detectionFactors.x = max(detectionFactors.x, PSMAA_THRESHOLD_FLOOR);

      float2 cmaaLCA = CalculateCMAALocalContrast(vertDeltas, horzDeltas, detectionFactors.y);
      float2 edges = DetectEdges(float2(vertDeltas.z, horzDeltas.y), detectionFactors.x, cmaaLCA);

      // discard if there is no edge
      if (dot(edges, float2(1.0, 1.0)) == 0.0f)
        { postfx_discard = true; return ; }

      // Get extremes for SMAA LCA
      float2 extremesDeltas = GetSMAAExtremesDeltas(deltaTex, offset[1]);

      // [  ]  [  ]   [  ]
      // [e.y] [vz]   [vw]
      // [  ]  [  ]   [  ]
      float3 vertDeltas2 = float3(vertDeltas.w, vertDeltas.z, extremesDeltas.x);
      // [  ]  [e.y]  [  ]
      // [  ]  [hy]   [  ]
      // [  ]  [hw]   [  ]
      float3 horzDeltas2 = float3(horzDeltas.w, horzDeltas.y, extremesDeltas.y);

      edgesOutput = ApplySMAALCA(edges, horzDeltas2, vertDeltas2, detectionFactors.zw, cmaaLCA);
    }

    /**
     * Wrapper around BeanSmoothing
     */
    void SmoothingPS(
        float2 texcoord,
        float4 offset,
        sampler deltaTex,
        sampler blendSampler,
        sampler colorTex,
        sampler lumaTex,
        out float3 color)
    {
      float4 deltas;
#if __RENDERER__ >= 0xa000 // if DX10 or above
      // get edge data from the bottom (x), bottom-right (y), right (z),
      // and current pixels (w), in that order.
      float4 leftDeltas = SmoothingGatherLeftDeltas(deltaTex, texcoord);
      float4 topDeltas = SmoothingGatherTopDeltas(deltaTex, texcoord);
      deltas = float4(
          leftDeltas.w,
          topDeltas.w,
          leftDeltas.z,
          topDeltas.x);
#else // if DX9
      deltas = float4(
          SmoothingSampleLevelZero(deltaTex, texcoord).rg,
          SmoothingSampleLevelZero(deltaTex, offset.xy).r,
          SmoothingSampleLevelZero(deltaTex, offset.zw).g);
#endif

      float maxLocalLuma = tex2D(lumaTex, texcoord).r;

      float mod = PSMAA::getSmoothingIterationsMod(deltas, maxLocalLuma);
      float threshold = lerp(PSMAA_SMOOTHING_THRESHOLDS.x, PSMAA_SMOOTHING_THRESHOLDS.y, maxLocalLuma);




      // TODO: consider turning into prepreocssor check for performance.
      // Consider turning into func that can detect early returns too by checking delta between new and old color
      if (PSMAA_SMOOTHING_DELTA_WEIGHT_DEBUG)
      {
        int maxIterations = BeanSmoothing::calcMaxSmoothingIterations(mod);
        float3 result = BeanSmoothing::smooth(texcoord, offset, colorTex, blendSampler, threshold, maxIterations);

        // Check if there's a diff between original and result
        // Helps to detect cases where smooth() did an early return and changed nothing
        float3 original = tex2D(colorTex, texcoord).rgb;
        // increase the change to make it more visible, especially for small changes
        float change = saturate(sqrt(GetDelta(original, result) * 9.0f));
        color = float3(0.0f, change, mod);
        return;
      }

      // if close to 0.0f, discard.
      if (mod < 1e-5f)
        { postfx_discard = true; return ; } // TODO: make standard 'nullish' function check.

      uint maxIterations = BeanSmoothing::calcMaxSmoothingIterations(mod);

      color = BeanSmoothing::smooth(texcoord, offset, colorTex, blendSampler, threshold, maxIterations);
    }

    void BlendingPS(
        float2 texcoord,
        float4 offset,
        sampler colorLinearSampler,
        sampler blendWeightSampler,
        sampler filterStrengthSampler,
        out float4 color)
    {
      color = SMAANeighborhoodBlendingPS(
                  texcoord,
                  offset,
                  colorLinearSampler,
                  blendWeightSampler)
                  .rgba;

      float2 strengthAndIsCorner = PSMAASamplePoint(filterStrengthSampler, texcoord).rg;
      if (strengthAndIsCorner.y < .9f || strengthAndIsCorner.x <= PSMAA_PRE_PROCESSING_STRENGTH_THRESH)
        return;

      float3 NW = PSMAASampleLevelZeroOffset(colorLinearSampler, texcoord, float2(-1, -1)).rgb;
      float3 W = PSMAASampleLevelZeroOffset(colorLinearSampler, texcoord, float2(-1, 0)).rgb;
      float3 SW = PSMAASampleLevelZeroOffset(colorLinearSampler, texcoord, float2(-1, 1)).rgb;
      float3 N = PSMAASampleLevelZeroOffset(colorLinearSampler, texcoord, float2(0, -1)).rgb;
      float3 S = PSMAASampleLevelZeroOffset(colorLinearSampler, texcoord, float2(0, 1)).rgb;
      float3 NE = PSMAASampleLevelZeroOffset(colorLinearSampler, texcoord, float2(1, -1)).rgb;
      float3 E = PSMAASampleLevelZeroOffset(colorLinearSampler, texcoord, float2(1, 0)).rgb;
      float3 SE = PSMAASampleLevelZeroOffset(colorLinearSampler, texcoord, float2(1, 1)).rgb;

      float3 filteredLocalAvg = AnomalousPixelBlending::CalcLocalAvg(
          NW, N, NE, W, color.rgb, E, SW, S, SE,
          strengthAndIsCorner.x);

      // OUTPUT with localavg and original alpha
      color = float4(filteredLocalAvg, color.a);
    }

    void SharpeningPS(
        float2 texcoord, // Integer pixel position in output.
        sampler initialLumaSampler,
        sampler colorGammaSampler,
        sampler deltaSampler,
        sampler colorLinearSampler,
        out float3 pix)
    {
      static const float NearZero = 1.0f / 127.0f;
      float initLuma = tex2D(initialLumaSampler, texcoord).r;
      float3 currentColor = tex2D(colorGammaSampler, texcoord).rgb;
      float currentLuma = Color::luma(currentColor);
      float blendingChange = saturate(abs(currentLuma - initLuma) - NearZero);

      pix = float(blendingChange).xxx;

      if ((blendingChange + PSMAA_SHARPENING_BLENDING_STRENGTH) <= 0.0f)
        { postfx_discard = true; return ; }

      // Shape blending change val limit compensation strength at extreme values
      float compensationStrength = scaleSignal(blendingChange, PSMAA_SHARPENING_COMPENSATION_STRENGTH, PSMAA_SHARPENING_COMPENSATION_CUTOFF);

      float4 deltas;
#if __RENDERER__ >= 0xa000 // if DX10 or above
      // get edge data from the bottom (x), bottom-right (y), right (z),
      // and current pixels (w), in that order.
      float4 leftDeltas = tex2Dgather(deltaSampler, texcoord, 0);
      float4 topDeltas = tex2Dgather(deltaSampler, texcoord, 1);
      deltas = float4(
          leftDeltas.w,
          topDeltas.w,
          leftDeltas.z,
          topDeltas.x);
#else // if DX9
      float4 offset = mad(PSMAA_BUFFER_METRICS.xyxy, float4(1.0f, 0.0f, 0.0f, 1.0f), texcoord.xyxy);
      deltas = float4(
          PSMAASampleLevelZero(deltaSampler, texcoord).rg,
          PSMAASampleLevelZero(deltaSampler, offset.xy).r,
          PSMAASampleLevelZero(deltaSampler, offset.zw).g);
#endif

      // Reduce sharpening strength based on number of edges
      // Edge Bias is negative or 0, so edgeBiasStrength is <= 0.0f;
      float edgeBiasStrength = calcEdgeBiasStrength(deltas);

      float sharpeningStrength = PSMAA_SHARPENING_BLENDING_STRENGTH + compensationStrength + edgeBiasStrength;

      if (PSMAA_SHARPENING_DEBUG)
      {
        pix = saturate(sharpeningStrength);
        return;
      }

      if (sharpeningStrength <= 0.0f)
        { postfx_discard = true; return ; }

      // sharpeningStrength of up to 1.0f is used to determine the blending strength...
      float blendingStrength = saturate(sharpeningStrength);
      // ...everything above that is sheared off and added to the sharpness
      float sharpness = saturate(PSMAA_SHARPENING_SHARPNESS + saturate(sharpeningStrength - 1.0f));

      float const1;
      CAS::CasSetup(const1, sharpness);
      CAS::CasFilter(texcoord, const1, blendingStrength, colorLinearSampler, pix);
    }
  }
}
#endif // PSMAA_FXH include guard
