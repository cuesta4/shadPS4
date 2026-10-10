// SPDX-FileCopyrightText: Copyright 2025 RdenBlaauwen
// SPDX-License-Identifier: LicenseRef-PSMAA-ThirdParty

#include "common.hlsl"
static bool postfx_discard = false;
#define __RENDERER__ 0x20000
#define SMAA_PRESET_CUSTOM
#define SMAA_CUSTOM_SL 1
#define PSMAA_USE_SIMPLIFIED_DELTA_CALCULATION 0
static const int _UIHelpText = 0;
static const float _PreProcessingThresholdMultiplier = 3.5f;
static const float _PreProcessingThresholdMargin = 1.8f;
static const float _PreProcessingStrength = .7;
static const float _PreProcessingStrengthThresh = .149;
static const float _PreProcessingLumaPreservationBias = .5f;
static const float _PreProcessingLumaPreservationStrength = 2.0f;
static const float _PreProcessingGreatestCornerCorrectionStrength = .85;
static const float2 _EdgeDetectionThreshold = float2(.005, .07);
static const float2 _CMAALCAFactor = float2(.22, .15);
static const float2 _SMAALCAFactor = float2(2.0f, 2.0f);
static const float2 _CMAALCAforSMAALCAFactor = float2(-.45, 0);
static const float _ThreshFloor = .015;
static const int _MaxSearchSteps = 32;
static const int _MaxSearchStepsDiag = 19;
static const int _CornerRounding = 0;
static const bool _SmoothingEnabled = true;
static const bool _SmoothingDeltaWeightDebug = false;
static const float2 _SmoothingDeltaWeights = float2(.1, .5);
static const float _SmoothingDeltaWeightDynamicThreshold = .65;
static const float2 _SmoothingThresholds = float2(.015, .09);
static const float _SmoothingThresholdDepthGrowthStart = .35;
static const float _SmoothingThresholdDepthGrowthFactor = 2.5;
static const bool _SharpeningEnabled = false;
static const float _SharpeningCompensationStrength = 1.5;
static const float _SharpeningCompensationCutoff = .15;
static const float _SharpeningEdgeBias = -1.5f;
static const float _SharpeningSharpness = 0.0f;
static const float _SharpeningBlendingStrength = .7;
static const bool _SharpeningDebug = false;
static const int _Debug = 0;
static const int _MacroHelpText = 0;
// PSMAA preprocessor variables
#define PSMAA_THRESHOLD_FLOOR _ThreshFloor
#define PSMAA_PRE_PROCESSING_THRESHOLD_MULTIPLIER _PreProcessingThresholdMultiplier
#define APB_LUMA_PRESERVATION_BIAS _PreProcessingLumaPreservationBias
#define APB_LUMA_PRESERVATION_STRENGTH _PreProcessingLumaPreservationStrength
#define PSMAA_PRE_PROCESSING_STRENGTH _PreProcessingStrength
#define PSMAA_PRE_PROCESSING_STRENGTH_THRESH _PreProcessingStrengthThresh
#define PSMAA_PRE_PROCESSING_GREATEST_CORNER_CORRECTION_STRENGTH _PreProcessingGreatestCornerCorrectionStrength
#define PSMAA_PRE_PROCESSING_THRESHOLD_MARGIN_FACTOR _PreProcessingThresholdMargin
#define PSMAA_EDGE_DETECTION_FACTORS_HIGH_LUMA float4(_EdgeDetectionThreshold.y, _CMAALCAFactor.y, _SMAALCAFactor.y, _CMAALCAforSMAALCAFactor.y)
#define PSMAA_EDGE_DETECTION_FACTORS_LOW_LUMA float4(_EdgeDetectionThreshold.x, _CMAALCAFactor.x, _SMAALCAFactor.x, _CMAALCAforSMAALCAFactor.x)
#define PSMAA_SMOOTHING_DELTA_WEIGHT_DEBUG _SmoothingDeltaWeightDebug
#define PSMAA_SMOOTHING_DELTA_WEIGHTS _SmoothingDeltaWeights
#define PSMAA_SMOOTHING_DELTA_WEIGHT_PREDICATION_FACTOR _SmoothingDeltaWeightDynamicThreshold
#define PSMAA_SMOOTHING_THRESHOLDS _SmoothingThresholds
#define SMOOTHING_THRESHOLD_DEPTH_GROWTH_START _SmoothingThresholdDepthGrowthStart
#define SMOOTHING_THRESHOLD_DEPTH_GROWTH_FACTOR _SmoothingThresholdDepthGrowthFactor
#ifndef PSMAA_SMOOTHING_USE_COLOR_SPACE
	#define PSMAA_SMOOTHING_USE_COLOR_SPACE 0
#endif
#define PSMAA_SHARPENING_COMPENSATION_STRENGTH _SharpeningCompensationStrength
#define PSMAA_SHARPENING_COMPENSATION_CUTOFF _SharpeningCompensationCutoff
#define PSMAA_SHARPENING_EDGE_BIAS _SharpeningEdgeBias
#define PSMAA_SHARPENING_SHARPNESS _SharpeningSharpness
#define PSMAA_SHARPENING_BLENDING_STRENGTH _SharpeningBlendingStrength
#define PSMAA_SHARPENING_DEBUG _SharpeningDebug

#ifdef SMAA_PRESET_CUSTOM
	#define SMAA_MAX_SEARCH_STEPS _MaxSearchSteps
	#define SMAA_MAX_SEARCH_STEPS_DIAG _MaxSearchStepsDiag
	#define SMAA_CORNER_ROUNDING _CornerRounding

	// dummy values. These don't do anything, but defining them keeps
	// them from showing up in the UI as preprocessor variables:
	#define SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR _SMAALCAFactor.y
	#define SMAA_THRESHOLD _EdgeDetectionThreshold.y
	#define SMAA_DEPTH_THRESHOLD (.1 * SMAA_THRESHOLD)
	#define SMAA_PREDICATION 0
	#define SMAA_PREDICATION_THRESHOLD .01
	#define SMAA_PREDICATION_SCALE 2.0f
	#define SMAA_PREDICATION_STRENGTH .4
	#define SMAA_REPROJECTION 0
	#define SMAA_REPROJECTION_WEIGHT_SCALE 30.0
#endif


#include "psmaa/PSMAA.fxh"

[numthreads(8,8,1)] void main(uint3 id : SV_DispatchThreadID) {
 if (any(id.xy >= uint2(params.target.zw))) return;
 float2 uv = (float2(id.xy) + 0.5) * params.metrics.xy;
 float4 color = tex2D(MakeTex(tex0), uv);
#if STAGE == 0
 float luma = 0, original = 0; float2 strength = 0;
 PSMAA::Pass::PreProcessingPS(uv, MakeTex(tex0), luma, original, strength);
 output0[id.xy] = float4(luma,0,0,0); output2[id.xy] = float4(strength,0,0);
#elif STAGE == 1
 float4 filtered = float4(Decode(color.rgb), 1);
 PSMAA::Pass::FilteringPS(uv, MakeTex(tex0,true,true), MakeTex(tex1), filtered);
 if (!postfx_discard) color = float4(Encode(filtered.rgb),1);
 output0[id.xy] = color;
#elif STAGE == 2
 float4 offsets[1]; float2 deltas = 0;
 PSMAA::Pass::DeltaCalculationVS(uv, offsets);
 PSMAA::Pass::DeltaCalculationPS(uv, offsets, MakeTex(tex0), deltas);
 output0[id.xy] = float4(deltas,0,0);
#elif STAGE == 3
 float4 offsets[2]; float2 edges = 0;
 PSMAA::Pass::EdgeDetectionVS(uv, offsets);
 PSMAA::Pass::EdgeDetectionPS(uv, offsets, MakeTex(tex1), MakeTex(tex2), edges);
 output0[id.xy] = postfx_discard ? 0 : float4(edges,0,0);
#elif STAGE == 4
 float2 pixcoord; float4 offsets[3];
 SMAABlendingWeightCalculationVS(uv, pixcoord, offsets);
 output0[id.xy] = SMAABlendingWeightCalculationPS(uv,pixcoord,offsets,MakeTex(tex1),MakeTex(tex6),MakeTex(tex7),0.0);
#elif STAGE == 5
 float4 offset; float4 blended = float4(Decode(color.rgb),1);
 SMAANeighborhoodBlendingVS(uv,offset);
 PSMAA::Pass::BlendingPS(uv,offset,MakeTex(tex0,true,true),MakeTex(tex1),MakeTex(tex2),blended);
 output0[id.xy] = postfx_discard ? color : float4(Encode(blended.rgb),1);
#elif STAGE == 6
 float4 offset; float3 smoothed = color.rgb;
 SMAANeighborhoodBlendingVS(uv,offset);
 PSMAA::Pass::SmoothingPS(uv,offset,MakeTex(tex1),MakeTex(tex2),MakeTex(tex0),MakeTex(tex3),smoothed);
 output0[id.xy] = postfx_discard ? color : float4(smoothed,1);
#endif
}
