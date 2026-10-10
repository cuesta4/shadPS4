#define SMAA_PRESET_CUSTOM
#define SMAA_CUSTOM_SL 1

#include "ReShadeUI.fxh"

uniform int _UIHelpText<
    ui_type = "radio";
    ui_category = "UI Explanation";
    ui_label = "    ";
		ui_category_closed = false;
    ui_text =
			"This is a beta build of PSMAA, and this UI is primarily for development\n"
			"purposes. They're basically 'advanced' settings, which is why this UI\n"
			"may feel pretty opaque. In case you just want to know which settings are\n"
			"most important for image quality, see:\n"
			"  - pre-processing > blending strength\n"
			"  - Edge detection > Edge detection threshold\n"
			"  - Smoothing > Enable smoothing\n"
			"  - Sharpening > Enable Sharpening\n"
			"               > Blur Compensation\n"
			"               > Edge bias\n"
			"               > Blending strength\n"
			"\n"
			"For explanation on what these do, check the UI controls' tooltips";
>;

// uniform bool _ShowOldPreProcessing < // TODO: remove?
// 	ui_category = "Pre-Processing";
// 	ui_label = "Use old pre-processing";
// > = false;

uniform float _PreProcessingThresholdMultiplier <
	ui_category = "Pre-Processing";
	ui_label = "Threshold multiplier";
	ui_type = "slider";
	ui_min = 1f;
	ui_max = 10f;
	ui_step = .1;
	ui_category_closed = true;
	ui_tooltip =
		"How much higher the pre-processing threshold is than\n"
		"the edge detection threshold. Higher values make pre-processing\n"
		"less sensitive to edges, preserving detail.\n"
		"Recommended values [2.2 - 4.5]";
> = 3.5f;

uniform float _PreProcessingThresholdMargin <
	ui_category = "Pre-Processing";
	ui_label = "Threshold margin";
	ui_type = "slider";
	ui_min = 1f;
	ui_max = 2f;
	ui_step = .01;
	ui_tooltip =
		"Turns the threshold into a soft threshold.\n"
		"Higher values mean deltas just below the threshold count more, while\n"
		"those just above count less, making the effect more gradual and precise.\n"
		"Recommended values [1.5 - 1.9]";
> = 1.8f;

uniform float _PreProcessingStrength <
	ui_category = "Pre-Processing";
	ui_label = "Blending strength";
	ui_type = "slider";
	ui_min = 0f;
	ui_max = 1f;
	ui_step = .01;
	ui_tooltip =
		"How much the resulting of the pass is applied to the output percentually.\n"
		"Recommended values [.45 - .85]";
> = .7;

uniform float _PreProcessingStrengthThresh <
	ui_category = "Pre-Processing";
	ui_label = "Min strength for filtering";
	ui_type = "slider";
	ui_min = 0f;
	ui_max = .15f;
	ui_step = .001;
	ui_tooltip =
		"The algorithm assigns each pixel a value representing how anomalous that\n"
		"pixel is. Values below this threshold are skipped (i.e. not softened).\n"
		"Higher values improve performance and image clarity,\n"
		"but may leave more anomalous pixels untreated.\n"
		"Recommended values [.05 - .15]";
> = .149;

uniform float _PreProcessingLumaPreservationBias <
	ui_category = "Pre-Processing";
	ui_label = "Luma preservation bias";
	ui_type = "slider";
	ui_min = -.8f;
	ui_max = .8f;
	ui_step = .05;
	ui_tooltip =
		"Bias for preserving pixels' brightness during pre-processing.\n"
		"Positive values preserve brightness, negative values preserve darkness.\n"
		"Recommended values [-0.5 - .8]";
> = .5f;

uniform float _PreProcessingLumaPreservationStrength <
	ui_category = "Pre-Processing";
	ui_label = "Luma preservation strength";
	ui_type = "slider";
	ui_min = 1f;
	ui_max = 3f;
	ui_step = .05;
	ui_tooltip =
		"Strength of luma preservation mechanism during pre-processing.\n"
		"Recommended values [1 - 2.5]";
> = 2f;

uniform float _PreProcessingGreatestCornerCorrectionStrength <
	ui_category = "Pre-Processing";
	ui_label = "Largest corner correction strength";
	ui_type = "slider";
	ui_min = 0f;
	ui_max = 1f;
	ui_step = .01;
	ui_tooltip =
		"The pre-processing pass performs a corner-check to prevent corners\n"
		"from being softened, to prevent interference with the edge detection pass.\n"
		"This mechanism makes the algorithm better at differentiating corners from\n"
		"other shapes, but may cause false negatives sometimes.\n"
		"Recommended values [.6 - 1.0]";
> = .85;

// uniform bool _UseOldBlending < // TODO: remove?
// 	ui_category = "Blending";
// 	ui_label = "Use old blending";
// 	ui_tooltip = "Use the older blending algorithm instead of the newer one.
// Old blending may be useful for compatibility or specific visual preferences.";
// > = false;

uniform float2 _EdgeDetectionThreshold <
	ui_category = "Edge detection";
	ui_label = "Edge detection threshold";
	ui_type = "slider";
	ui_min = .004;
	ui_max = .15;
	ui_step = .001;
	ui_category_closed = false;
	ui_tooltip =
		"Thresholds for detecting edges during anti-aliasing.\n"
		"The left value is for darker areas, the right is for lighter areas.\n"
		"Recommended values [(.005, 0.05) - (0.025, 0.15)]";
> = float2(.005, .07);

uniform float2 _CMAALCAFactor <
	ui_category = "Edge detection";
	ui_label = "Circumferential LCA factor";
	ui_type = "slider";
	ui_min = 0;
	ui_max = .3;
	ui_step = .01;
	ui_tooltip =
		"Local contrast adaptation factors for deltas surrounding potential edges.\n"
		"Makes edge detection less sensitive for pixels where neighbouring pixels\n"
		"have large differences between each other, to prevent spurious detections.\n"
		"Higher values tend to reduce artifacts, especially in noisy areas,\n"
		"but may cause it to miss some edges.\n"
		"The left value is for darker areas, the right is for lighter areas.\n"
		"Recommended values [.1 - .3]";
> = float2(.22, .15);

uniform float2 _SMAALCAFactor <
	ui_category = "Edge detection";
	ui_label = "Parallel LCA factors";
	ui_type = "slider";
	ui_min = 1.5;
	ui_max = 4f;
	ui_step = .1;
	ui_tooltip =
	  "Local contrast adaptation factors for deltas parallel to potential edges.\n"
		"Makes edge detection less sensitive for pixels where parallel deltas along\n"
		"the same axis are large. Lower values tend to reduce artifacts, especially\n"
		"in gradients, but may miss some edges.\n"
		"This is the same LCA used in vanilla SMAA.\n"
		"The left value is for darker areas, the right is for lighter areas.\n"
		"Recommended values [1.6 - 3.5]";
> = float2(2f, 2f);

uniform float2 _CMAALCAforSMAALCAFactor <
	ui_category = "Edge detection";
	ui_label = "LCA synergy factors";
	ui_type = "slider";
	ui_min = -1;
	ui_max = 1;
	ui_step = .01;
	ui_tooltip =
		"Factors for how circumferential LCA affects parallel LCA.\n"
		"Negative values lower the parallel LCA when circumferential LCA is strong.\n"
		"This may reduce artifacts, but may also cause false negatives.\n"
		"The left value is for darker areas, the right is for lighter areas.\n"
		"Recommended values [-0.5 - 0]";
> = float2(-.45, 0);

uniform float _ThreshFloor < __UNIFORM_DRAG_FLOAT1
	ui_category = "Edge detection";
	ui_label = "Threshold floor";
	ui_min = .004;
	ui_max = .03;
	ui_step = .001;
	ui_tooltip =
		"The absolute minimum the edge detection threshold can go.\n"
		"Prevents edge detection in extremely low contrast areas, saving performance.\n"
		"Recommended values [.001 - .025]";
> = .015;

uniform int _MaxSearchSteps < __UNIFORM_DRAG_INT1
	ui_category = "Blending weight calculation";
	ui_label = "Max search steps";
	ui_min = 0;
	ui_max = 128;
	ui_step = 1;
	ui_category_closed = true;
	ui_tooltip =
		"Maximum number of steps to search for edges in a horizontal/vertical fashion.\n"
		"Higher values detect larger patterns but are more computationally expensive.\n"
		"Recommended values [4 - 32]";
> = 32;

uniform int _MaxSearchStepsDiag < __UNIFORM_DRAG_INT1
	ui_category = "Blending weight calculation";
	ui_label = "Max diagonal search steps";
	ui_min = 0;
	ui_max = 64;
	ui_step = 1;
	ui_tooltip =
		"Maximum number of steps to search for edges in diagonal directions.\n"
		"Higher values detect larger patterns but are more computationally expensive.\n"
		"Recommended values [8 - 24]";
> = 19;

uniform int _CornerRounding < __UNIFORM_DRAG_INT1
	ui_category = "Blending weight calculation";
	ui_label = "Corner rounding";
	ui_min = 0;
	ui_max = 100;
	ui_step = 1;
	ui_tooltip =
		"Specifies how much sharp corners will be rounded.\n"
		"Higher values create smoother corners but more blur.\n"
		"Recommended values [0 - 25]";
> = 0;

uniform bool _SmoothingEnabled <
	ui_category = "Smoothing";
	ui_label = "Enable smoothing";
	ui_category_toggle = true;
	ui_category_closed = true;
	ui_tooltip =
		"This increases the smoothness of anti-aliased edges beyond what\n"
		"normal SMAA can do, provided PSMAA_SMOOTHING_USE_COLOR_SPACE=0,\n"
		"and helps reduce residual aliasing artifacts after the main AA passes.\n"
		"Uses a modified version of an algorithm made by Lord Bean for an \n"
		"experimental version of his TSMAA shader.\n"
		"  - enable with PSMAA_SMOOTHING_USE_COLOR_SPACE=1 to\n"
		"      only eliminate residual aliasing.\n"
		"  - enable with PSMAA_SMOOTHING_USE_COLOR_SPACE=0 to\n"
		"      sacrifice clarity for smoothness.\n"
		"  - disable for best performance.\n";
> = true;

// uniform bool _OldSmoothingEnabled < // TODO: remove?
// 	ui_category = "Smoothing";
// 	ui_label = "Use old smoothing";
// 	ui_tooltip =
//	  "Use the older smoothing algorithm instead of the newer one.\n"
//    "Old smoothing may be useful for compatibility or specific visual preferences.";
// > = false;

uniform bool _SmoothingDeltaWeightDebug <
	ui_category = "Smoothing";
	ui_label = "Smoothing delta weight debug";
	ui_tooltip =
		"Enable debugging for smoothing delta weights.\n"
		"This is a development tool and should normally be disabled.";
> = false;

uniform float2 _SmoothingDeltaWeights <
	ui_category = "Smoothing";
	ui_label = "Smoothing delta weights";
	ui_type = "slider";
	ui_min = 0f;
	ui_max = .75;
	ui_step = 0.01f;
	ui_tooltip =
		"These values determine how large the largest edges of a pixel must be\n"
		"for smoothing to apply the maximum number of search steps 'n' to them.\n"
		"The left represents the threshold where the smallest n is used,\n"
		"the right the threshold above which the largest n is used.\n"
		"Lower values means the algorithm smoothes small differences more agressively,\n"
		"which may cause blurriness and worse performance if you overdo it,\n"
		"while higher values give better performance but may miss some edges.";
> = float2(.1, .5);

uniform float _SmoothingDeltaWeightDynamicThreshold <
	ui_category = "Smoothing";
	ui_label = "Dynamic threshold";
	ui_type = "slider";
	ui_min = 0f;
	ui_max = 1f;
	ui_step = 0.01f;
	ui_tooltip =
		"A percentage which controls how much the delta weights are scaled with\n"
		"local luminosity. The higher the value, the more the weights are decreased\n"
		"in darker areas. Higher values make smoothing more accurate and sensitive\n"
		"in darker areas, but may cause worse performance and blur.\n"
		"Recommended values [.4 - .9]";
> = .65;

uniform float2 _SmoothingThresholds <
	ui_category = "Smoothing";
	ui_label = "Smoothing thresholds";
	ui_type = "slider";
	ui_min = .01;
	ui_max = .25;
	ui_step = .001;
	ui_tooltip =
		"Contrast thresholds above which the smoothing algorithm activates.\n"
		"Controls when smoothing is applied based on pixel brightness.\n"
		"The left value is for darker areas, the right is for lighter areas.\n"
		"Recommended values [(.01, .05) - (.25, .15)]";
> = float2(.015, .09);

uniform float _SmoothingThresholdDepthGrowthStart <
	ui_category = "Smoothing";
	ui_label = "Smoothing threshold depth growth start";
	ui_type = "slider";
	ui_min = 0f;
	ui_max = 1f;
	ui_step = .01;
	ui_tooltip =
		"Distance where the smoothing threshold starts growing with depth.\n"
		"Distance meaning the value of the depth from the depth buffer, where\n"
		"0 = no distance and 1 = max distance.\n"
		"Lower values help prevent blur, especially closer up,\n"
		"but may cause false negatives.\n"
		"Recommended values [.25 - .5]";
> = .35;

uniform float _SmoothingThresholdDepthGrowthFactor <
	ui_category = "Smoothing";
	ui_label = "Smoothing threshold depth growth factor";
	ui_type = "slider";
	ui_min = 1f;
	ui_max = 4f;
	ui_step = .1;
	ui_tooltip =
		"Multiplier for how much the smoothing thresholds grow with distance.\n"
		"Distance meaning the value of the depth from the depth buffer, where\n"
		"0 = no distance and 1 = max distance.\n"
		"Higher values help prevent blur, especially further away,\n"
		"but may cause false negatives.\n"
		"Recommended values [1.5 - 4]";
> = 2.5;

uniform bool _SharpeningEnabled <
	ui_category = "Sharpening";
	ui_label = "Enable sharpening";
	ui_category_toggle = true;
	ui_category_closed = true;
	ui_tooltip = "Enable AMD FX CAS.\n";
> = false;

uniform float _SharpeningCompensationStrength <
	ui_category = "Sharpening";
	ui_label = "Blur compensation";
	ui_type = "slider";
	ui_min = 0f;
	ui_max = 2f;
	ui_step = .1;
	ui_tooltip =
		"Increases sharpening strength the more a pixel has been changed\n"
		"by the preceding passes. Thus 'compensating' for any blur.\n"
		"This works even when sharpening blending strength is zero.\n"
		"Recommended values [.8 - 1.5]";
> = 1.5;

uniform float _SharpeningCompensationCutoff <
	ui_category = "Sharpening";
	ui_label = "Compensation cutoff";
	ui_type = "slider";
	ui_min = .05;
	ui_max = 1f;
	ui_step = .01;
	ui_tooltip =
		"The max pixel change, after which compensation strength stops growing.\n"
		"Lower values make sure that sharpening doesn't bring back AA artifacts.\n"
		"Recommended values [.05 - .3]";
> = .15;

uniform float _SharpeningEdgeBias <
	ui_category = "Sharpening";
	ui_label = "Edge bias";
	ui_type = "slider";
	ui_min = -4f;
	ui_max = 0f;
	ui_step = .1;
	ui_tooltip =
		"Bias applied to the sharpening strength of high-contrast pixels.\n"
		"Lower values mean less sharpening at pixels which are already sharp.\n"
		"Recommended values [-2.0 - -0.5]";
> = -1.5f;

uniform float _SharpeningSharpness <
	ui_category = "Sharpening";
	ui_label = "Sharpness";
	ui_type = "slider";
	ui_min = 0f;
	ui_max = 1f;
	ui_step = .01;
	ui_tooltip =
		"The 'sharpness' parameter for AMD FX CAS sharpening.\n"
		"Equivalent to the 'Contrast Adaptation' parameter in ReShade's\n"
		"standard CAS implementation.\n"
		"Higher values make the sharpening effect stronger.\n"
		"Recommended values [0 - .5]";
> = 0f;

uniform float _SharpeningBlendingStrength <
	ui_category = "Sharpening";
	ui_label = "Blending strength";
	ui_type = "slider";
	ui_min = 0f;
	ui_max = 1f;
	ui_step = .01;
	ui_tooltip =
		"The percentage by which the sharpened result is applied to the output.\n"
		"Basically, higher values = stronger sharpening appears on-screen.";
> = .7;

uniform bool _SharpeningDebug <
	ui_category = "Sharpening";
	ui_label = "Sharpening debug";
	ui_type = "radio";
	ui_tooltip =
		"Shows the strength of the sharpening.\n"
		"Lighter pixels = more sharpening.\n"
		"The effect of the \'Sharpness\' value is not included here.\n";
		// TODO: check if colors are correct!
> = false;


// Debug output options START

#ifndef SHOW_DEBUG
	#define SHOW_DEBUG 0
#endif
// preprocessor variable dedicated to debug library, turns it on or off
#define SHARED_DEBUG__ACTIVE_ SHOW_DEBUG

#if SHOW_DEBUG

uniform int _Debug <
	ui_category = "Debug";
	ui_type = "combo";
	ui_label = "Debug output";
	ui_tooltip = "Outputs the contents of various internal buffers for debugging purposes.";
	ui_items = "None\0Max Local Luma\0Luma\0Filter strength weights\0Filtered image only\0Deltas\0Edges\0";
> = 0;

#endif

#include "./reshade-shared/libraries/debug.fxh"

// Debug output options END


uniform int _MacroHelpText<
    ui_type = "radio";
    ui_category = "Preprocessor variable explanation";
    ui_label = "    ";
		ui_category_closed = false;
    ui_text =
			"PSMAA_SMOOTHING_USE_COLOR_SPACE: setting this to 1 will get rid of\n"
			"blur and 'darkening' the smoothing pass normally causes, but will\n"
			"mostly eliminate its ability to make jaggies which have already been\n"
			"anti-aliased by the SMAA passes even smoother.\n"
			"If this is left off (0), using sharpening is recommended.\n"
			"\n"
			"PSMAA_USE_SIMPLIFIED_DELTA_CALCULATION: debugging feature. set this to 1\n"
			"to use a faster but less precise form of delta calculation.\n"
			"\n"
			"SHOW_DEBUG: set this to 1 to enable debug output options in the UI.";
>;

#ifndef PSMAA_USE_SIMPLIFIED_DELTA_CALCULATION
	#define PSMAA_USE_SIMPLIFIED_DELTA_CALCULATION 0
#endif

#include "ReShade.fxh"

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
	#define SMAA_PREDICATION_SCALE 2f
	#define SMAA_PREDICATION_STRENGTH .4
	#define SMAA_REPROJECTION 0
	#define SMAA_REPROJECTION_WEIGHT_SCALE 30.0
#endif

#include "./PSMAA.fxh"

texture colorInputTex : COLOR;
sampler colorGammaSampler
{
	Texture = colorInputTex;
	MipFilter = POINT;
};
sampler colorLinearSampler
{
	Texture = colorInputTex;
	MipFilter = Point;
	SRGBTexture = true;
};

texture maxLocalLumaTex
{
	Width = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = R8;
};
sampler maxLocalLumaSampler
{
	Texture = maxLocalLumaTex;
};

texture originalLumaTex
{
	Width = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = R8;
};
sampler originalLumaSampler
{
	Texture = originalLumaTex;
};

texture filterStrengthTex
{
	Width = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = RG8;
};
sampler filterStrengthSampler
{
	Texture = filterStrengthTex;
};

texture deltaTex
{
	Width = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = RG8;
};
sampler deltaSampler
{
	Texture = deltaTex;
};

texture edgesTex
{
	Width = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = RG8;
};
sampler edgesSampler
{
	Texture = edgesTex;
};

texture weightTex
{
	Width = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = RGBA8;
};
sampler weightSampler
{
	Texture = weightTex;
};

texture areaTex < source = "AreaTex.png"; >
{
	Width = 160;
	Height = 560;
	Format = RG8;
};
sampler areaSampler
{
	Texture = areaTex;
};

texture searchTex < source = "SearchTex.png"; >
{
	Width = 64;
	Height = 16;
	Format = R8;
};
sampler searchSampler
{
	Texture = searchTex;
	MipFilter = Point;
	MinFilter = Point;
	MagFilter = Point;
};

void PSMAAPreProcessingPSWrapper(
		float4 position : SV_POSITION,
		float2 texcoord : TEXCOORD0,
		out float maxLocalLuma : SV_TARGET0,
		out float originalLuma : SV_TARGET1,
		out float2 filteringStrength : SV_TARGET2)
{
	// if (_ShowOldPreProcessing)
	// {
	// 	PSMAA::Pass::PreProcessingPSOld(texcoord, colorGammaSampler, maxLocalLuma, originalLuma, filteringStrength);
	// 	return;
	// }

	PSMAA::Pass::PreProcessingPS(texcoord, colorGammaSampler, maxLocalLuma, originalLuma, filteringStrength);
}

void PSMAAFilteringPSWrapper(
		float4 position : SV_Position,
		float2 texcoord : TEXCOORD0,
		out float4 color : SV_Target)
{
	// if (_ShowOldPreProcessing)
	// {
	// 	PSMAAOld::Pass::FilteringPS(texcoord, colorLinearSampler, filterStrengthSampler, color);
	// 	return;
	// }
	PSMAA::Pass::FilteringPS(texcoord, colorLinearSampler, filterStrengthSampler, color);
}

// TODO: consider trying to calculate this in the PS instead.
void PSMAADeltaCalulationVSWrapper(
		in uint id : SV_VertexID,
		out float4 position : SV_Position,
		out float2 texcoord : TEXCOORD0,
		out float4 offset[1] : TEXCOORD1)
{
	PostProcessVS(id, position, texcoord);
	PSMAA::Pass::DeltaCalculationVS(texcoord, offset);
}

void PSMAADeltaCalulationPSWrapper(
		float4 position : SV_Position,
		float2 texcoord : TEXCOORD0,
		float4 offset[1] : TEXCOORD1,
		out float2 deltas : SV_Target0)
{
	PSMAA::Pass::DeltaCalculationPS(texcoord, offset, colorGammaSampler, deltas);
}

void PSMAAEdgeDetectionVSWrapper(
		in uint id : SV_VertexID,
		out float4 position : SV_Position,
		out float2 texcoord : TEXCOORD0,
		out float4 offset[2] : TEXCOORD1)
{
	PostProcessVS(id, position, texcoord);
	PSMAA::Pass::EdgeDetectionVS(texcoord, offset);
}

void PSMAAEdgeDetectionPSWrapper(
		float4 position : SV_Position,
		float2 texcoord : TEXCOORD0,
		float4 offset[2] : TEXCOORD1,
		out float2 edges : SV_Target)
{
	PSMAA::Pass::EdgeDetectionPS(texcoord, offset, deltaSampler, maxLocalLumaSampler, edges);
}

void SMAABlendingWeightCalculationVSWrapper(
		in uint id : SV_VertexID,
		out float4 position : SV_Position,
		out float2 texcoord : TEXCOORD0,
		out float2 pixcoord : TEXCOORD1,
		out float4 offset[3] : TEXCOORD2)
{
	PostProcessVS(id, position, texcoord);
	SMAABlendingWeightCalculationVS(texcoord, pixcoord, offset);
}

float4 SMAABlendingWeightCalculationPSWrapper(
		float4 position : SV_Position,
		float2 texcoord : TEXCOORD0,
		float2 pixcoord : TEXCOORD1,
		float4 offset[3] : TEXCOORD2) : SV_Target
{
	return SMAABlendingWeightCalculationPS(texcoord, pixcoord, offset, edgesSampler, areaSampler, searchSampler, 0.0);
}

void SMAANeighborhoodBlendingVSWrapper(
		in uint id : SV_VertexID,
		out float4 position : SV_Position,
		out float2 texcoord : TEXCOORD0,
		out float4 offset : TEXCOORD1)
{
	PostProcessVS(id, position, texcoord);
	SMAANeighborhoodBlendingVS(texcoord, offset);
}

void PSMAABlendingPSWrapper(
		float4 position : SV_Position,
		float2 texcoord : TEXCOORD0,
		float4 offset : TEXCOORD1,
		out float4 color : SV_Target)
{

#if SHOW_DEBUG

	if (_Debug == 0)
	{
		// if (_UseOldBlending)
		// {
		// 	color = SMAANeighborhoodBlendingPS(texcoord, offset, colorLinearSampler, weightSampler).rgba;
		// 	return;
		// }
		PSMAA::Pass::BlendingPS(texcoord, offset, colorLinearSampler, weightSampler, filterStrengthSampler, color);
	}
	else if (_Debug == 4)
	{
		color = tex2D(colorLinearSampler, texcoord);
	}
	else
	{
		discard;
	}

#else
	// if (_UseOldBlending)
	// {
	// 	color = SMAANeighborhoodBlendingPS(texcoord, offset, colorLinearSampler, weightSampler).rgba;
	// 	return;
	// }
	PSMAA::Pass::BlendingPS(texcoord, offset, colorLinearSampler, weightSampler, filterStrengthSampler, color);

#endif
}

void SmoothingPSWrapper(
		float4 position : SV_Position,
		float2 texcoord : TEXCOORD0,
		float4 offset : TEXCOORD1,
		out float3 color : SV_Target)
{
	if (!_SmoothingEnabled)
		discard;
// if (_OldSmoothingEnabled){
// 	PSMAAOld::Pass::SmoothingPS(texcoord, offset, deltaSampler, weightSampler, colorLinearSampler, maxLocalLumaSampler, color);
// 	return;
// }
#if PSMAA_SMOOTHING_USE_COLOR_SPACE
	PSMAA::Pass::SmoothingPS(texcoord, offset, deltaSampler, weightSampler, colorLinearSampler, maxLocalLumaSampler, color);
#else
	PSMAA::Pass::SmoothingPS(texcoord, offset, deltaSampler, weightSampler, colorGammaSampler, maxLocalLumaSampler, color);
#endif
}

void CASPSWrapper(
		float4 position : SV_Position,
		float2 texcoord : TEXCOORD0,
		out float3 color : SV_Target)
{
	if (!_SharpeningEnabled)
		discard;
	// if (_UseOldCas)
	// {
	// 	CASOld::CASPS(texcoord, colorLinearSampler, color);
	// 	return;
	// }
	// CAS::CASPS(texcoord, colorLinearSampler, color);
	PSMAA::Pass::SharpeningPS(texcoord, originalLumaSampler, colorGammaSampler, deltaSampler, colorLinearSampler, color);
}

#if SHOW_DEBUG
void PSMAADebugPS(
		float4 position : SV_Position,
		float2 texcoord : TEXCOORD0,
		float4 offset : TEXCOORD1,
		out float4 color : SV_Target)
{
	if (_Debug == 1)
	{
		color = tex2D(maxLocalLumaSampler, texcoord).rrrr;
	}
	else if (_Debug == 2)
	{
		color = tex2D(originalLumaSampler, texcoord).rrrr;
	}
	else if (_Debug == 3)
	{
		color = float4(tex2D(filterStrengthSampler, texcoord).rg, 0f, 0f);
	}
	else if (_Debug == 5)
	{
		color = tex2D(deltaSampler, texcoord).rgba;
	}
	else if (_Debug == 6)
	{
		color = tex2D(edgesSampler, texcoord).rgba;
	}
	else
	{
		color = tex2D(colorLinearSampler, texcoord);
	}

	color = Debug::applyDebugOptions(color);
}
#endif

technique PSMAA
{
	pass PreProcessing
	{
		VertexShader = PostProcessVS;
		PixelShader = PSMAAPreProcessingPSWrapper;
		RenderTarget0 = maxLocalLumaTex;
		RenderTarget1 = originalLumaTex;
		RenderTarget2 = filterStrengthTex;
		// ClearRenderTargets = true;
	}
	pass Filtering
	{
		VertexShader = PostProcessVS;
		PixelShader = PSMAAFilteringPSWrapper;
		SRGBWriteEnable = true;
	}
	pass DeltaCalculation
	{
		VertexShader = PSMAADeltaCalulationVSWrapper;
		PixelShader = PSMAADeltaCalulationPSWrapper;
		RenderTarget = deltaTex;
		// TODO: test if these are necessary!
		// Especially the stencil stuff
		// https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md#techniques
		ClearRenderTargets = true; // TODO: test if this is needed
		// StencilEnable = true;
		// StencilPass = REPLACE;
		// StencilRef = 1;
	}
	pass EdgeDetection
	{
		VertexShader = PSMAAEdgeDetectionVSWrapper;
		PixelShader = PSMAAEdgeDetectionPSWrapper;
		RenderTarget = edgesTex;
		ClearRenderTargets = true;
		StencilEnable = true;
		StencilPass = REPLACE;
		StencilRef = 1;
	}
	pass BlendWeightCalculationPass
	{
		VertexShader = SMAABlendingWeightCalculationVSWrapper;
		PixelShader = SMAABlendingWeightCalculationPSWrapper;
		RenderTarget = weightTex;
		ClearRenderTargets = true;
		StencilEnable = true;
		StencilPass = KEEP;
		StencilFunc = EQUAL;
		StencilRef = 1;
	}
	pass Blending
	{
		// TODO: consider renaming this VSWrapper to something more generic
		// alternatively, Consider giving this pass it's own VS
		VertexShader = SMAANeighborhoodBlendingVSWrapper;
		PixelShader = PSMAABlendingPSWrapper;
		SRGBWriteEnable = true;
	}
	pass Smoothing
	{
		VertexShader = SMAANeighborhoodBlendingVSWrapper;
		PixelShader = SmoothingPSWrapper;
#if PSMAA_SMOOTHING_USE_COLOR_SPACE
		SRGBWriteEnable = true;
#endif
	}
	pass Sharpening
	{
		VertexShader = PostProcessVS;
		PixelShader = CASPSWrapper;
		SRGBWriteEnable = true;
	}
#if SHOW_DEBUG
	pass Debug
	{
		VertexShader = SMAANeighborhoodBlendingVSWrapper;
		PixelShader = PSMAADebugPS;
		SRGBWriteEnable = true;
	}
#endif
}
