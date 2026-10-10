#ifndef _DEBUG_FXH
#define _DEBUG_FXH

#include "functions.fxh"

// Place these preprocessor vars in file
// #define SHARED_DEBUG__ACTIVE_ 1

#define SHARED_DEBUG__NEAR_ZERO 0.0001f // TODO: consider moving this into the #if SHARED_DEBUG__ACTIVE_ block
#define SHARED_DEBUG__BRANCH [branch]

#if SHARED_DEBUG__ACTIVE_ == 1
namespace Debug
{
  /** float2 windowThresholds: the range of values to visualize
  *  - x: window floor
  *  - y: window ceiling
  * bool singleChannelAsWhite: assumes that only one channel is active (sum of channelsToDisplay <= 1.0). Shows channel as white.
  */
  struct DebugOptions
  {
    float4 channelsToDisplay;
    bool outputChannelsAsSum;
    uint alphaOutputChannel;
    bool singleChannelAsWhite;
    bool applyWindow;
    float2 windowThresholds;
    bool highlightOutOfRange;
    float4 lowColor;
    float4 highColor;
  };

  namespace UIControls
  {
    uniform float4 _DebugChannelsToDisplay <
        ui_category = "Debug";
        ui_label = "Channels To Display (RGBA)";
        ui_type = "color";
        ui_tooltip = "Set which color channels to display (RGBA).";
    > = float4(1, 1, 1, 1);

    uniform bool _DebugOutputChannelsAsSum <
        ui_category = "Debug";
        ui_label = "Output Channels As Sum";
        ui_tooltip = "If enabled, outputs the sum of the selected channels.";
    > = false;

    uniform bool _DebugSingleChannelAsWhite <
        ui_category = "Debug";
        ui_label = "Single Channel As White";
        ui_tooltip = "If enabled, displays a single active channel as white.";
    > = false;

    uniform int _DebugAlphaOutputChannel <
        ui_category = "Debug";
        ui_label = "Alpha Output Channel";
        ui_type = "combo";
        ui_items = "Red\0Green\0Blue\0Alpha\0";
        ui_tooltip = "Select which channel receives the alpha value.";
    > = 3;

    uniform int _Divider1 <
      ui_category = "Debug";
      ui_type = "radio";
      ui_label = " ";
    >;

    uniform bool _ApplyWindow <
        ui_category = "Debug";
        ui_label = "Apply Window";
    > = false;

    uniform float2 _DebugWindowThresholds <
        ui_category = "Debug";
        ui_label = "Window Thresholds";
        ui_type = "slider";
        ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
        ui_tooltip = "Set the window floor (x) and ceiling (y) for value visualization.";
    > = float2(0.0, 1.0);

    uniform bool _DebugHighlightOutOfRange <
        ui_category = "Debug";
        ui_label = "Highlight Out Of Range";
        ui_tooltip = "If enabled, highlights values outside the window thresholds.";
    > = false;

    uniform float3 _DebugLowColor <
        ui_category = "Debug";
        ui_label = "Low Color";
        ui_type = "color";
        ui_tooltip = "Color to use for values below the window floor.";
    > = float3(0, 1.0f, 1.0f);

    uniform float3 _DebugHighColor <
        ui_category = "Debug";
        ui_label = "High Color";
        ui_type = "color";
        ui_tooltip = "Color to use for values above the window ceiling.";
    > = float3(1.0f, 1.0f, 0);

    DebugOptions bootstrapDebugOptions()
    {
      DebugOptions opts;
      opts.channelsToDisplay = _DebugChannelsToDisplay;
      opts.outputChannelsAsSum = _DebugOutputChannelsAsSum;
      opts.alphaOutputChannel = _DebugAlphaOutputChannel;
      opts.singleChannelAsWhite = _DebugSingleChannelAsWhite;
      opts.applyWindow = _ApplyWindow;
      opts.windowThresholds = _DebugWindowThresholds;
      opts.highlightOutOfRange = _DebugHighlightOutOfRange;
      opts.lowColor = float4(_DebugLowColor, 1.0f);
      opts.highColor = float4(_DebugHighColor, 1.0f);
      return opts;
    }
  }

  float4 routeAlpha(float4 color, uint channel)
  {
    SHARED_DEBUG__BRANCH
    switch (channel)
    {
    case 0:
      color.r = color.a;
      break;
    case 1:
      color.g = color.a;
      break;
    case 2:
      color.b = color.a;
      break;
      // case 3: color.a = color.a; // no-op
    }
    return color;
  }

  float4 colorAsSum(float4 color, float divisor)
  {
    // sum all channels together. divide by 4 to prevent overflow
    float packedSum = Functions::sum(color) / divisor;
    return float2(packedSum, 0.0f).xyyy;
  }

  float4 singleChannelToWhite(float4 color)
  {
    float singleValue = saturate(Functions::sum(color));
    return float(singleValue).xxxx;
  }

  float4 applyWindow(float4 color, float2 window)
  {
    return smoothstep(window.x, window.y, color);
  }

  float4 applyWindow(float4 color, float2 window, float4 lowColor, float4 highColor)
  {
    if (any(color < window.x))
      return lowColor;
    if (any(color > window.y))
      return highColor;

    return applyWindow(color, window);
  }

  float4 applyDebugOptions(float4 color, DebugOptions opts)
  {
    // change strength of each channel, or even turn them off
    color *= opts.channelsToDisplay;
    float activeChannelsWeight = Functions::sum(opts.channelsToDisplay);

    if (activeChannelsWeight == 0.0f)
    {
      // no channels are active, early return
      // to prevent division by zero and avoid performance hit
      return color;
    }

    if (opts.outputChannelsAsSum)
    {
      color = colorAsSum(color, activeChannelsWeight);
    }
    else
    {
      color = routeAlpha(color, opts.alphaOutputChannel);
    }

    if (opts.singleChannelAsWhite)
    {
      float nrOfActiveChannels = Functions::sum(step(SHARED_DEBUG__NEAR_ZERO, opts.channelsToDisplay));
      if (nrOfActiveChannels == 1.0f || opts.outputChannelsAsSum)
      {
        color = singleChannelToWhite(color);
      }
    }
    if (opts.applyWindow)
    {
      if (opts.highlightOutOfRange)
      {
        color = applyWindow(color, opts.windowThresholds, opts.lowColor, opts.highColor);
      }
      else
      {
        color = applyWindow(color, opts.windowThresholds);
      }
    }

    return color;
  }

  float4 applyDebugOptions(float4 color)
  {
    DebugOptions opts = UIControls::bootstrapDebugOptions();

    return applyDebugOptions(color, opts);
  }
}
#endif

#endif // DEBUG_FXH include guard
