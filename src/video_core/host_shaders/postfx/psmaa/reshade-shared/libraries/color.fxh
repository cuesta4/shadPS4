
#ifndef _COLOR_FXH
#define _COLOR_FXH

namespace Color
{
  // TODO: try changing into pre-processor val
  // static const float3 LUMA_WEIGHTS = float3(0.2126, 0.7152, 0.0722);
  static const float3 LUMA_WEIGHTS = float3(0.299, 0.587, 0.114);

  float luma(float3 rgb) {
    // const float3 LUMA_WEIGHTS = float3(0.2126, 0.7152, 0.0722);
    return dot(rgb, LUMA_WEIGHTS);
  }
}

#endif // COLOR_FXH include guard
