#ifndef _RESHADE_SHARED_FUNCTIONS_FHX
#define _RESHADE_SHARED_FUNCTIONS_FHX

#include "macros.fxh"
/**
* Commonly used convenience functions.
*/
namespace Functions
{
  namespace Shared
  {
    float dotArithmetic(float2 vec, float weight)
    {
      return dot(vec, float2(weight, weight));
    }
    float dotArithmetic(float3 vec, float weight)
    {
      return dot(vec, float3(weight, weight, weight));
    }
    float dotArithmetic(float4 vec, float weight)
    {
      return dot(vec, float4(weight, weight, weight, weight));
    }
  }

  #define GEN_OVERLOADS_UP_TO_16_PARAMS(type, func_name) \
    GEN_FUNC_3_PARAMS(type, func_name) \
    GEN_FUNC_4_PARAMS(type, func_name) \
    GEN_FUNC_5_PARAMS(type, func_name) \
    GEN_FUNC_6_PARAMS(type, func_name) \
    GEN_FUNC_7_PARAMS(type, func_name) \
    GEN_FUNC_8_PARAMS(type, func_name) \
    GEN_FUNC_9_PARAMS(type, func_name) \
    GEN_FUNC_10_PARAMS(type, func_name) \
    GEN_FUNC_11_PARAMS(type, func_name) \
    GEN_FUNC_12_PARAMS(type, func_name) \
    GEN_FUNC_13_PARAMS(type, func_name) \
    GEN_FUNC_14_PARAMS(type, func_name) \
    GEN_FUNC_15_PARAMS(type, func_name) \
    GEN_FUNC_16_PARAMS(type, func_name)

  GEN_OVERLOADS_UP_TO_16_PARAMS(float, max)
  GEN_OVERLOADS_UP_TO_16_PARAMS(float2, max)
  GEN_OVERLOADS_UP_TO_16_PARAMS(float3, max)
  GEN_OVERLOADS_UP_TO_16_PARAMS(float4, max)
  GEN_OVERLOADS_UP_TO_16_PARAMS(float, min)
  GEN_OVERLOADS_UP_TO_16_PARAMS(float2, min)
  GEN_OVERLOADS_UP_TO_16_PARAMS(float3, min)
  GEN_OVERLOADS_UP_TO_16_PARAMS(float4, min)

  float max(float2 rg)
  {
    return max(rg.r, rg.g);
  }
  float max(float3 rgb)
  {
    return max(max(rgb.rg), rgb.b);
  }
  float max(float4 rgba)
  {
    return max(max(rgba.rg), max(rgba.ba));
  }

  float min(float2 rg)
  {
    return min(rg.r, rg.g);
  }
  float min(float3 rgb)
  {
    return min(min(rgb.rg), rgb.b);
  }
  float min(float4 rgba)
  {
    return min(min(rgba.rg), min(rgba.ba));
  }

  float sum(float2 vec)
  {
    return Shared::dotArithmetic(vec, 1.0f);
  }
  float sum(float3 vec)
  {
    return Shared::dotArithmetic(vec, 1.0f);
  }
  float sum(float4 vec)
  {
    return Shared::dotArithmetic(vec, 1.0f);
  }

  float avg(float2 vec)
  {
    return Shared::dotArithmetic(vec, 0.5);
  }
  float avg(float3 vec)
  {
    const float mod = 1.0f / 3.0f;
    return Shared::dotArithmetic(vec, mod);
  }
  float avg(float4 vec)
  {
    return Shared::dotArithmetic(vec, 0.25);
  }
  float avg(float x, float y, float z, float w)
  {
    return avg(float4(x,y,z,w));
  }

  /**
   * @SCALE_LINEAR
   * Meant for turning linear values super-linear: Makes it's input bigger in such a way that lower values become
   * proportionally bigger than higher values. Output never exceeds 1.0;
   *
   * @param `val` input to be scaled
   * @return output val. Amplified in a non-linear fashion.
   */
  float sineScale(float val){
    const float piHalf = 1.5707;
    return val = sin(val * piHalf);
  }

  /**
   * @param input some factor with a value of threshold floor 0.0 - 1.0
   */
  float clampScale(float input, float modifier, float floor, float ceil)
  {
    return clamp(input * modifier, floor, ceil);
  }

}
#endif // RESHADE_SHARED_FUNCTIONS_FHX include guard
