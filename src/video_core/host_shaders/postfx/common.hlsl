// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
struct Constants { float4 metrics; float4 target; uint conversion; };
[[vk::push_constant]] Constants params;
[[vk::binding(0)]] Texture2D<float4> tex0;
[[vk::binding(1)]] Texture2D<float4> tex1;
[[vk::binding(2)]] Texture2D<float4> tex2;
[[vk::binding(3)]] Texture2D<float4> tex3;
[[vk::binding(4)]] Texture2D<float4> tex4;
[[vk::binding(5)]] Texture2D<float4> tex5;
[[vk::binding(6)]] Texture2D<float4> tex6;
[[vk::binding(7)]] Texture2D<float4> tex7;
[[vk::binding(8)]] SamplerState point_sampler;
[[vk::binding(9)]] SamplerState linear_sampler;
[[vk::binding(10)]] RWTexture2D<float4> output0;
[[vk::binding(11)]] RWTexture2D<float4> output1;
[[vk::binding(12)]] RWTexture2D<float4> output2;
#define BUFFER_WIDTH params.metrics.z
#define BUFFER_HEIGHT params.metrics.w
#define BUFFER_RCP_WIDTH params.metrics.x
#define BUFFER_RCP_HEIGHT params.metrics.y
#define BUFFER_PIXEL_SIZE params.metrics.xy
float3 Decode(float3 c) { return select(c <= 0.04045, c / 12.92, pow(max((c + 0.055) / 1.055, 0.0), 2.4)); }
float3 Encode(float3 c) { return select(c <= 0.0031308, c * 12.92, 1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055); }
struct Tex { Texture2D<float4> image; bool is_linear; bool srgb; };
Tex MakeTex(Texture2D<float4> image, bool is_linear = true, bool srgb = false) { Tex t = {image, is_linear, srgb}; return t; }
float4 tex2D(Tex t, float2 uv) { float4 c = t.is_linear ? t.image.SampleLevel(linear_sampler, uv, 0) : t.image.SampleLevel(point_sampler, uv, 0); if (t.srgb) c.rgb = Decode(c.rgb); return c; }
float4 tex2Dlod(Tex t, float4 uv) { return tex2D(t, uv.xy); }
float4 tex2Doffset(Tex t, float2 uv, float2 off) { return tex2D(t, uv + off * params.metrics.xy); }
float4 tex2Dlodoffset(Tex t, float4 uv, float2 off) { return tex2D(t, uv.xy + off * params.metrics.xy); }
float4 tex2Dgather(Tex t, float2 uv, int channel) {
 if (channel == 0) return t.image.GatherRed(point_sampler, uv);
 if (channel == 1) return t.image.GatherGreen(point_sampler, uv);
 if (channel == 2) return t.image.GatherBlue(point_sampler, uv);
 return t.image.GatherAlpha(point_sampler, uv);
}
#define sampler Tex
#define sampler2D Tex
