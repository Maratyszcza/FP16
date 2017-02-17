#pragma once
#ifndef FP16_PSIMD_H
#define FP16_PSIMD_H

#if defined(__cplusplus) && (__cplusplus >= 201103L)
	#include <cstdint>
	#include <cmath>
#elif !defined(__OPENCL_VERSION__)
	#include <stdint.h>
	#include <math.h>
#endif

#include <psimd.h>


PSIMD_INTRINSIC psimd_f32 fp16_ieee_to_fp32_psimd(psimd_u16 half) {
	const psimd_u32 word = (psimd_u32) psimd_unpacklo_u16(psimd_zero_u16(), half);

	const psimd_u32 sign = word & psimd_splat_u32(UINT32_C(0x80000000));
	const psimd_u32 shr3_nonsign = (word + word) >> psimd_splat_u32(4);

	const psimd_u32 exp_offset = psimd_splat_u32(UINT32_C(0x70000000));
	const psimd_f32 exp_scale = psimd_splat_f32(0x1.0p-112f);
	const psimd_f32 norm_nonsign = (psimd_f32) (shr3_nonsign + exp_offset) * exp_scale;

	const psimd_u16 magic_mask = psimd_splat_u16(UINT16_C(0x3E80));
	const psimd_f32 magic_bias = psimd_splat_f32(0.25f);
	const psimd_f32 denorm_nonsign = (psimd_f32) psimd_unpacklo_u16(half + half, magic_mask) - magic_bias;

	const psimd_s32 denorm_cutoff = psimd_splat_s32(INT32_C(0x00800000));
	const psimd_s32 denorm_mask = (psimd_s32) shr3_nonsign < denorm_cutoff;
	return (psimd_f32) (sign | (psimd_s32) psimd_blend_f32(denorm_mask, denorm_nonsign, norm_nonsign));
}

PSIMD_INTRINSIC psimd_f32 fp16_alt_to_fp32_psimd(psimd_u16 half) {
	const psimd_u32 word = (psimd_u32) psimd_unpacklo_u16(psimd_zero_u16(), half);

	const psimd_u32 sign = word & psimd_splat_u32(UINT32_C(0x80000000));
	const psimd_u32 shr3_nonsign = (word + word) >> psimd_splat_u32(4);

	const psimd_u32 exp_offset = psimd_splat_u32(UINT32_C(0x38000000));
	const psimd_f32 norm_nonsign = (psimd_f32) (shr3_nonsign + exp_offset);

	const psimd_u16 magic_mask = psimd_splat_u16(UINT16_C(0x3E80));
	const psimd_f32 magic_bias = psimd_splat_f32(0.25f);
	const psimd_f32 denorm_nonsign = (psimd_f32) psimd_unpacklo_u16(half + half, magic_mask) - magic_bias;

	const psimd_s32 denorm_cutoff = psimd_splat_s32(INT32_C(0x00800000));
	const psimd_s32 denorm_mask = (psimd_s32) shr3_nonsign < denorm_cutoff;
	return (psimd_f32) (sign | (psimd_s32) psimd_blend_f32(denorm_mask, denorm_nonsign, norm_nonsign));
}

#endif /* FP16_PSIMD_H */
