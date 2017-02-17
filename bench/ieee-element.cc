#include <benchmark/benchmark.h>

#include <fp16.h>

#if (defined(__i386__) || defined(__x86_64__)) && defined(__F16C__)
	#include <immintrin.h>
#endif


static void fp16_ieee_to_fp32_bits(benchmark::State& state) {
	uint16_t fp16 = UINT16_C(0x7C00);
	while (state.KeepRunning()) {
		const uint32_t fp32 = fp16_ieee_to_fp32_bits(fp16++);
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_ieee_to_fp32_bits);

static void fp16_ieee_to_fp32_value(benchmark::State& state) {
	uint16_t fp16 = UINT16_C(0x7C00);
	while (state.KeepRunning()) {
		const float fp32 = fp16_ieee_to_fp32_value(fp16++);
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_ieee_to_fp32_value);

static void fp16_ieee_to_fp32_psimd(benchmark::State& state) {
	psimd_u16 fp16 = (psimd_u16) { 0x7C00, 0x7C01, 0x7C02, 0x7C03 };
	const psimd_u16 increment = psimd_splat_u16(4);
	while (state.KeepRunning()) {
		const psimd_f32 fp32 = fp16_ieee_to_fp32_psimd(fp16);
		fp16 += increment;
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_ieee_to_fp32_psimd);

static void fp16_ieee_to_fp32x2_psimd(benchmark::State& state) {
	psimd_u16 fp16 =
		(psimd_u16) { 0x7C00, 0x7C01, 0x7C02, 0x7C03, 0x7C04, 0x7C05, 0x7C06, 0x7C07 };
	const psimd_u16 increment = psimd_splat_u16(8);
	while (state.KeepRunning()) {
		const psimd_f32x2 fp32 = fp16_ieee_to_fp32x2_psimd(fp16);
		fp16 += increment;
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_ieee_to_fp32x2_psimd);

static void fp16_ieee_from_fp32_value(benchmark::State& state) {
	uint32_t fp32 = UINT32_C(0x7F800000);
	while (state.KeepRunning()) {
		const uint16_t fp16 = fp16_ieee_from_fp32_value(fp32_from_bits(fp32++));
		benchmark::DoNotOptimize(fp16);
	}
}
BENCHMARK(fp16_ieee_from_fp32_value);

#if (defined(__i386__) || defined(__x86_64__)) && defined(__F16C__)
	static void fp16_ieee_from_fp32_hardware(benchmark::State& state) {
		uint32_t fp32 = UINT32_C(0x7F800000);
		while (state.KeepRunning()) {
			const uint16_t fp16 = static_cast<uint16_t>(
				_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(fp32++), _MM_FROUND_CUR_DIRECTION)));
			benchmark::DoNotOptimize(fp16);
		}
	}
	BENCHMARK(fp16_ieee_from_fp32_hardware);
#endif

BENCHMARK_MAIN();
