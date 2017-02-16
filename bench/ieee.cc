#include <benchmark/benchmark.h>

#include <fp16.h>

#if (defined(__i386__) || defined(__x86_64__)) && defined(__F16C__)
	#include <x86intrin.h>
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
