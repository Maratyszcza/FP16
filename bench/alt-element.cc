#include <benchmark/benchmark.h>

#include <fp16.h>
#include <fp16/psimd.h>

static void fp16_alt_to_fp32_bits(benchmark::State& state) {
	uint16_t fp16 = UINT16_C(0x7C00);
	while (state.KeepRunning()) {
		const uint32_t fp32 = fp16_alt_to_fp32_bits(fp16++);
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_alt_to_fp32_bits);

static void fp16_alt_to_fp32_value(benchmark::State& state) {
	uint16_t fp16 = UINT16_C(0x7C00);
	while (state.KeepRunning()) {
		const float fp32 = fp16_alt_to_fp32_value(fp16++);
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_alt_to_fp32_value);

static void fp16_alt_to_fp32_psimd(benchmark::State& state) {
	psimd_u16 fp16 = (psimd_u16) { 0x7C00, 0x7C01, 0x7C02, 0x7C03 };;
	const psimd_u16 increment = psimd_splat_u16(4);
	while (state.KeepRunning()) {
		const psimd_f32 fp32 = fp16_alt_to_fp32_psimd(fp16);
		fp16 += increment;
		benchmark::DoNotOptimize(fp32);
	}
}
BENCHMARK(fp16_alt_to_fp32_psimd);

static void fp16_alt_from_fp32_value(benchmark::State& state) {
	uint32_t fp32 = UINT32_C(0x7F800000);
	while (state.KeepRunning()) {
		const uint16_t fp16 = fp16_alt_from_fp32_value(fp32_from_bits(fp32++));
		benchmark::DoNotOptimize(fp16);
	}
}
BENCHMARK(fp16_alt_from_fp32_value);

BENCHMARK_MAIN();
