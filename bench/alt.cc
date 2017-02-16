#include <benchmark/benchmark.h>

#include <fp16.h>

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

static void fp16_alt_from_fp32_value(benchmark::State& state) {
	uint32_t fp32 = UINT32_C(0x7F800000);
	while (state.KeepRunning()) {
		const uint16_t fp16 = fp16_alt_from_fp32_value(fp32_from_bits(fp32++));
		benchmark::DoNotOptimize(fp16);
	}
}
BENCHMARK(fp16_alt_from_fp32_value);

BENCHMARK_MAIN();
