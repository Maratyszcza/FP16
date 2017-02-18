#include <benchmark/benchmark.h>

#include <fp16.h>
#ifndef EMSCRIPTEN
	#include <fp16/psimd.h>
#endif

#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>


static void fp16_alt_to_fp32_bits(benchmark::State& state) {
	const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::bind(std::uniform_int_distribution<uint16_t>(0, 0x7BFF), std::mt19937(seed));

	std::vector<uint16_t> fp16(state.range(0));
	std::vector<uint32_t> fp32(state.range(0));
	std::generate(fp16.begin(), fp16.end(),
		[&rng]{ return fp16_alt_from_fp32_value(rng()); });

	while (state.KeepRunning()) {
		uint16_t* input = fp16.data();
		benchmark::DoNotOptimize(input);

		uint32_t* output = fp32.data();
		const size_t n = state.range(0);
		for (size_t i = 0; i < n; i++) {
			output[i] = fp16_alt_to_fp32_bits(input[i]);
		}

		benchmark::DoNotOptimize(output);
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}
BENCHMARK(fp16_alt_to_fp32_bits)->RangeMultiplier(2)->Range(1<<10, 64<<20);

static void fp16_alt_to_fp32_value(benchmark::State& state) {
	const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::bind(std::uniform_int_distribution<uint16_t>(0, 0x7BFF), std::mt19937(seed));

	std::vector<uint16_t> fp16(state.range(0));
	std::vector<float> fp32(state.range(0));
	std::generate(fp16.begin(), fp16.end(),
		[&rng]{ return fp16_alt_from_fp32_value(rng()); });

	while (state.KeepRunning()) {
		uint16_t* input = fp16.data();
		benchmark::DoNotOptimize(input);

		float* output = fp32.data();
		const size_t n = state.range(0);
		for (size_t i = 0; i < n; i++) {
			output[i] = fp16_alt_to_fp32_value(input[i]);
		}

		benchmark::DoNotOptimize(output);
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}
BENCHMARK(fp16_alt_to_fp32_value)->RangeMultiplier(2)->Range(1<<10, 64<<20);

#ifndef EMSCRIPTEN
	static void fp16_alt_to_fp32_psimd(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_int_distribution<uint16_t>(0, 0x7BFF), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_alt_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n - 4; i += 4) {
				psimd_store_f32(&output[i],
					fp16_alt_to_fp32_psimd(
						psimd_load_u16(&input[i])));
			}
			const psimd_u16 last_vector = { input[n - 4], input[n - 3], input[n - 2], input[n - 1] };
			psimd_store_f32(&output[n - 4],
				fp16_alt_to_fp32_psimd(last_vector));

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(fp16_alt_to_fp32_psimd)->RangeMultiplier(2)->Range(1<<10, 64<<20);

	static void fp16_alt_to_fp32x2_psimd(benchmark::State& state) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_int_distribution<uint16_t>(0, 0x7BFF), std::mt19937(seed));

		std::vector<uint16_t> fp16(state.range(0));
		std::vector<float> fp32(state.range(0));
		std::generate(fp16.begin(), fp16.end(),
			[&rng]{ return fp16_alt_from_fp32_value(rng()); });

		while (state.KeepRunning()) {
			uint16_t* input = fp16.data();
			benchmark::DoNotOptimize(input);

			float* output = fp32.data();
			const size_t n = state.range(0);
			for (size_t i = 0; i < n; i += 8) {
				const psimd_f32x2 data =
					fp16_alt_to_fp32x2_psimd(
						psimd_load_u16(&input[i]));
				psimd_store_f32(&output[i], data.lo);
				psimd_store_f32(&output[i + 4], data.hi);
			}

			benchmark::DoNotOptimize(output);
		}
		state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
	}
	BENCHMARK(fp16_alt_to_fp32x2_psimd)->RangeMultiplier(2)->Range(1<<10, 64<<20);
#endif

BENCHMARK_MAIN();
