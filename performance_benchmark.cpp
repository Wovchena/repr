// A command to remind myself how to disassemble after compiling. Treat headers as system's to suppress their warnings. Add
// -ggdb for debug symbols
// -finline-limit=1234 to inline functions
// g++ -O3 -DNDEBUG -Wall -Wextra -pedantic -Werror --std c++23 -Xlinker -rpath -Xlinker /home/vzlobin/r/v/bin/intel64/Release -lpthread -isystem /home/vzlobin/r/v/src/core/include/ -isystem /home/vzlobin/r/v/src/inference/include/ -isystem /home/vzlobin/r/repr/benchmark/include/ /home/vzlobin/r/repr/performance_benchmark.cpp /home/vzlobin/r/v/bin/intel64/Release/libopenvino.so.2025.1.0 /home/vzlobin/r/repr/build/benchmark/src/libbenchmark.a /home/vzlobin/r/repr/build/benchmark/src/libbenchmark_main.a && gdb -ex 'set pagination off' -ex 'disassemble /s create_pair_caller' --args ./a.out --benchmark_min_warmup_time=0.9 --benchmark_min_time=5s

// Reduce variance:
// https://github.com/google/benchmark/blob/main/docs/reducing_variance.md#reducing-variance-in-benchmarks
// Fix CPU frequency
// Mann–Whitney U test: https://github.com/google/benchmark/blob/main/docs/tools.md
#include <benchmark/benchmark.h>
#include <openvino/runtime/core.hpp>
using namespace ov;
using namespace std;
namespace {

Tensor create_first() {
    Tensor tensor1{element::f32, {0}};
    tensor1.data<float>();
    return tensor1;
}

Tensor create_second() {
    Tensor tensor2{element::f32, {1}};
    tensor2.data<float>()[0] = 123.0f;
    return tensor2;
}

pair<Tensor, Tensor> create_pair() {
    return {create_first(), create_second()};
}

/// @brief Expected to be equivalent to create_pair() but it's not
pair<Tensor, Tensor> create_pair_move() {
    return {move(create_first()), move(create_second())};
}

pair<Tensor, Tensor> inplace() {
    Tensor tensor1{element::f32, {0}};
    tensor1.data<float>();
    Tensor tensor2{element::f32, {1}};
    tensor2.data<float>()[0] = 123.0f;
    return {tensor1, tensor2};
}

pair<Tensor, Tensor> inplace_move() {
    Tensor tensor1{element::f32, {0}};
    tensor1.data<float>();
    Tensor tensor2{element::f32, {1}};
    tensor2.data<float>()[0] = 123.0f;
    return {move(tensor1), move(tensor2)};
}

pair<Tensor, Tensor> inplace_create_call() {
    Tensor tensor1 = create_first();
    Tensor tensor2 = create_second();
    return {tensor1, tensor2};
}

pair<Tensor, Tensor> inplace_create_call_move() {
    Tensor tensor1 = create_first();
    Tensor tensor2 = create_second();
    return {move(tensor1), move(tensor2)};
}
}

void create_pair_caller() {
    auto [tensor1, tensor2] = create_pair();
}

void create_pair_move_caller() {
    auto [tensor1, tensor2] = create_pair_move();
}

void inplace_caller() {
    auto [tensor1, tensor2] = inplace();
}

void inplace_move_caller() {
    auto [tensor1, tensor2] = inplace_move();
}

void inplace_create_call_caller() {
    auto [tensor1, tensor2] = inplace_create_call();
}


void inplace_create_call_move_caller() {
    auto [tensor1, tensor2] = inplace_create_call_move();
}

// 2025-03-16T11:22:13+04:00
// Running ./a.out
// Run on (24 X 5100 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x12)
//   L1 Instruction 32 KiB (x12)
//   L2 Unified 1280 KiB (x12)
//   L3 Unified 30720 KiB (x1)
// Load Average: 0.58, 0.74, 0.79
// ***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
// ---------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations
// ---------------------------------------------------------------------
// create_pair_b                     201 ns          201 ns    300000000
// create_pair_move_b                202 ns          202 ns    300000000
// inplace_move_b                    202 ns          202 ns    300000000
// inplace_b                         203 ns          203 ns    300000000
// inplace_create_call_b             204 ns          203 ns    300000000
// inplace_create_call_move_b        204 ns          204 ns    300000000

// create_pair_move_b                201 ns          201 ns    300000000
// create_pair_b                     202 ns          202 ns    300000000
// inplace_move_b                    202 ns          202 ns    300000000
// inplace_b                         203 ns          203 ns    300000000
// inplace_create_call_b             204 ns          204 ns    300000000
// inplace_create_call_move_b        204 ns          204 ns    300000000

// create_pair_b                     201 ns          201 ns    300000000
// create_pair_move_b                202 ns          202 ns    300000000
// inplace_move_b                    202 ns          202 ns    300000000
// inplace_b                         203 ns          203 ns    300000000
// inplace_create_call_b             203 ns          203 ns    300000000
// inplace_create_call_move_b        204 ns          204 ns    300000000

// create_pair_move_b                201 ns          201 ns    300000000
// create_pair_b                     202 ns          202 ns    300000000
// inplace_move_b                    202 ns          202 ns    300000000
// inplace_b                         203 ns          203 ns    300000000
// inplace_create_call_b             203 ns          203 ns    300000000
// inplace_create_call_move_b        204 ns          204 ns    300000000

// inplace_create_call_b is faster than inplace_create_call_move_b. I see that move triggers swap in disassemble instead of new shared_ptr. At the same time cleanup part is shorter for inplace_create_call_b.
// inplace_b is faster than inplace_create_call_b. The difference is in clean up implementation.
// inplace_move_b is faster than inplace_b. inplace_move_b uses swap and clean up is larger for it only by a few lines.
// create_pair_b and create_pair_move_b is fasther than inplace_move_b. Source code marks are gone for create_pair_move_b for swap call, but assembly code is there. create_pair_move_b calls _ZN2ov6TensorD1Ev 4 times and inplace_move_b calls it two extra times on on the very end. _ZN2ov9AllocatorD1Ev is called 2 times for the faster version and 4 times for the slower version. The strange thing is that the reported time is identical, bot there's fewer assembly code.
// create_pair_b is on par with create_pair_move_b. create_pair_b has source code marks, but create_pair_move_b doesn't. The assembly is the same for construction. The difference is in cleanup. create_pair_b has extra tail in the very end. Maybe this tail is to handle another case more efficiently

// Overall conclusion is that unnamed return value optimization is more efficient than a named one because create_pair_* are the fastest.
// Move of a return value doesn't change performance although it makes clean up implementation simpler.
// Move is debatable for named return values. It's faster for inplace_move_b, but slower for inplace_create_call_move_b.

// Boilerplate
namespace {
void create_pair_b(benchmark::State& state) {
    for (benchmark::State::StateIterator::Value _ : state) {
        create_pair_caller();
    }
}
void create_pair_move_b(benchmark::State& state) {
    for (benchmark::State::StateIterator::Value _ : state) {
        create_pair_move_caller();
    }
}
void inplace_b(benchmark::State& state) {
    for (benchmark::State::StateIterator::Value _ : state) {
        inplace_caller();
    }
}
void inplace_move_b(benchmark::State& state) {
    for (benchmark::State::StateIterator::Value _ : state) {
        inplace_move_caller();
    }
}
void inplace_create_call_b(benchmark::State& state) {
    for (benchmark::State::StateIterator::Value _ : state) {
        inplace_create_call_caller();
    }
}
void inplace_create_call_move_b(benchmark::State& state) {
    for (benchmark::State::StateIterator::Value _ : state) {
        inplace_create_call_move_caller();
    }
}
BENCHMARK(create_pair_b);
BENCHMARK(create_pair_move_b);
BENCHMARK(inplace_b);
BENCHMARK(inplace_move_b);
BENCHMARK(inplace_create_call_b);
BENCHMARK(inplace_create_call_move_b);
}
BENCHMARK_MAIN();
