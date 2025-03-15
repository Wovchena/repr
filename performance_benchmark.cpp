// A command to remind myself how to disassemble after compiling Release version but link with OV Debug because I usually build OV in Debug. -ggdb for debug symbols but -O3 and -DNDEBUG for Release: Treat headers as system's to suppress their warnings.
// Add -finline-limit=1234 to inline functions
// g++ -ggdb -O3 -DNDEBUG -Wall -Wextra -pedantic -Werror --std c++23 -Xlinker -rpath -Xlinker /home/vzlobin/r/v/bin/intel64/Debug -lpthread -isystem /home/vzlobin/r/v/src/core/include/ -isystem /home/vzlobin/r/v/src/inference/include/ -isystem /home/vzlobin/r/benchmark/include/ ov.cpp /home/vzlobin/r/v/bin/intel64/Debug/libopenvino.so.2025.1.0 /home/vzlobin/r/benchmark/build/src/libbenchmark_main.a /home/vzlobin/r/benchmark/build/src/libbenchmark.a && gdb -ex 'set pagination off' -ex 'disassemble /s experiment_implementation' --args ./a.out --benchmark_min_time=5s
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
    Tensor tensor2{element::f32, {0}};
    tensor2.data<float>();
    return tensor2;
}

[[maybe_unused]] pair<Tensor, Tensor> create_pair() {
    return {create_first(), create_second()};
}

[[maybe_unused]] pair<Tensor, Tensor> inplace() {
    // TODO: call create_*
    Tensor tensor1{element::f32, {0}};
    tensor1.data<float>();
    Tensor tensor2{element::f32, {0}};
    tensor2.data<float>();
    return {move(tensor1), move(tensor2)};  // TODO: benchmark move and no move. Also benchmark inplace()
}
}

void baseline_implementation() {
    auto [tensor1, tensor2] = inplace();
}

void experiment_implementation() {
    auto [tensor1, tensor2] = create_pair();
}

// Boilerplate
namespace {
void baseline(benchmark::State& state) {
    for (benchmark::State::StateIterator::Value _ : state) {
        baseline_implementation();
    }
}
void experiment(benchmark::State& state) {
    for (benchmark::State::StateIterator::Value _ : state) {
        experiment_implementation();
    }
}
BENCHMARK(baseline);
BENCHMARK(experiment);
}
BENCHMARK_MAIN();
