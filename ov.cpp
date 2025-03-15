#include <benchmark/benchmark.h>
#include <openvino/openvino.hpp>
// A command to remind myself how to disassemble after compiling Release version but link with OV Debug because I usually build OV in Debug. -ggdb for debug symbols but -O3 and -DNDEBUG for Release: Treat headers as system's to suppress their warnings.
// Add -finline-limit=999 to inline functions
// g++ -ggdb -O3 -DNDEBUG -Wall -Wextra -pedantic -Werror --std c++23 -Xlinker -rpath -Xlinker /home/vzlobin/r/v/bin/intel64/Debug -lpthread -isystem /home/vzlobin/r/v/src/core/include/ -isystem /home/vzlobin/r/v/src/inference/include/ -isystem /home/vzlobin/r/benchmark/include/ ov.cpp /home/vzlobin/r/v/bin/intel64/Debug/libopenvino.so.2025.1.0 /home/vzlobin/r/benchmark/build/src/libbenchmark_main.a /home/vzlobin/r/benchmark/build/src/libbenchmark.a && gdb -ex 'set pagination off' -ex 'disassemble /s function_of_interest' --args ./a.out --benchmark_min_time=1s
using namespace ov;
using namespace std;
namespace {
Tensor create_first() {
    ov::Tensor tensor1{ov::element::f32, {1}};
    tensor1.data<float>()[0] = 0;
    return tensor1;
}

Tensor create_second() {
    ov::Tensor tensor2{ov::element::f32, {2}};
    tensor2.data<float>()[0] = 1;
    return tensor2;
}

[[maybe_unused]] pair<Tensor, Tensor> create_pair() {
    return {create_first(), create_second()};
}

[[maybe_unused]] pair<Tensor, Tensor> inplace() {
    // TODO: call create_*
    ov::Tensor tensor1{ov::element::f32, {1}};
    tensor1.data<float>()[0] = 0;
    ov::Tensor tensor2{ov::element::f32, {2}};
    tensor2.data<float>()[0] = 1;
    return {move(tensor1), move(tensor2)};  // TODO: benchmark move and no move. Also benchmark inplace()
}
}

void function_of_interest() {
    auto [tensor1, tensor2] = create_pair();
}

// Boilerplate
namespace {
void main_loop(benchmark::State& state) {
    for (auto _ : state) {
        function_of_interest();
    }
}
BENCHMARK(main_loop);
BENCHMARK_MAIN();
}
