#include <openvino/openvino.hpp>
// A command to remind myself how to disassemble after compiling Release version but link with OV Debug because I usually build OV in Debug. -ggdb for debug symbols but -O3 and -DNDEBUG for Release: Treat headers as system's to suppress their warnings.
// g++ -ggdb -O3 -DNDEBUG -Wall -Wextra -pedantic -Werror --std c++23 -Xlinker -rpath -Xlinker /home/vzlobin/r/v/bin/intel64/Debug -isystem /home/vzlobin/r/v/src/core/include/ -isystem /home/vzlobin/r/v/src/inference/include/ ov.cpp /home/vzlobin/r/v/bin/intel64/Debug/libopenvino.so.2025.1.0 && gdb -ex 'set pagination off' -ex 'disassemble /s main' --args ./a.out arg
int main() {
    using namespace std;
    volatile size_t vector_size = 0;
    vector<int> vector(vector_size);
    for (size_t idx = 0; idx < vector.size(); ++idx) {
        volatile size_t do_not_optimize_away = idx;
        (void)do_not_optimize_away;
    }
    // ov::Tensor tensor{ov::element::f32, {0}};
    // struct TensorIterator {
    //     size_t idx;
    //     size_t operator*() {return idx;}
    //     TensorIterator& operator++() {++idx; return *this;}
    //     bool operator!=(const TensorIterator& other) {return idx != other.idx;}
    // };
    // struct IteratorProvider {
    //     const ov::Tensor& tensor;
    //     TensorIterator begin() {return {0};}
    //     TensorIterator end() {return {tensor.get_size()};}
    // };
    // IteratorProvider provider{tensor};

    // float* data = tensor.data<float>();
    // for (size_t idx : provider) {
    //     data[idx] = idx;
    // }

    // for (size_t idx = 0; idx < tensor.get_size(); ++idx) {
    //     volatile size_t do_not_optimize_away = idx;
    // }

    // for (size_t idx : provider) {
    //     tensor.data<float>()[idx] = 0;
    // }
}
