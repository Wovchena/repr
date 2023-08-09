#include <openvino/openvino.hpp>
#include "openvino/opsets/opset11.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    ov::Core core;
    auto model = core.read_model("/home/wov/.cache/omz/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml");
    ov::CompiledModel compiled = core.compile_model(model, "GPU", {  // Replace ov::Model with string to see speed up
        {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::hint::num_requests(3)}  // Set num_requests(2) to see speed up
    });
    ov::InferRequest req = compiled.create_infer_request();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; ; ++i) {
        req.infer();
        std::cout << "FPS: " << i * 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count() << '\n';
    }
}
