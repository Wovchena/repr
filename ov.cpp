#include <openvino/openvino.hpp>
#include "openvino/opsets/opset11.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    ov::Core core;
    auto model = core.read_model("/home/wov/.cache/omz/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml");
    ov::set_batch(model, {1, 2});
    core.compile_model(model, "HETERO:GPU");  // fails
}
