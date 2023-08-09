#include <openvino/openvino.hpp>
#include "openvino/opsets/opset11.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    ov::Core core;
    auto model = core.read_model("/home/wov/.cache/omz/intel/semantic-segmentation-adas-0001/FP16/semantic-segmentation-adas-0001.xml");
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    core.compile_model(model, "GPU");
}
