#include <openvino/openvino.hpp>
#include "openvino/opsets/opset11.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

float absdiff(const cv::Mat ref, ov::Tensor pred) {
    float diff = 0.0f;
    float* ref_data = (float*)(ref.data);
    float* pred_data = pred.data<float>();
    for (size_t i = 0; i < pred.get_size(); ++i) {
        diff = max(std::abs(ref_data[i] - pred_data[i]), diff);
    }
    return diff;
}

int main() {
    // identity
    ov::Shape input_shape{1, 20, 20, 3};
    auto param_node = make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    ov::Core core;
    shared_ptr<ov::Model> model = make_shared<ov::Model>(param_node, ov::ParameterVector{param_node});
    ov::CompiledModel identity = core.compile_model(model, "CPU");
    ov::InferRequest req = identity.create_infer_request();
    cv::Mat inp{cv::Size(20, 20), CV_32FC3};
    cv::randu(inp, 0.0, 255.0);
    req.set_input_tensor(ov::Tensor{ov::element::f32, input_shape, inp.data});
    req.infer();
    if (0.0f != absdiff(inp, req.get_output_tensor())) {
        cerr << "identity diff\n";
        exit(1);
    }
    // scale
    ov::preprocess::PrePostProcessor ppp{model};
    ppp.input().preprocess().scale({255.0f});
    model = ppp.build();
    ov::CompiledModel scaler = core.compile_model(model, "CPU");
    req = scaler.create_infer_request();
    req.set_input_tensor(ov::Tensor{ov::element::f32, input_shape, inp.data});
    req.infer();
    if (0.0f != absdiff(inp / 255.0f, req.get_output_tensor())) {
        cerr << "scale diff\n";
        exit(1);
    }
    return 0;
}
