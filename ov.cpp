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

void test_float_resize() {
    ov::Shape input_shape{1, 20, 20, 3};
    cv::Mat inp{cv::Size(20, 20), CV_32FC3}, resized;
    cv::randu(inp, 0.0, 255.0);
    cv::resize(inp, resized, {40, 40});
    auto param_node = make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    ov::opset11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::opset11::Interpolate::InterpolateMode::LINEAR;
    attrs.shape_calculation_mode = ov::opset11::Interpolate::ShapeCalcMode::SIZES;
    auto interpolate = std::make_shared<ov::opset11::Interpolate>(param_node, ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {40, 40}), ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2}), attrs);
    auto model = std::make_shared<ov::Model>(interpolate, ov::ParameterVector{param_node});
    ov::Core core;
    ov::CompiledModel interpolator = core.compile_model(model, "CPU");
    ov::InferRequest req = interpolator.create_infer_request();
    req.set_input_tensor(ov::Tensor{ov::element::f32, input_shape, inp.data});
    req.infer();
    std::cout << absdiff(resized, req.get_output_tensor()) << '\n';  // diff == 3e-5
}

int main() {
    cv::Mat inp{cv::Size(20, 20), CV_32FC3};
    cv::randu(inp, 0.0, 255.0);
    cv::Mat resized, int_inp;
    inp.convertTo(int_inp, CV_8UC3);
    cv::resize(int_inp, resized, {40, 40});
    ov::Shape input_shape{1, 20, 20, 3};
    auto int_param_node = make_shared<ov::op::v0::Parameter>(ov::element::u8, input_shape);
    ov::opset11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::opset11::Interpolate::InterpolateMode::LINEAR;
    attrs.shape_calculation_mode = ov::opset11::Interpolate::ShapeCalcMode::SIZES;
    auto interpolate = std::make_shared<ov::opset11::Interpolate>(int_param_node, ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {40, 40}), ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2}), attrs);
    shared_ptr<ov::Model> model = std::make_shared<ov::Model>(interpolate, ov::ParameterVector{int_param_node});
    ov::Core core;
    ov::CompiledModel interpolator = core.compile_model(model, "CPU");
    ov::InferRequest req = interpolator.create_infer_request();
    req.set_input_tensor(ov::Tensor{ov::element::u8, input_shape, int_inp.data});
    req.infer();
    ov::Tensor pred = req.get_output_tensor();
    int diff = 0;
    uint8_t* ref_data = (uint8_t*)(resized.data);
    uint8_t* pred_data = pred.data<uint8_t>();
    for (size_t i = 0; i < pred.get_size(); ++i) {
        diff = max(std::abs(int(ref_data[i]) - int(pred_data[i])), diff);
    }
    std::cout << diff << '\n';  // diff == 1
    test_float_resize();
    return 0;
}
