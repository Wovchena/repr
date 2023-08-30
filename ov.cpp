#include <openvino/openvino.hpp>
#include <openvino/opsets/opset11.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    float DIVIDEND = 0x1.b7f1200000000p+0;
    float DIVISOR = 0x1.8000000000000p+0;  // 1.5
    std::cout << std::hexfloat << DIVIDEND << ' ' << DIVISOR << '\n';  // Make sure it's unchanged
    auto param_node = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    shared_ptr<ov::Model> model = make_shared<ov::Model>(param_node, ov::ParameterVector{param_node});
    ov::preprocess::PrePostProcessor ppp{model};
    ppp.input().preprocess().scale({DIVISOR});
    ov::CompiledModel scaler = ov::Core{}.compile_model(ppp.build(), "CPU");
    ov::InferRequest req = scaler.create_infer_request();
    req.get_input_tensor().data<float>()[0] = DIVIDEND;
    req.infer();
    std::cout << std::hexfloat << (DIVIDEND / DIVISOR) << ' ' << req.get_output_tensor().data<float>()[0] << '\n';
}
