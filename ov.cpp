#include <openvino/openvino.hpp>
using namespace ov;
using namespace std;
int main() {
    Core core;
    shared_ptr<Model> model = core.read_model(
        "C:\\Users\\vzlobin\\Downloads\\d\\intel\\age-gender-recognition-retail-0013\\FP32\\age-gender-recognition-retail-0013.xml");
    Shape inShape = model->input().get_shape();
    preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_shape({
        numeric_limits<int64_t>::max(), inShape[1], inShape[2], inShape[3]});
    try {
        model = ppp.build();
    } catch(exception e) {std::cout << e.what();}
    return 0;
}
