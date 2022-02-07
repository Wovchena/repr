#include <openvino/openvino.hpp>
using namespace ov;
using namespace std;
struct Callback {
    ov::InferRequest ireq;
    Callback() {
        std::cout << "default ctor\n";
    }
    Callback(const ov::InferRequest& ireq): ireq{ireq} {
        std::cout << "ctor\n";
    }
    Callback(const Callback& other): ireq{other.ireq} {
        std::cout << "copy ctor\n";
    }
    Callback(Callback&& other): ireq{move(other.ireq)} {
        std::cout << "move ctor\n";
    }
    ~Callback() {
        cout << "dtor\n";
    }
    void operator()(exception_ptr) {
        std::cout << "callback\n";
    }
};

int main() {
    Core core;
    auto ireq = core.compile_model(core.read_model(
            "C:\\Users\\vzlobin\\Downloads\\d\\intel\\age-gender-recognition-retail-0013\\FP32\\age-gender-recognition-retail-0013.xml"))
        .create_infer_request();
    ireq.set_callback(Callback{ireq});
    ireq.start_async();
    ireq.cancel();
    try {
        // ireq.set_callback([](exception_ptr){});  // [ INFER_CANCELLED ]
    } catch(exception e) {std::cout << e.what();}
    return 0;
}
