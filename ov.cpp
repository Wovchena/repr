#include <openvino/openvino.hpp>
// #include <opencv2/opencv.hpp>
using namespace std;

int main(int argc, char* argv[]) {
    // ov::Core core;
    ov::CompiledModel compiled_model = ov::Core{}.compile_model(argv[1], "CPU");
    try {
        compiled_model.get_property(ov::optimal_number_of_infer_requests);
    } catch (const std::exception& ex) {
        std::cerr << ex.what();
        return 1;
    }
    return 0;
}
