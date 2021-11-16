#include <openvino/openvino.hpp>
using namespace ov;
using namespace std;
int main() {
    runtime::Core core;
    shared_ptr<Function> model = core.read_model("C:\\Users\\vzlobin\\Downloads\\d\\public\\yolo-v3-tiny-tf\\FP32\\yolo-v3-tiny-tf.xml");
    Output<Node> out = model->input();
    cout << "get_any_name:\n";
    cout << out.get_any_name() << '\n';
    return 0;
}
