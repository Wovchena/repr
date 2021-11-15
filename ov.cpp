#include <openvino/openvino.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace ov::runtime;
using namespace cv;
using namespace std;
int main() {
    Core core;
    shared_ptr<ov::Function> model = core.read_model("C:\\Users\\vzlobin\\Downloads\\human-pose-estimation-0001.xml");
    InferRequest req = core.compile_model(model, "GPU").create_infer_request();
    Mat mat = imread("c:\\Users\\vzlobin\\r\\repr\\ILSVRC2012_val_00000002.JPEG");
    Tensor in = req.get_input_tensor();

    float* data = in.data<float>();
    int C = in.get_shape()[1], H = in.get_shape()[2], W = in.get_shape()[3];
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                data[c*H*W + h*W + w] = mat.at<Vec3b>(h, w)[c];
    imshow("in", Mat{H, W, CV_32F, data}/255);  // verify first input plane is same
    req.infer();

    ov::Output<ov::Node> heatMapsOut = model->outputs()[1];
    cout << heatMapsOut.get_any_name() << '\n';
    Tensor out = req.get_output_tensor(heatMapsOut.get_index());

    imshow("out", Mat{32, 57, CV_32F, out.data<float>()});
    waitKey();
    return 0;
}
