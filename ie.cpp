#include <inference_engine.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace std;
int main() {
    InferenceEngine::Core core;
    CNNNetwork model = core.ReadNetwork("C:\\Users\\vzlobin\\Downloads\\human-pose-estimation-0001.xml");
    InferRequest req = core.LoadNetwork(model, "GPU").CreateInferRequest();
    Mat mat = imread("c:\\Users\\vzlobin\\r\\repr\\ILSVRC2012_val_00000002.JPEG");
    MemoryBlob::Ptr in = static_pointer_cast<MemoryBlob>(req.GetBlob(model.getInputsInfo().begin()->first));
    LockedMemory<void> inMem = in->buffer();
    float* data = inMem.as<float*>();
    int C = in->getTensorDesc().getDims()[1], H = in->getTensorDesc().getDims()[2], W = in->getTensorDesc().getDims()[3];
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                data[c*H*W + h*W + w] = mat.at<Vec3b>(h, w)[c];
    imshow("in", Mat{H, W, CV_32F, data}/255);  // verify first input plane is same
    req.Infer();

    string heatMapsName = (++model.getOutputsInfo().begin())->first;
    cout << heatMapsName << '\n';
    MemoryBlob::Ptr out = static_pointer_cast<MemoryBlob>(req.GetBlob(heatMapsName));
    LockedMemory<void> outMem = out->buffer();
    imshow("out", Mat{32, 57, CV_32F, outMem.as<float*>()});
    waitKey();
    return 0;
}
