#include <openvino/openvino.hpp>
#include "openvino/opsets/opset11.hpp"
#include <opencv2/opencv.hpp>

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/optional.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/gstreaming.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/parsers.hpp>
#include <opencv2/gapi/render/render.hpp>
#include <opencv2/gapi/streaming/format.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/render/render_types.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/infer/parsers.hpp>

using namespace std;

using PAInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat>;
G_API_NET(PersonDetActionRec, <PAInfo(cv::GMat)>, "person-detection-action-recognition");

cv::GRunArgs generator(const cv::GTypesInfo& info) {
    return std::vector<cv::GRunArg>{cv::GRunArg(cv::Mat{cv::Size(1920, 180), CV_8UC3})};
}

int main() {
    const std::array<std::string, 7> action_detector_6 = {"ActionNet/out_detection_loc",
                                                            "ActionNet/out_detection_conf",
                                                            "ActionNet/action_heads/out_head_1_anchor_1",
                                                            "ActionNet/action_heads/out_head_2_anchor_1",
                                                            "ActionNet/action_heads/out_head_2_anchor_2",
                                                            "ActionNet/action_heads/out_head_2_anchor_3",
                                                            "ActionNet/action_heads/out_head_2_anchor_4"};
    auto action_net =
        cv::gapi::ie::Params<PersonDetActionRec>{
            // https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-action-recognition-0006/FP16
            "/home/wov/.cache/omz/intel/person-detection-action-recognition-0006/FP16/person-detection-action-recognition-0006.xml",
            "/home/wov/.cache/omz/intel/person-detection-action-recognition-0006/FP16/person-detection-action-recognition-0006.bin",
            "CPU",
        }.cfgOutputLayers(action_detector_6);
    cv::gapi::GNetPackage networks;
    networks+= cv::gapi::networks(action_net);

    cv::GMat in;
    cv::GMat location, detect_confidences, priorboxes, action_con1, action_con2, action_con3, action_con4;
    std::tie(location, detect_confidences, priorboxes, action_con1, action_con2, action_con3, action_con4) =
        cv::gapi::infer<PersonDetActionRec>(in);
    auto outs = GOut(location);

    cv::GComputation pipeline(cv::GIn(in), std::move(outs));
    cv::GStreamingCompiled stream =
            pipeline.compileStreaming(cv::compile_args(cv::gapi::kernels<>(), networks));
    stream.setSource(cv::detail::ExtractArgsCallback{generator});
    stream.start();
    cv::Mat location_mat;
    stream.pull(cv::gout(location_mat));  // InferRequest for model: /home/wov/.cache/omz/intel/person-detection-action-recognition-0006/FP16/person-detection-action-recognition-0006.xml finished with InferenceEngine::StatusCode: -1
    // I also encountered error "double free or corruption (out)"
}
