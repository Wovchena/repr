#include <openvino/openvino.hpp>
#include <openvino/opsets/opset11.hpp>
#include <opencv2/opencv.hpp>

constexpr size_t BATCH_SIZE = 1;

void second_infer(ov::InferRequest req, ov::InferRequest cache, const std::vector<ov::Output<ov::Node>>& inputs) {
    req.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    req.get_tensor("attention_mask").set_shape({BATCH_SIZE, cache.get_tensor("attention_mask").get_size() + 1});
    std::fill_n(req.get_tensor("attention_mask").data<int64_t>(), req.get_tensor("attention_mask").get_size(), 1);
    req.get_tensor("input_ids").data<int64_t>()[0] = 408;
    req.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
    req.get_tensor("position_ids").data<int64_t>()[0] = cache.get_tensor("attention_mask").get_size() - 2;
    for (size_t tensor_idx = 3; tensor_idx < inputs.size(); ++tensor_idx) {
        req.set_input_tensor(tensor_idx, cache.get_output_tensor(tensor_idx - 2));
    }
    req.infer();
    ov::Tensor logits_tensor = req.get_tensor("logits");
    size_t vocab_size = req.get_tensor("logits").get_shape().back();
    float* data = logits_tensor.data<float>() + (logits_tensor.get_shape()[1] - 1) * vocab_size;
    std::cout << data[263] << ' ' << data[2176] << '\n';
}

int main(int argc, char* argv[]) {
    ov::Core core;
    ov::Tensor input_ids{ov::element::i64, {1, 3}};
    input_ids.data<int64_t>()[0] = 1;
    input_ids.data<int64_t>()[1] = 408;
    input_ids.data<int64_t>()[2] = 2176;
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    constexpr size_t BATCH_SIZE = 1;
    std::map<size_t, ov::PartialShape> shapes = {
        {0, ov::PartialShape{
            BATCH_SIZE, -1
        }},
        {1, ov::PartialShape{
            BATCH_SIZE, -1
        }}
    };
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    for (size_t idx = 3; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = BATCH_SIZE;
        shapes.emplace(idx, shape);
    }
    model->reshape(shapes);
    ov::CompiledModel compiled = core.compile_model(model, "CPU");
    ov::InferRequest ireq = compiled.create_infer_request();
    for (size_t idx = 2; idx < inputs.size(); ++idx) {
        ireq.get_input_tensor(idx).set_shape(inputs.at(idx).get_partial_shape().get_min_shape());
    }
    ireq.get_tensor("input_ids").set_shape(input_ids.get_shape());  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
    ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("input_ids").get_size()});
    std::copy_n(input_ids.data<const int64_t>(), input_ids.get_size(), ireq.get_tensor("input_ids").data<int64_t>());
    std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), input_ids.get_size(), 1);
    ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
    std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);
    ireq.infer();

    second_infer(compiled.create_infer_request(), ireq, inputs);
    second_infer(ireq, ireq, inputs);
}
