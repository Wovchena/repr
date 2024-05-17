#include <openvino/openvino.hpp>

int main(int argc, char* argv[]) {
    ov::Core core;
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
    ov::Tensor input_ids{ov::element::i64, {BATCH_SIZE, 3}};
    input_ids.data<int64_t>()[0] = 1;
    input_ids.data<int64_t>()[1] = 408;
    input_ids.data<int64_t>()[2] = 2176;
    ov::InferRequest ireq = compiled.create_infer_request();
    ireq.set_tensor("input_ids", input_ids);
    ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("input_ids").get_size()});
    std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), input_ids.get_size(), 1);
    ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
    std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);
    for (size_t idx = 3; idx < inputs.size(); ++idx) {
        ireq.get_input_tensor(idx).set_shape(inputs.at(idx).get_partial_shape().get_min_shape());
    }
    ireq.infer();
    ov::InferRequest ireq2 = ireq;
    // ireq2 = compiled.create_infer_request();  // This fixes the problem
    ireq2.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    ireq2.get_tensor("input_ids").data<int64_t>()[0] = 408;
    ireq2.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("attention_mask").get_size() + 1});
    std::fill_n(ireq2.get_tensor("attention_mask").data<int64_t>(), ireq2.get_tensor("attention_mask").get_size(), 1);
    ireq2.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
    ireq2.get_tensor("position_ids").data<int64_t>()[0] = ireq2.get_tensor("attention_mask").get_size() - 1;
    for (size_t tensor_idx = 3; tensor_idx < inputs.size(); ++tensor_idx) {
        ireq2.set_input_tensor(tensor_idx, ireq.get_output_tensor(tensor_idx - 2));
    }
    ireq2.infer();
}
a
