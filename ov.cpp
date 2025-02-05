#include <openvino/openvino.hpp>

namespace {
void iter(ov::InferRequest& embed, ov::InferRequest& llm, const ov::Tensor& attention_mask) {
    embed.infer();
    llm.set_tensor("attention_mask", attention_mask);
    llm.infer();
}
}

int main(int argc, char* argv[]) {
    constexpr size_t NUM_ITERS = 100;
    ov::Core core;
    ov::InferRequest embed = core.compile_model(argv[1] + std::string{"/openvino_text_embeddings_model.xml"}, "GPU", ov::cache_dir("vlm_cache")).create_infer_request();
    ov::InferRequest llm = core.compile_model(argv[1] + std::string{"/openvino_language_model.xml"}, "GPU", ov::cache_dir("vlm_cache")).create_infer_request();
    ov::Tensor input_ids{ov::element::i64, {1, 1}};
    input_ids.data<int64_t>()[0] = 1;
    embed.set_input_tensor(input_ids);
    // ov::RemoteContext context = embed.get_compiled_model().get_context();
    // embed.set_output_tensor(context.create_tensor(ov::element::f32, {1, 1, 3072}));
    embed.infer();
    ov::Tensor inputs_embeds = embed.get_output_tensor();
    llm.set_tensor("inputs_embeds", inputs_embeds);
    std::vector<int64_t> attention_mask_data(NUM_ITERS, 1);
    llm.set_tensor("attention_mask", ov::Tensor{ov::element::i64, {1, 1}, attention_mask_data.data()});
    ov::Tensor position_ids{ov::element::i64, {1, 1}};
    position_ids.data<int64_t>()[0] = 0;
    llm.set_tensor("position_ids", position_ids);
    ov::Tensor beam_idx{ov::element::i32, {1}};
    beam_idx.data<int32_t>()[0] = 0;
    llm.set_tensor("beam_idx", beam_idx);
    llm.infer();

    position_ids.data<int64_t>()[0] = 1;
    iter(embed, llm, ov::Tensor{ov::element::i64, {1, 2}, attention_mask_data.data()});

    auto t0 = std::chrono::steady_clock::now();
    for (size_t id = 2; id < NUM_ITERS; ++id) {
        position_ids.data<int64_t>()[0] = id;
        input_ids.data<int64_t>()[0] = id;
        // embed.set_output_tensor(context.create_tensor(ov::element::f32, {1, 1, 3072}));
        iter(embed, llm, ov::Tensor{ov::element::i64, {1, id+1}, attention_mask_data.data()});
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << '\n';
}
