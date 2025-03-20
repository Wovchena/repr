import openvino_genai
import openvino_tokenizers
import optimum.intel
import transformers


def main():
    model_id = 'katuni4ka/tiny-random-minicpmv-2_6'
    align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **align_with_optimum_cli)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
    model = optimum.intel.openvino.OVModelForVisualCausalLM.from_pretrained(model_id, compile=False, device="CPU", export=True, load_in_8bit=False, trust_remote_code=True)
    openvino_genai.VLM(model.vision_embeddings_model, model.text_embeddings_model, model.lm_model, ov_tokenizer, ov_detokenizer, 'CPU')


if '__main__' == __name__:
    main()
