import transformers
import openvino
import openvino_tokenizers


def main():
    input_ids = openvino.op.Constant(openvino.Type.i64, openvino.Shape([0, 0]), []).output(0)
    input_ids.get_tensor().set_names({"input_ids"})
    attention_mask = openvino.op.Constant(openvino.Type.i64, openvino.Shape([0, 0]), []).output(0)
    attention_mask.get_tensor().set_names({"attention_mask"})
    model = openvino.Model(
        [openvino.op.Result(input_ids), openvino.op.Result(attention_mask)],
        [openvino.op.Parameter(openvino.Type.string, openvino.Shape([1]))]
    )
    core = openvino.Core()
    core.compile_model(model, "CPU")
    tokenizer = transformers.AutoProcessor.from_pretrained("katuni4ka/tiny-random-llava-next").tokenizer
    ov_tokenizer = openvino_tokenizers.convert_tokenizer(tokenizer)
    core.compile_model(ov_tokenizer, "CPU")


if __name__ == '__main__':
    main()
