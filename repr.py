import transformers
import openvino
import openvino_tokenizers
import subprocess


def main():
    tokenizer_init_kwargs = {"padding_side": "left", "truncation_side": "left"}
    tokenizer = transformers.AutoTokenizer.from_pretrained("katuni4ka/tiny-random-llava-next", padding_side="left", truncation_side="left")
    ov_tokenizer = openvino_tokenizers.convert_tokenizer(tokenizer)
    prompt = ["1" + "<image>"*2048 + "1"]
    core = openvino.Core()
    openvino.save_model(ov_tokenizer, "code.xml")
    code = core.compile_model("code.xml", "CPU").create_infer_request()
    code.set_input_tensor(openvino.Tensor(prompt))
    code.infer()
    code_res = code.get_tensor("input_ids").data
    subprocess.run(["optimum-cli", "export", "openvino", "--trust-remote-code", "--model", "katuni4ka/tiny-random-llava-next", "./", "--weight-format", "fp32", "--task", "image-text-to-text"], check=True)
    cli = core.compile_model("openvino_tokenizer.xml", "CPU").create_infer_request()
    cli.set_input_tensor(openvino.Tensor(prompt))
    cli.infer()
    cli_res = cli.get_tensor("input_ids").data
    print(code_res, code_res.shape)
    print(cli_res, cli_res.shape)
    assert (code_res == cli_res).all()


if __name__ == '__main__':
    main()
