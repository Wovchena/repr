import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
import numpy as np


def main():
    DIVIDEND = float.fromhex('0x1.b7f1200000000p+0')
    NP_DIVIDEND = np.array([DIVIDEND], np.float32)
    DIVISOR = float.fromhex('0x1.8000000000000p+0')  # 1.5
    NP_DIVISOR = np.array([DIVISOR], np.float32)
    assert DIVIDEND.hex() == NP_DIVIDEND.item().hex()
    assert DIVISOR.hex() == NP_DIVISOR.item().hex()
    param_node = ov.op.Parameter(ov.Type.f32, ov.Shape([1]))
    model = ov.Model(param_node, [param_node])
    ppp = PrePostProcessor(model)
    ppp.input().preprocess().scale(NP_DIVISOR)
    scaler = ov.Core().compile_model(ppp.build(), "CPU")
    scaled_pred = scaler({0: NP_DIVIDEND})[0]
    print((NP_DIVIDEND / NP_DIVISOR).item().hex(), scaled_pred.item().hex())
    assert (NP_DIVIDEND / NP_DIVISOR).item().hex() == scaled_pred.item().hex()


if __name__ == '__main__':
    main()
