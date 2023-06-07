import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
import numpy as np


def main():
    # identity
    input_shape = [1, 20, 20, 3]
    param_node = ov.op.Parameter(ov.Type.f32, ov.Shape(input_shape))
    core = ov.Core()
    model = ov.Model(param_node, [param_node])
    identity = core.compile_model(model, "CPU")
    inp = np.random.random(input_shape) * 224.0 + np.random.random(input_shape)
    pred = next(iter(identity({0: inp}).values()))
    print(np.abs(inp - pred).max())  # non zero
    # scale
    aligned_inp = pred  # Assume openvino currupts input always in the same way
    ppp = PrePostProcessor(model)
    ppp.input().preprocess().scale([255.0])
    model = ppp.build()
    scaler = core.compile_model(model, "CPU")
    scaled_pred = next(iter(scaler({0: inp}).values()))
    scaled_ref = aligned_inp / 255.0
    print(np.abs(scaled_pred - scaled_ref).max())  # non zero


if __name__ == '__main__':
    main()
