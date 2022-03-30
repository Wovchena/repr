import openvino.runtime as ov
import collections
import numpy as np


def main():
    cml = ov.Core().compile_model(r"c:\Users\vzlobin\Downloads\d\public\yolo-v3-tf\FP32\yolo-v3-tf.xml", "CPU", {'CPU_THROUGHPUT_STREAMS': '4'})
    empty_ireqs = [cml.create_infer_request() for _ in range(5)]
    busy_ireqs = collections.deque(maxlen=len(empty_ireqs))

    while True:
        if empty_ireqs:
            ireq = empty_ireqs.pop()
            ireq.start_async({'input_1': np.zeros([1, 416, 416, 3], np.uint8)})
            busy_ireqs.append(ireq)
        else:
            busy_ireqs[0].wait()
            ireq = busy_ireqs.popleft()
            empty_ireqs.append(ireq)


if __name__ == '__main__':
    main()
