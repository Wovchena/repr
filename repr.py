import openvino.runtime as ov
import sys


def main():
    print('OpenVINO:')
    print(f"{'Build ':.<39} {ov.get_version()}")
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <path_to_model>, <device>')
        return 1
    core = ov.Core()
    model = core.read_model(sys.argv[1])
    compiled_model = core.compile_model(model, sys.argv[2])
    keys = compiled_model.get_property('SUPPORTED_PROPERTIES')
    print("Model:")
    for k in keys:
        if k not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
            print(k)
            print(f'  {k}: {compiled_model.get_property(k)}')


if __name__ == '__main__':
    main()
