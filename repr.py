import cv2
from openvino.model_api.models import ClassificationModel


def simple():
    model = ClassificationModel.create_model("resnet-18-pytorch", device="CPU")
    print(model(cv2.imread("/home/wov/Pictures/dog.jpg")))  # https://storage.openvinotoolkit.org/test_data/images/dog.jpg
    print(model(cv2.imread("/home/wov/Pictures/cat.jpg")))  # https://storage.openvinotoolkit.org/test_data/images/cat.jpg


def infer_sync():
    model = ClassificationModel.create_model("resnet-18-pytorch", device="CPU")
    dict_data, input_meta = model.preprocess(cv2.imread("/home/wov/Pictures/dog.jpg"))
    raw_result = model.inference_adapter.infer_sync(dict_data)
    print(model.postprocess(raw_result, input_meta))
    dict_data, input_meta = model.preprocess(cv2.imread("/home/wov/Pictures/cat.jpg"))
    raw_result = model.inference_adapter.infer_sync(dict_data)
    print(model.postprocess(raw_result, input_meta))


def main():
    simple()
    infer_sync()


if __name__ == "__main__":
    main()
