
import os
import time
import numpy as np
from PIL import Image
from keras.models import load_model

from yolo.common.yolo_postprocess_np import yolo3_postprocess_np
from yolo.common.data_utils import preprocess_image
from yolo.common.utils import get_custom_objects, get_classes, get_anchors, get_colors, draw_boxes

MODEL_PATH = '/home/friday/HSE/animal-recognition-with-voice/bot/static/models/yolo3_mobilenet_lite_saved_model.h5'

config = {
    "pruning_model": False,
    "anchors_path": os.path.join('tiny_yolo3_anchors.txt'),
    "classes_path": os.path.join('yolo_classes.txt'),
    "score": 0.1,
    "iou": 0.4,
    "model_image_size": (416, 416),
    "elim_grid_sense": False,
    "gpu_num": 1,
}


def load(model_path: str):
    custom_object_dict = get_custom_objects()
    model = load_model(model_path, compile=False,
                       custom_objects=custom_object_dict)
    return model


def detect_image(model, image, config: dict):
    model_image_size = config['model_image_size']
    class_names = get_classes(config['classes_path'])
    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'

    image_data = preprocess_image(image, model_image_size)
    # origin image shape, in (height, width) format
    image_shape = tuple(reversed(image.size))

    start = time.time()
    out_boxes, out_classes, out_scores = yolo3_postprocess_np(
        model.predict(image_data),
        image_shape,
        get_anchors(config['anchors_path']),
        len(class_names),
        model_image_size,
        max_boxes=100,
        confidence=config['score'],
        iou_threshold=config['iou'],
        elim_grid_sense=config['elim_grid_sense']
    )
    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    end = time.time()
    print("Inference time: {:.8f}s".format(end - start))

    # draw result on input image
    image_array = np.array(image, dtype='uint8')
    image_array = draw_boxes(
        image_array, out_boxes, out_classes, out_scores, class_names, get_colors(class_names))

    out_classnames = [class_names[c] for c in out_classes]
    return Image.fromarray(image_array), out_boxes, out_classnames, out_scores


def detect_img(model):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except Exception:
            print('Open Error! Try again!')
            continue
        else:
            r_image, _, _, _ = detect_image(model, image)
            r_image.show()


if os.path.isfile(MODEL_PATH):
    print('Exists!')
model = load(MODEL_PATH)

data = np.random.randint(0, 256, size=(3, 128, 128, 3)).astype("float32")

processed_data = model.predict(data)
print(processed_data)


# dataset = keras.preprocessing.image_dataset_from_directory(
#     'path/to/main_directory', batch_size=64, image_size=(200, 200))

# # For demonstration, iterate over the batches yielded by the dataset.
# for data, labels in dataset:
#     print(data.shape)  # (64, 200, 200, 3)
#     print(data.dtype)  # float32
#     print(labels.shape)  # (64,)
#     print(labels.dtype)  # int32


# # Example image data, with values in the [0, 255] range
# training_data = np.random.randint(
#     0, 256, size=(64, 200, 200, 3)).astype("float32")

# normalizer = Normalization(axis=-1)
# normalizer.adapt(training_data)

# normalized_data = normalizer(training_data)
# print("var: %.4f" % np.var(normalized_data))
# print("mean: %.4f" % np.mean(normalized_data))


# # Example image data, with values in the [0, 255] range
# training_data = np.random.randint(
#     0, 256, size=(64, 200, 200, 3)).astype("float32")

# cropper = CenterCrop(height=150, width=150)
# scaler = Rescaling(scale=1.0 / 255)

# output_data = scaler(cropper(training_data))
# print("shape:", output_data.shape)
# print("min:", np.min(output_data))
# print("max:", np.max(output_data))


# new_model = keras.models.load_model('path_to_my_model.h5')
