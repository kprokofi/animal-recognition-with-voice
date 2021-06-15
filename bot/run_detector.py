
import os
import time
import numpy as np
from PIL import Image
from keras.models import load_model

from recognizer_bot.yolo.common.yolo_postprocess_np import yolo3_postprocess_np
from recognizer_bot.yolo.common.data_utils import preprocess_image
from recognizer_bot.yolo.common.utils import get_custom_objects, get_classes, get_anchors, get_colors, draw_boxes

MODEL_PATH = '/home/friday/HSE/animal-recognition-with-voice/bot/static/models/yolo3_mobilenet_lite_saved_model.h5'

config = {
    "pruning_model": False,
    "anchors_path": '/home/friday/HSE/animal-recognition-with-voice/bot/yolo4_anchors.txt',
    "classes_path": '/home/friday/HSE/animal-recognition-with-voice/bot/yolo_classes.txt',
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
    return image_array, out_boxes, out_classnames, out_scores


def detect_img(model, config):
    while True:
        # /home/friday/HSE/animal-recognition-with-voice/yolo_example/horse.jpg
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except Exception:
            print('Open Error! Try again!')
            continue
        else:
            image_array, out_boxes, out_classnames, out_scores = detect_image(
                model, image, config)
            print(out_classnames)
            print(out_scores)
            print(out_boxes)

            for box in out_boxes:
                xmin, ymin, xmax, ymax = map(int, box)
                Image.fromarray(image_array[ymin:ymax, xmin:xmax]).show()

            Image.fromarray(image_array).show()


if os.path.isfile(MODEL_PATH):
    print('Exists!')
model = load(MODEL_PATH)
detect_img(model, config)
