import time
import numpy as np
from keras.models import load_model

from yolo.common.yolo_postprocess_np import yolo3_postprocess_np
from yolo.common.data_utils import preprocess_image
from yolo.common.utils import get_custom_objects, get_classes, get_anchors, get_colors, draw_boxes


class Detector():
    def __init__(self, model_path: str, config: dict):
        self.model = self.load(model_path)
        self.config = config

    @staticmethod
    def load(model_path: str):
        custom_object_dict = get_custom_objects()
        model = load_model(model_path, compile=False,
                           custom_objects=custom_object_dict)
        return model


def detect_image(self, image):
    model_image_size = self.config['model_image_size']
    class_names = get_classes(self.config['classes_path'])
    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'

    image_data = preprocess_image(image, model_image_size)
    # origin image shape, in (height, width) format
    image_shape = tuple(reversed(image.size))

    start = time.time()
    out_boxes, out_classes, out_scores = yolo3_postprocess_np(
        self.model.predict(image_data),
        image_shape,
        get_anchors(self.config['anchors_path']),
        len(class_names),
        model_image_size,
        max_boxes=100,
        confidence=self.config['score'],
        iou_threshold=self.config['iou'],
        elim_grid_sense=self.config['elim_grid_sense']
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
