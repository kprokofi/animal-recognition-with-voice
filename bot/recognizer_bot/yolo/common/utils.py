import numpy as np
import time
import cv2
import colorsys
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, ReLU, Multiply

# Custom objects from backbones package https://github.com/david8862/keras-YOLOv3-model-set/tree/master/common/backbones


def mish(x):
    return x * K.tanh(K.softplus(x))


def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])


def hard_sigmoid(x):
    return ReLU(6.)(x + 3.) * (1. / 6.)


def swish(x):
    """Swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if K.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return K.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * K.sigmoid(x)


def get_custom_objects():
    '''
    form up a custom_objects dict so that the customized
    layer/function call could be correctly parsed when keras
    .h5 model is loading or converting
    '''
    custom_objects_dict = {
        'tf': tf,
        'swish': swish,
        'hard_sigmoid': hard_sigmoid,
        'hard_swish': hard_swish,
        'mish': mish
    }

    return custom_objects_dict


def get_multiscale_list():
    input_shape_list = [(320, 320), (352, 352), (384, 384), (416, 416),
                        (448, 448), (480, 480), (512, 512), (544, 544), (576, 576), (608, 608)]

    return input_shape_list


def resize_anchors(base_anchors, target_shape, base_shape=(416, 416)):
    '''
    original anchor size is clustered from COCO dataset
    under input shape (416,416). We need to resize it to
    our train input shape for better performance
    '''
    return np.around(base_anchors*target_shape[::-1]/base_shape[::-1])


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def get_colors(class_names):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    # Shuffle colors to decorrelate adjacent classes.
    np.random.shuffle(colors)
    np.random.seed(None)  # Reset seed to default.
    return colors


def get_dataset(annotation_file, shuffle=True):
    with open(annotation_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    if shuffle:
        np.random.seed(int(time.time()))
        np.random.shuffle(lines)
        # np.random.seed(None)

    return lines


def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width,
                  y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)

    return image


def draw_boxes(image, boxes, classes, scores, class_names, colors, show_score=True):
    if boxes is None or len(boxes) == 0:
        return image
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = map(int, box)

        class_name = class_names[cls]
        if show_score:
            label = '{} {:.2f}'.format(class_name, score)
        else:
            label = '{}'.format(class_name)
        #print(label, (xmin, ymin), (xmax, ymax))

        # if no color info, use black(0,0,0)
        if colors is None:
            color = (0, 0, 0)
        else:
            color = colors[cls]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
        image = draw_label(image, label, color, (xmin, ymin))

    return image
