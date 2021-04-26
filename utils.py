import matplotlib.pyplot as plt

import os
import numpy as np
from six import BytesIO
from PIL import Image
import shutil
import tqdm

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

from dataloaders.animal_set import build_datasets

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path.

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
    """Wrapper function to visualize detections.

    Args:
      image_np: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      figsize: size for the figure.
      image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)

def load_dataset(data_root, annotation_root):
    train_dataset_dict, val_dataset_dict = build_datasets(data_root=data_root,
                                                          ann_root=annotation_root)
    images_names = [x['image'] for x in train_dataset_dict.values()] + [x['image'] for x in val_dataset_dict.values()]
    new_image_names = ['val_original/' + ''.join(x.split('/')[1:]) if str(x).startswith('val') else x for x in images_names]

    for name in tqdm.tqdm(new_image_names):
        dir = './' + '/'.join(name.split('/')[:-1])
        os.makedirs(dir, exist_ok=True)
        src = data_root + '/' + name
        shutil.copyfile(src, name)
