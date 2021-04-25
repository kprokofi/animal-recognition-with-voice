from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
import cv2
from pathlib import Path
import tensorflow as tf

def build_datasets(data_root, ann_root, batch_size_train=128, batch_size_val=128):
    '''
    function that build tensorflow dataset
    :return: Train and val datasets in format: dict(id: imgenet_id, dict(image_path, bbox_list, classes_list))
    '''
    imdb = loadmat(ann_root + os.sep + 'imdb-animalParts-eye.mat')
    bbox = imdb['bbx']['loc'][0,0].T
    bbox_to_id = imdb['bbx']['imageId'][0][0][0]
    classes = imdb['bbx']['class'][0][0][0]
    set_ = imdb['bbx']['set'][0][0][0]
    mapped_array_bbox = [(id_,bb) for id_,bb in zip(bbox_to_id, bbox)]
    zip_ = zip(list(imdb['images']['id'][0][0][0]), list(imdb['images']['name'][0][0][0]))
    dict_images = dict()
    for k, v in zip_:
        dict_images[k] = v[0]

    train_dataset_dict = dict()
    val_dataset_dict = dict()

    unique_classes = np.unique(classes)
    unique_classes.sort()
    map_ = {class_: new_class for class_, new_class in zip(unique_classes, range(len(unique_classes)))}

    # get classes
    for id_, bb, cls, s in zip(bbox_to_id, bbox, classes, set_, ):
        cls = map_[cls]
        if s == 1:
            if id_ in train_dataset_dict:
                train_dataset_dict[id_]['bboxes'].append(bb)
                train_dataset_dict[id_]['classes'].append(cls)
            else:
                train_dataset_dict[id_] = dict(image=dict_images[id_],bboxes=[bb],classes=[cls])
        else:
            if id_ in val_dataset_dict:
                val_dataset_dict[id_]['bboxes'].append(bb)
                val_dataset_dict[id_]['classes'].append(cls)
            else:
                val_dataset_dict[id_] = dict(image=dict_images[id_],bboxes=[bb],classes=[cls])

    return train_dataset_dict, val_dataset_dict

#def read_image(dict_object):
#    image = tf.io.read_file(dict_object['image'])
#    image = tf.image.decode_jpeg(image, channels=3)
#    shapes = tf.cast(tf.shape(image), tf.float32)
#    w,h = shapes[0], shapes[1]
#
#    return image, [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h], label

def augment(image, bbox, labels):
    image = tf.image.resize(image, [640,640])
    return image, bbox, labels

def test():
    ds_train, ds_val = build_datasets('/mnt/pai_share/datasets/imagenet', 'data')
    for it, (img, bb, l) in enumerate(ds_train):
        print(it, img, bb, l)
        pass

if __name__ == "__main__":
    test()
