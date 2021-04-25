from scipy.io import loadmat
import numpy as np
import os
import cv2
from pathlib import Path
import tensorflow as tf

def build_datasets(data_root, ann_root, batch_size_train=128, batch_size_val=128):
    ''' function that build tensorflow dataset '''
    imdb = loadmat(ann_root + os.sep + 'imdb-animalParts-eye.mat')
    bbox = imdb['bbx']['loc'][0,0].T
    bbox_to_id = imdb['bbx']['imageId'][0][0][0]
    set_ = imdb['bbx']['set'][0][0][0]
    classes = imdb['bbx']['class'][0][0][0]
    train_classes = classes[set_ == 1]
    val_classes = classes[set_ == 2]
    train_ids = bbox_to_id[set_ == 1]
    val_ids = bbox_to_id[set_ == 2]
    zip_ = zip(list(imdb['images']['id'][0][0][0]), list(imdb['images']['name'][0][0][0]))
    dict_ = dict()
    for k,v in zip_:
        if v[0].startswith('val'):
            new_file = 'val_original/' + v[0][4:]
        else:
            new_file = v[0]
        dict_[k] = new_file
    # dict_ = {k : v[0] for k,v in zip_}
    images_train = [data_root + os.sep + dict_[id_] for id_ in train_ids if id_ in dict_]
    bbox_train = bbox[set_ == 1]
    images_val = [data_root + os.sep + dict_[id_] for id_ in val_ids if id_ in dict_]
    bbox_val = bbox[set_ == 2]

    ds_train = tf.data.Dataset.from_tensor_slices((images_train, bbox_train, train_classes))
    ds_val = tf.data.Dataset.from_tensor_slices((images_val, bbox_val, val_classes))

    ds_train = ds_train.map(read_image).map(augment).batch(batch_size_train)
    ds_val = ds_val.map(read_image).map(augment).batch(batch_size_val)

    return ds_train, ds_val

def read_image(image_path, bbox, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    w = tf.shape(image)[0]
    shapes = tf.cast(tf.shape(image), tf.float32)
    w,h = shapes[0], shapes[1]
    return image, [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h], label

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
