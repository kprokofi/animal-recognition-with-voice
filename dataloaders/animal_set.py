from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path
import tensorflow as tf

def build_datasets(data_root, ann_root, batch_size_train=128, batch_size_val=128):
    ''' function that build tensorflow dataset '''
    imdb = loadmat(ann_root / 'imdb-animalParts-eye.mat')
    bbox = imdb['bbx']['loc'][0,0].T
    bbox_to_id = imdb['bbx']['imageId'][0][0][0]
    set_ = imdb['bbx']['set'][0][0].reshape(-1)
    train_ids = bbox_to_id[set_ == 1]
    val_ids = bbox_to_id[set_ == 2]
    zip_ = zip(list(imdb['images']['id'][0][0][0]), list(imdb['images']['name'][0][0][0]))
    dict_ = {k : v for k,v in zip_}

    images_train = [data_root + os.sep + dict_[id_][0] for id_ in train_ids if id_ in dict_]
    bbox_train = bbox[set_ == 1]
    images_val = [data_root + os.sep + dict_[id_][0] for id_ in val_ids if id_ in dict_]
    bbox_val = bbox[set_ == 2]

    ds_train = tf.data.Dataset.from_tensor_slices((images_train, bbox_train))
    ds_val = tf.data.Dataset.from_tensor_slices((images_val, bbox_val))

    ds_train = ds_train.map(read_image).map(augment).batch(batch_size_train)
    ds_val = ds_val.map(read_image).map(augment).batch(batch_size_val)

    return ds_train, ds_val

def read_image(image_path, bbox):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    return image, bbox

def augment(image, bbox):
    # do augmentations here
    return image, bbox

def test():
    ds_train, ds_val = build_datasets('data/', 'data/')
    for it, (img, bb) in enumerate(ds_train):
        print(it, img.shape, bb.shape)
        pass

if __name__ == "__main__":
    test()
