#! /usr/bin/env python
# coding=utf-8

import os
import glob
import numpy as np
from shutil import copyfile

def get_img_paths(dir_path):
    img_paths = glob.glob(os.path.join(dir_path, '*.jpeg'))
    img_paths.extend(glob.glob(os.path.join(dir_path, '*.png')))
    img_paths.extend(glob.glob(os.path.join(dir_path, '*.jpg')))
    return img_paths

def get_ann_paths(dir_path):
    ann_paths = glob.glob(os.path.join(dir_path, '*.xml'))
    return ann_paths

if __name__ == '__main__':
    image_dir = "/home/luo13/workspace/datasets/detection/food/VOC2007/JPEGImages"
    ann_dir = "/home/luo13/workspace/datasets/detection/food/VOC2007/Annotations"

    train_dir = "/home/luo13/workspace/datasets/detection/food/train"
    test_dir = "/home/luo13/workspace/datasets/detection/food/test"
    
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    train_image_dir = os.path.join(train_dir, "image")
    train_ann_dir = os.path.join(train_dir, "ann")
    test_image_dir = os.path.join(test_dir, "image")
    test_ann_dir = os.path.join(test_dir, "ann")
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_ann_dir):
        os.mkdir(train_ann_dir)
    if not os.path.exists(test_image_dir):
        os.mkdir(test_image_dir)
    if not os.path.exists(test_ann_dir):
        os.mkdir(test_ann_dir)
    
    img_paths = get_img_paths(image_dir)
    ann_paths = get_ann_paths(ann_dir)

    assert len(img_paths) == len(ann_paths)
    print("Length of images are equal to annotations.")
    
    np.random.shuffle(img_paths)

    train_size = int(0.7 * len(img_paths))
    test_size = len(img_paths) - train_size
    print("train_size: ", train_size, "\n", "test_size: ", test_size)

    for image_index, image_path in enumerate(img_paths):
        if image_index < train_size:
            src_image_dir = train_image_dir
            scr_ann_dir = train_ann_dir
        else:
            src_image_dir = test_image_dir
            scr_ann_dir = test_ann_dir
        image_name = image_path.split('/')[-1][:-4]
        xml_path = os.path.join(ann_dir, image_path.split('/')[-1][:-4] + '.xml')
        target_image_path = os.path.join(src_image_dir, image_name + '.jpg')
        target_ann_path = os.path.join(scr_ann_dir, image_name + '.xml')
        copyfile(image_path, target_image_path)
        copyfile(xml_path, target_ann_path)
