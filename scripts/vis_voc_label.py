#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import glob
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

def get_img_paths(dir_path):
    img_paths = glob.glob(os.path.join(dir_path, '*.jpeg'))
    img_paths.extend(glob.glob(os.path.join(dir_path, '*.png')))
    img_paths.extend(glob.glob(os.path.join(dir_path, '*.jpg')))
    return img_paths

def get_ann_paths(dir_path):
    ann_paths = glob.glob(os.path.join(dir_path, '*.xml'))
    return ann_paths

if __name__ == "__main__":
    image_dir = "/home/luo13/workspace/datasets/detection/food/VOC2007/JPEGImages"
    ann_dir = "/home/luo13/workspace/datasets/detection/food/VOC2007/Annotations"

    img_paths = get_img_paths(image_dir)
    ann_paths = get_ann_paths(ann_dir)

    assert len(img_paths) == len(ann_paths)
    print("Length of images are equal to annotations.")
    print("Visualizing.")
    
    food_names = []
    for image_index, image_path in enumerate(img_paths):
        image = cv2.imread(image_path)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # print ("%d/%d Drawing %s"%(image_index + 1, len(img_paths), image_path))
        xml_path = os.path.join(ann_dir, image_path.split('/')[-1][:-4] + '.xml')
        xml_tree = ET.parse(xml_path)
        root = xml_tree.getroot()
        all_object = []
        for ann in root.iter("annotation"):
            for ann_object in ann.iter("object"):
                object_dict = {"name": "none", "location": []}
                location = []
                for name in ann_object.iter("name"):
                    object_dict["name"] = name.text
                    if name.text == "虫草":
                        print (image_path)
                    if name.text not in food_names:
                        food_names.append(name.text)
                for bndbox in ann_object.iter("bndbox"):
                    for xmin in bndbox.iter("xmin"):
                        location.append(int(xmin.text))
                    for ymin in bndbox.iter("ymin"):
                        location.append(int(ymin.text))
                    for xmax in bndbox.iter("xmax"):
                        location.append(int(xmax.text))
                    for ymax in bndbox.iter("ymax"):
                        location.append(int(ymax.text))
                    object_dict["location"] = location
                all_object.append(object_dict)
                image = cv2.rectangle(image, (location[0], location[1]), (location[2], location[3]), (255, 0, 0), 2)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
    print (food_names)