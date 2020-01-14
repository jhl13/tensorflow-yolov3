import os
import argparse
import xml.etree.ElementTree as ET
import glob

def get_ann_paths(dir_path):
    ann_paths = glob.glob(os.path.join(dir_path, '*.xml'))
    return ann_paths

def convert_voc_annotation(data_dir, anno_txt, use_difficult_bbox=True):

    classes = ['爽快生菜', '当冬菇遇上鸡', '米饭', '忘情糯米鸡', '玉米约肉包', '痴心鱼蛋粉', \
        '真心卤蛋', '蒸凤爪', '蒸蛋', '小鲜肉滚粥', '黄汽水', '绿汽水', '奶茶', '小红莓布丁', \
        '情意绵绵粥', '蒸能量花卷', '蒸花蛋', '鸡汤', '油焖茄子', '虫草花牛肉粥', '古惑烧麦', \
        '念念冬菇肉饼', '忘情花生猪手', '淡定瘦肉粉', '蒸米粉', '豆浆', '看气质肠粉', '蒸排骨', \
            '玉米蒸饭', '柔情菜心', '忘不了香汁排骨', '虫草', '香辣排骨', '勃勃生鸡粥', '给力蒸饺', \
                '狠香牛腩粉', '甜在心奶黄包', '蛋蛋布丁', '辣手卤翅', '念念香菇肉饼', '云吞']
    ann_dir = os.path.join(data_dir, "Annotations")
    ann_paths = get_ann_paths(ann_dir)
    print(ann_dir)
    with open(anno_txt, 'a') as f:
        for ann in ann_paths:
            image_name = ann.split('/')[-1][:-4]
            image_path = os.path.join(data_dir, 'JPEGImages', image_name + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_dir, 'Annotations', image_name + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(ann_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/ljh/workspace/datasets/food")
    parser.add_argument("--train_annotation", default="./data/food/food_train.txt")
    parser.add_argument("--test_annotation",  default="./data/food/food_test.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)

    num1 = convert_voc_annotation(os.path.join(flags.data_path, 'train/VOCdevkit'), flags.train_annotation, False)
    num2 = convert_voc_annotation(os.path.join(flags.data_path, 'test/VOCdevkit'), flags.test_annotation, False)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' %(num1, num2))
