#!/usr/bin/env python

"""
Creates COCO-like annotations from annotations created with PyLabelMe
@author: Bernhard Bermeitinger
"""

import argparse
import glob
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple
import shutil
import cv2
import pycocotools.mask as maskUtils
from sklearn.model_selection import train_test_split
exclude=['bandera_s','adler_stil','oun2','hitler','Hitlerbart','bandera']
exclude_files=[]
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(filename)-12s: %(message).1000s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

COCO_META = {
    "info": {
        "date_created": "2018-02-08",
        "version": "1",
        "description": "Bandera Corpus, Version v1",
        "contributor": "PACE",
        "url": "",
        "year": 2018
    },
    "licenses": [
        {
            "name": "NONE",
            "url": "Don't distribute",
            "id": 1
        }
    ]
}

ID_COUNTER = {
    'image': 0,
    'annotation': 0
}


def convert_annotation_and_image(annotation_file, image_file, label_map) -> Tuple[List[Dict], Dict]:
    global exclude
    assert os.path.splitext(os.path.split(annotation_file)[1])[0] == os.path.splitext(os.path.split(image_file)[1])[
        0]
    image_information = {
        # hard coded
        'license': 1,
        'coco_url': "",
        'flickr_url': "",

        # generated
        'id': ID_COUNTER['image'],
        'file_name': os.path.split(image_file)[1],
        'date_captured': datetime.fromtimestamp(
            os.path.getmtime(image_file)
        ).strftime('%Y-%m-%d %H:%M:%S')
    }
    if not os.path.isfile(image_file):
        print('problem mit', image_file)
    print(image_file)
    image = cv2.imread(image_file)
    image_information['height'] = image.shape[0]
    image_information['width'] = image.shape[1]


    with open(annotation_file, 'r') as fi:
        source_annotation = json.load(fi)
    assert 'shapes' in source_annotation, "annotation does not have a 'shapes'"

    target_annotations = []
    for annotation in source_annotation['shapes']:
        if annotation['label'] not in exclude:
            polygon = [[point for points in annotation['points'] for point in points]]
            rle = maskUtils.merge(maskUtils.frPyObjects(polygon, image_information['height'], image_information['width']))
            target_annotation = {
                'id': ID_COUNTER['annotation'],
                'image_id': ID_COUNTER['image'],
                'category_id': label_map[annotation['label']],
                'bbox': list(maskUtils.toBbox(rle)),
                'area': int(maskUtils.area(rle)),
                'segmentation': polygon,
                'iscrowd': 0
            }
            assert len(target_annotation['bbox']) == 4, "bounding box is wrong"
            assert target_annotation['area'] > 10, "area is too small"
            target_annotations.append(target_annotation)

            ID_COUNTER['annotation'] += 1
    ID_COUNTER['image'] += 1

    return target_annotations, image_information


def get_annotation_and_image_files(source_annotation_folder, source_image_folder):
    #print(os.path.join(source_annotation_folder, "*.json"))
    source_annotations = list(glob.glob(os.path.join(source_annotation_folder, "*.json")))
    source_images = list(glob.iglob(os.path.join(source_image_folder, "*")))

    if len(source_annotations) != len(source_images):
        log.warning("The amount of JSON files doesn't match the amount of images: %s:%s",
                    len(source_annotations), len(source_images))

    annotation_file_names = [os.path.splitext(os.path.split(p)[1])[0] for p in source_annotations]
    image_file_names = [os.path.splitext(os.path.split(p)[1])[0] for p in source_images]
    #print(annotation_file_names)
    #print(image_file_names)
    missing_annotations = []
    missing_images = []

    for image_file_name in image_file_names:
        if image_file_name not in annotation_file_names:
            missing_annotations.append(image_file_name)
    for annotation_file_name in annotation_file_names:
        if annotation_file_name not in image_file_names:
            missing_images.append(annotation_file_name)

    if len(missing_annotations) > 0:
        log.warning("There are %s annotation files missing.", len(missing_annotations))
        log.debug("Missing annotations: %s", missing_annotations)
    if len(missing_images) > 0:
        log.warning("There are %s images missing.", len(missing_images))
        log.debug("Missing images: %s", missing_images)

    annotation_file_names = sorted([p for p in annotation_file_names
                                    if p not in missing_annotations and p not in missing_images])
    image_file_names = sorted([p for p in image_file_names
                               if p not in missing_images and p not in missing_annotations])

    log.info("There are %s annotation files", len(annotation_file_names))
    log.info("There are %s images", len(image_file_names))

    #assert len(annotation_file_names) == len(image_file_names), "The lengths are still not the same"

    for a, i in zip(annotation_file_names, image_file_names):
        print(a,i)
        assert a == i, "Mismatch between annotation file name and image file name"

    annotation_files = []
    image_files = []

    # align now filtered and sorted file names with source folder again
    for other_annotation_file in source_annotations:
        path, file = os.path.split(other_annotation_file)
        if os.path.splitext(file)[0] in annotation_file_names:
            annotation_files.append(other_annotation_file)
    for other_image_file in source_images:
        path, file = os.path.split(other_image_file)
        if os.path.splitext(file)[0] in image_file_names:
            image_files.append(other_image_file)

    annotation_files = sorted(annotation_files, key=lambda x: os.path.splitext(os.path.split(x)[1])[0])
    image_files = sorted(image_files, key=lambda x: os.path.splitext(os.path.split(x)[1])[0])

    return annotation_files, image_files


def get_label_map(files: List[str],source_image_folder: List[str],output_folder) -> Dict[str, int]:
    global exclude,exclude_files
    label_map: Dict[str, int] = {}
    annotation_files,image_files=[],[]
    for file in files:
        try:
            with open(file, 'r') as fi:
                annotation = json.load(fi)
        except:
            print('could not load file:', file)
        for shape in annotation['shapes']:
            if shape['label'] not in exclude:
                if shape['label'] not in label_map:
                    label_map[shape['label']] = len(label_map)+100
                if shape['label'] == 'ss_rune':
                    print('_______________________________________________',file)
                imagefile=os.path.splitext(os.path.split(file)[1])[0]
                if not file in annotation_files and file not in exclude_files:
                    annotation_files.append(file)
                    image_files.append(glob.glob(os.path.join(source_image_folder,imagefile+ ".*"))[0])
                    if not os.path.exists(os.path.join(output_folder,shape['label'])) and not os.path.isdir(os.path.join(output_folder,shape['label'])):
                        os.makedirs(os.path.join(output_folder,shape['label']))
                shutil.copy(glob.glob(os.path.join(source_image_folder,imagefile+ ".*"))[0], os.path.join(os.path.join(output_folder,shape['label']), '.'))

    print('image_files:',len(image_files))
    print('annotation_files:',len(annotation_files))
    print('Labels:',len(label_map))
    return [label_map,image_files,annotation_files]


def _main(source_annotation_folder, source_image_folder, output_folder):
    annotation_files, image_files = get_annotation_and_image_files(
        source_annotation_folder, source_image_folder)
    #print(annotation_files)
    results = get_label_map(annotation_files,source_image_folder,output_folder)
    label_map: Dict[str, int]=results[0]
    annotation_files=results[2]
    image_files=results[1]
    train_annotations, train_images = [], []
    test_annotations, test_images = [], []
    val_annotations, val_images = [], []
    annotations_train, annotations_test, images_train, images_test = train_test_split(annotation_files, image_files,train_size=0.7,test_size=0.3)
    annotations_val, annotations_test, images_val, images_test = train_test_split(annotations_test, images_test,
                                                                                      train_size=0.5, test_size=0.5)
    for annotation_file, image_file in zip(annotations_train, images_train):
        converted_annotations, converted_image = convert_annotation_and_image(annotation_file, image_file, label_map)
        train_annotations.extend(converted_annotations)
        train_images.append(converted_image)

    for annotation_file, image_file in zip(annotations_test, images_test):
        converted_annotations, converted_image = convert_annotation_and_image(annotation_file, image_file, label_map)
        test_annotations.extend(converted_annotations)
        test_images.append(converted_image)
    for annotation_file, image_file in zip(annotations_val, images_val):
        converted_annotations, converted_image = convert_annotation_and_image(annotation_file, image_file, label_map)
        val_annotations.extend(converted_annotations)
        val_images.append(converted_image)
    stat={}
    for ann in test_annotations:
        if ann['category_id'] in stat.keys():
            stat[ann['category_id']]=stat[ann['category_id']]+1
        else:
            stat[ann['category_id']] =1
    print('Anzahl der Annotationen Pro Kategorie in der test:')
    for key in stat.keys():
        print(key,':',stat[key])
    stat={}
    for ann in val_annotations:
        if ann['category_id'] in stat.keys():
            stat[ann['category_id']]=stat[ann['category_id']]+1
        else:
            stat[ann['category_id']] =1
    print('Anzahl der Annotationen Pro Kategorie in der Val:')
    for key in stat.keys():
        print(key,':',stat[key])
    stat={}
    for ann in train_annotations:
        if ann['category_id'] in stat.keys():
            stat[ann['category_id']]=stat[ann['category_id']]+1
        else:
            stat[ann['category_id']] =1
    print('Anzahl der Annotationen Pro Kategorie in der train:')
    for key in stat.keys():
        print(key,':',stat[key])
    print('keys:',label_map)
    dirpath = os.path.join(os.getcwd(),output_folder, 'val_images')
    print(dirpath)
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(os.path.join(dirpath))
    for testimage in images_val:
        #print(os.path.split(testimage)[1])
        shutil.copy(testimage, os.path.join(dirpath,'.'))
    dirpath = os.path.join(os.getcwd(),output_folder, 'test_images')
    print(dirpath)
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(os.path.join(dirpath))
    for testimage in images_test:
        #print(os.path.split(testimage)[1])
        shutil.copy(testimage, os.path.join(dirpath,'.'))
    dirpath = os.path.join(os.getcwd(),output_folder, 'train_images')
    print(dirpath)
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(os.path.join(dirpath))
    for testimage in images_train:
        #print(os.path.split(testimage)[1])
        shutil.copy(testimage, os.path.join(dirpath,'.'))

    finished_categories = [{
        'id': label_id,
        'name': label_name,
        'supercategory': 'bandera'
    } for label_name, label_id in label_map.items()]

    train = deepcopy(COCO_META)
    train.update({
        'annotations': train_annotations,
        'images': train_images,
        'categories': finished_categories
    })
    test = deepcopy(COCO_META)
    test.update({
        'annotations': test_annotations,
        'images': test_images,
        'categories': finished_categories
    })
    val = deepcopy(COCO_META)
    val.update({
        'annotations': val_annotations,
        'images': val_images,
        'categories': finished_categories
    })
    with open(os.path.join(output_folder, "train.json"), 'w') as fo:
        log.info("Will store train JSON in %s", os.path.join(output_folder, "train.json"))
        json.dump(train, fo, indent=2)
    with open(os.path.join(output_folder, "val.json"), 'w') as fo:
        log.info("Will store test JSON in %s", os.path.join(output_folder, "val.json"))
        json.dump(val, fo, indent=2)
    with open(os.path.join(output_folder, "test.json"), 'w') as fo:
        log.info("Will store test JSON in %s", os.path.join(output_folder, "test.json"))
        json.dump(test, fo, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Create COCO-like annotations from PyLabelMe annotations')

    parser.add_argument(
        "--jsons",
        dest="jsons",
        help="JSON files created by PyLabelMe",
        required=True,
        type=str,
        default=None
    )

    parser.add_argument(
        "--images",
        dest="images",
        help="Folder that contains annotated images",
        required=True,
        type=str,
        default=None
    )

    parser.add_argument(
        "--output",
        dest="output",
        help="Output folder, will be removed and/or overwritten",
        type=str,
        default=None
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log.info("Called with arguments:")
    log.info(args)

    if args.jsons is not None:
        jsons_path = os.path.normpath(os.path.realpath(args.jsons))
        log.info("Source JSON files are used from: %s", jsons_path)
        if not os.path.isdir(jsons_path):
            log.error("The path for the source images is not a directory: %s", jsons_path)
            sys.exit(1)
    else:
        log.error("You have to give the --jsons argument.")
        sys.exit(1)

    if args.images is not None:
        images_path = os.path.normpath(os.path.realpath(args.images))
        log.info("Source images are used from: %s", images_path)
        if not os.path.isdir(images_path):
            log.error("The path for the source images is not a directory: %s", images_path)
            sys.exit(1)
    else:
        log.error("You have to give the --images argument.")
        sys.exit(1)

    if args.output is None:
        log.warning("You didn't give an --output folder, will use current directory.")
        output_folder = os.getcwd()
    else:
        output_folder = os.path.normpath(os.path.realpath(args.output))
        log.info("Will use following output folder: %s", output_folder)

    _main(jsons_path, images_path, output_folder)

    from pycocotools import coco


    train_data_set = coco.COCO(os.path.join(output_folder, "train.json"))
    test_data_set = coco.COCO(os.path.join(output_folder, "test.json"))

    val_data_set = coco.COCO(os.path.join(output_folder, "val.json"))

    print(train_data_set.info())
    print(test_data_set.info())
    print(val_data_set.info())
