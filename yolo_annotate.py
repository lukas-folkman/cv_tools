import os
import shutil
import sys
from ultralytics import YOLO
import itertools
import copy
import cv2
import torch
import numpy as np
from tempfile import TemporaryDirectory
from ultralytics.engine.results import Results

if os.path.exists('utils.py'):
    import utils
    from yolo_predict import yolo_predict
else:
    import cv_tools as utils
    from yolo_utils import yolo_predict


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--output_fn', required=True)
    parser.add_argument('--conf_thr', type=float, default=0)
    args = parser.parse_args()

    frames_fns = sorted([os.path.join(args.img_dir, fn) for fn in os.listdir(args.img_dir) if utils.is_image(fn)])
    dataset = yolo_annotate(model=args.model, img_dir=args.img_dir, frames_fns=frames_fns,
                            conf_thr=args.conf_thr)
    utils.fill_in_areas(dataset['annotations'], fill_in_ids=True, fill_in_iscrowd=True)
    utils.save_json(dataset, args.output_fn)


def yolo_annotate(model, img_dir, frames_fns, conf_thr, labels=None):

    with TemporaryDirectory() as tmp_dir:
        model_labels = YOLO(model).names
        model_labels = [model_labels[class_id] for class_id in range(len(model_labels))]
        if labels is not None:
            assert set(model_labels).issubset(labels), \
                'Model\'s labels must be a subset of the requested labels, ' \
                f'model: {model_labels}, requested: {labels}'
            assert all([model_labels[i] == labels[i] for i in range(len(model_labels))]), \
                'The order of the model\'s labels must match the order of the requested labels, ' \
                f'model: {model_labels}, requested: {labels}'
        else:
            labels = model_labels
        dataset = utils.create_coco_dataset(*frames_fns, cat_names=labels)
        dataset.pop('annotations')
        input_json_fn = os.path.join(tmp_dir, 'input.json')
        utils.save_json(dataset, fn=input_json_fn, only_imgs=True)

        _, predictions = yolo_predict(
            model=model, dataset=(input_json_fn, img_dir), output_dir=tmp_dir,
            threshold=conf_thr, NMS_threshold=0.7, detections_per_image=100
        )
        dataset['annotations'] = predictions
        print(f'{len(predictions)} predicted annotations at confidence threshold of {conf_thr}\n')
        # Useful for debugging:
        # utils.fill_in_areas(dataset['annotations'], fill_in_ids=True, fill_in_iscrowd=True)
        # utils.save_json(dataset, os.path.join(tmp_dir, 'dataset.json'))
    return dataset


if __name__ == '__main__':
    main()
