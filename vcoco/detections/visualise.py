"""
Visualise the detection results

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import os
import sys
import json
import argparse
import numpy as np

import torch
from PIL import ImageDraw
from torchvision.ops import nms

sys.path.append('..')
from vcoco import VCOCO

def visualize(args):
    image_dir = dict(
        train='mscoco2014/train2014',
        val='mscoco2014/train2014',
        trainval='mscoco2014/train2014',
        test='mscoco2014/val2014'
    )
    dataset = VCOCO(
        root=os.path.join(args.data_root, image_dir[args.partition]),
        anno_file=os.path.join(
            args.data_root,
            'instances_vcoco_{}.json'.format(args.partition)
    ))
    image, _ = dataset[args.image_idx]
    image_name = dataset.filename(args.image_idx)
    detection_path = os.path.join(
        args.partition,
        image_name.replace('.jpg', '.json')
    )
    with open(detection_path, 'r') as f:
        detections = json.load(f)
    # Remove low-scoring boxes
    box_score_thresh = args.box_score_thresh
    boxes = np.asarray(detections['boxes'])
    scores = np.asarray(detections['scores'])
    keep_idx = np.where(scores >= box_score_thresh)[0]
    boxes = boxes[keep_idx, :]
    scores = scores[keep_idx]
    # Perform NMS
    keep_idx = nms(
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
        args.nms_thresh
    )
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    # Draw boxes
    canvas = ImageDraw.Draw(image)
    for idx in range(boxes.shape[0]):
        coords = boxes[idx, :].tolist()
        canvas.rectangle(coords)
        canvas.text(coords[:2], str(scores[idx])[:4])

    image.show()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-idx', type=int, default=0)
    parser.add_argument('--data-root', type=str, default='../')
    parser.add_argument('--partition', type=str, default='trainval')
    parser.add_argument('--box-score-thresh', type=float, default=0.2)
    parser.add_argument('--nms-thresh', type=float, default=0.5)
    args = parser.parse_args()

    visualize(args)
