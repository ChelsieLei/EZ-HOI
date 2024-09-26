"""
Run Faster R-CNN with ResNet50-FPN on V-COCO

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import json
import torch
import argparse
import torchvision
from tqdm import tqdm

sys.path.append('..')
from vcoco import VCOCO
from pocket.models import fasterrcnn_resnet_fpn

def main(args):
	cache_dir = os.path.join(args.cache_dir, args.partition)
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)

	image_root = dict(
		train='../mscoco2014/train2014',
		val='../mscoco2014/train2014',
		trainval='../mscoco2014/train2014',
		test='../mscoco2014/val2014'
	)
	dataset = VCOCO(
		root=image_root[args.partition],
		anno_file='../instances_vcoco_{}.json'.format(args.partition)
	)

	detector = fasterrcnn_resnet_fpn('resnet50',
		pretrained=True,
		box_score_thresh=args.score_thresh, 
		box_nms_thresh=args.nms_thresh,
		box_detections_per_img=args.num_detections_per_image
	)
	if os.path.exists(args.ckpt_path):
		detector.load_state_dict(torch.load(args.ckpt_path)['model_state_dict'])
		print("Checkpoint loaded from ", args.ckpt_path)
	detector.eval()
	detector.cuda()

	for idx, (image, _) in enumerate(tqdm(dataset)):

		image = torchvision.transforms.functional.to_tensor(image).cuda()
		with torch.no_grad():
			detections = detector([image])[0]

		detections['boxes'] = detections['boxes'].tolist()
		detections['scores'] = detections['scores'].tolist()
		detections['labels'] = detections['labels'].tolist()

		with open(os.path.join(
			cache_dir,
			dataset.filename(idx).replace('jpg', 'json')
		), 'w') as f:
			json.dump(detections, f)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--partition', type=str, default='train')
	parser.add_argument('--data-root', type=str, default='../')
	parser.add_argument('--cache-dir', type=str, default='./')
	parser.add_argument('--nms-thresh', type=float, default=0.5)
	parser.add_argument('--score-thresh', type=float, default=0.05)
	parser.add_argument('--num-detections-per-image', type=int, default=100)
	parser.add_argument('--ckpt-path', type=str, default='',
			help="Path to a checkpoint that contains the weights for a model")
	args = parser.parse_args()

	print(args)
	main(args)
