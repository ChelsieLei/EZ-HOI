"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from code import interact
from fileinput import filename
from locale import normalize
import os
import torch
import pickle
import numpy as np
import scipy.io as sio
import json

from torchvision.transforms import Resize, CenterCrop

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet
from hico_text_label import hico_unseen_index, HOI_TO_AO, ACT_IDX_TO_ACT_NAME, obj_to_name, OBJ_IDX_TO_COCO_ID, hico_text_label
from vcoco_text_label import MAP_AO_TO_HOI_COCO
import sys
sys.path.append('../pocket/pocket')
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

import sys
sys.path.append('detr')
import detr.datasets.transforms_clip as T
import pdb
import copy 
import pickle
import torch.nn.functional as F
import clip
from util import box_ops
from PIL import Image
from hicodet.static_hico import HICO_INTERACTIONS
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def custom_collate(batch):
    images = []
    targets = []
    # images_clip = []
    
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
        
        # images_clip.append(im_clip)
    return images, targets

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, clip_model_name, zero_shot=False, zs_type='rare_first', num_classes=600, detr_backbone="R50"): ##ViT-B/16, ViT-L/14@336px
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        assert clip_model_name in ['ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
        self.clip_model_name = clip_model_name
        if self.clip_model_name == 'ViT-B/16' or  self.clip_model_name == 'ViT-B/32':
            self.clip_input_resolution = 224
        elif self.clip_model_name == 'ViT-L/14@336px':
            self.clip_input_resolution = 336
        
        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            # pdb.set_trace()
        # add clip normalization
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
            ])
            self.clip_transforms = T.Compose([
                T.IResize([self.clip_input_resolution,self.clip_input_resolution]),
            ])
        else:   
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
            ])
            self.clip_transforms = T.Compose([
                T.IResize([self.clip_input_resolution,self.clip_input_resolution]),
            ])
        
        self.partition = partition
        self.name = name
        self.count=0
        self.zero_shot = zero_shot
        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_type = zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]

        device = "cuda"
        # _, self.process = clip.load('ViT-B/16', device=device)
        print(self.clip_model_name)
        _, self.process = clip.load(self.clip_model_name, device=device)

        self.keep = [i for i in range(len(self.dataset))]

        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_keep = []
            self.remain_hoi_idx = [i for i in np.arange(600) if i not in self.filtered_hoi_idx]

            for i in self.keep:
                (image, target), filename = self.dataset[i]
                # if 1 in target['hoi']:
                #     pdb.set_trace()
                mutual_hoi = set(self.remain_hoi_idx) & set([_h.item() for _h in target['hoi']])
                if len(mutual_hoi) != 0:
                    self.zs_keep.append(i)
            self.keep = self.zs_keep

            self.zs_object_to_target = [[] for _ in range(self.dataset.num_object_cls)]
            if num_classes == 600:
                for corr in self.dataset.class_corr:
                    if corr[0] not in self.filtered_hoi_idx:
                        self.zs_object_to_target[corr[1]].append(corr[0])
            else:
                for corr in self.dataset.class_corr:
                    if corr[0] not in self.filtered_hoi_idx:
                        self.zs_object_to_target[corr[1]].append(corr[2])
        
    def __len__(self):
        return len(self.keep)
        return len(self.dataset)

    # train detr with roi
    def __getitem__(self, i):
        (image, target), filename = self.dataset[self.keep[i]]
        # (image, target), filename = self.dataset[i]
        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            _boxes_h, _boxes_o, _hoi, _object, _verb = [], [], [], [], []
            for j, hoi in enumerate(target['hoi']):
                if hoi in self.filtered_hoi_idx:
                    # pdb.set_trace()
                    continue
                _boxes_h.append(target['boxes_h'][j])
                _boxes_o.append(target['boxes_o'][j])
                _hoi.append(target['hoi'][j])
                _object.append(target['object'][j])
                _verb.append(target['verb'][j])           
            target['boxes_h'] = torch.stack(_boxes_h)
            target['boxes_o'] = torch.stack(_boxes_o)
            target['hoi'] = torch.stack(_hoi)
            target['object'] = torch.stack(_object)
            target['verb'] = torch.stack(_verb)
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w])

        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')
            ## TODO add target['hoi']
        
        image, target = self.transforms(image, target)
        image_clip, target = self.clip_transforms(image, target)  
        image, _ = self.normalize(image, None)
        image_clip, target = self.normalize(image_clip, target)
        target['filename'] = filename

        return (image,image_clip), target
        image_0, target_0 = self.transforms[0](image, target)
        image, _ = self.transforms[1](image_0, None)
        # pdb.set_trace()
        target_0['valid_size'] = torch.as_tensor(image.shape[-2:])
        # pdb.set_trace()
        image_clip, target = self.transforms[3](image_0, target_0) # resize 
        image_clip, target = self.transforms[2](image_clip, target) # normlize
        if image_0.size[-1] >1344 or image_0.size[-2] >1344:print(image_0.size)
        target['filename'] = filename
        return (image,image_clip), target

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def get_region_proposals(self, results,image_h, image_w):
        human_idx = 0
        min_instances = 3
        max_instances = 15
        region_props = []
        # for res in results:
        # pdb.set_trace()
        bx = results['ex_bbox']
        sc = results['ex_scores']
        lb = results['ex_labels']
        hs = results['ex_hidden_states']
        is_human = lb == human_idx
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human.sum(); n_object = len(lb) - n_human
        # Keep the number of human and object instances in a specified interval
        device = torch.device('cpu')
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            # keep_h = torch.nonzero(is_human[keep]).squeeze(1)
            # keep_h = keep[keep_h]
            keep_h = hum

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            # keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
            # keep_o = keep[keep_o]
            keep_o = obj

        keep = torch.cat([keep_h, keep_o])

        boxes=bx[keep]
        scores=sc[keep]
        labels=lb[keep]
        hidden_states=hs[keep]
        is_human = labels == human_idx
            
        n_h = torch.sum(is_human); n = len(boxes)
        # Permute human instances to the top
        if not torch.all(labels[:n_h]==human_idx):
            h_idx = torch.nonzero(is_human).squeeze(1)
            o_idx = torch.nonzero(is_human == 0).squeeze(1)
            perm = torch.cat([h_idx, o_idx])
            boxes = boxes[perm]; scores = scores[perm]
            labels = labels[perm]; unary_tokens = unary_tokens[perm]
        # Skip image when there are no valid human-object pairs
        if n_h == 0 or n <= 1:
            print(n_h, n)
            # boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
            # continue

        # Get the pairwise indices
        x, y = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )
        # pdb.set_trace()
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
        sub_boxes = boxes[x_keep]
        obj_boxes = boxes[y_keep]
        lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
        rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
        union_boxes = torch.cat([lt,rb],dim=-1)
        sub_boxes[:,0].clamp_(0, image_w)
        sub_boxes[:,1].clamp_(0, image_h)
        sub_boxes[:,2].clamp_(0, image_w)
        sub_boxes[:,3].clamp_(0, image_h)

        obj_boxes[:,0].clamp_(0, image_w)
        obj_boxes[:,1].clamp_(0, image_h)
        obj_boxes[:,2].clamp_(0, image_w)
        obj_boxes[:,3].clamp_(0, image_h)

        union_boxes[:,0].clamp_(0, image_w)
        union_boxes[:,1].clamp_(0, image_h)
        union_boxes[:,2].clamp_(0, image_w)
        union_boxes[:,3].clamp_(0, image_h)
    
        # region_props.append(dict(
        #     boxes=bx[keep],
        #     scores=sc[keep],
        #     labels=lb[keep],
        #     hidden_states=hs[keep],
        #     mask = ms[keep]
        # ))

        # return sub_boxes.int(), obj_boxes.int(), union_boxes.int()
        return sub_boxes, obj_boxes, union_boxes

    def get_union_mask(self, bbox, image_size):
        n = len(bbox)
        masks = torch.zeros
        pass

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

from torch.cuda import amp
class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes
        # self.scaler = amp.GradScaler(enabled=True)

    def _on_each_iteration(self):
        # with amp.autocast(enabled=True):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()



    @torch.no_grad()
    def test_hico(self, dataloader, args=None):
        net = self._state.net
        net.eval()
        dataset = dataloader.dataset.dataset
        interaction_to_verb = torch.as_tensor(dataset.interaction_to_verb)

        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        tgt_num_classes = 600
        
        num_gt = dataset.anno_interaction if args.dataset == "hicodet" else None
        meter = DetectionAPMeter(
            tgt_num_classes, nproc=1,
            num_gt=num_gt,
            algorithm='11P'
        )
        dict_hois = {}
        count = 0

        dict_hoi_score = {}
        tsne_feat_list = []
        tsne_label_list = []
        text_feat_list = []
        text_label_list = []
        name_list = {}
        pred_list = {}
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])

            outputs = net(inputs,batch[1])
            # Skip images without detections
            if outputs is None or len(outputs) == 0:
                continue

            else:
                for output, target in zip(outputs, batch[-1]):
                    count += 1
                    output = pocket.ops.relocate_to_cpu(output, ignore=True)
                    # Format detections
                    boxes = output['boxes']
                    boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
                    objects = output['objects']
                    scores = output['scores']
                    verbs = output['labels']

                    if net.module.num_classes==117 or net.module.num_classes==407:
                        interactions = conversion[objects, verbs]
                    else:
                        interactions = verbs

                    # Recover target box scale
                    gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
                    gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
                    # Associate detected pairs with ground truth pairs
                    labels = torch.zeros_like(scores)
                    unique_hoi = interactions.unique()
                    for hoi_idx in unique_hoi:
                        gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                        det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                        if len(gt_idx):
                            labels[det_idx] = associate(
                                (gt_bx_h[gt_idx].view(-1, 4),
                                gt_bx_o[gt_idx].view(-1, 4)),
                                (boxes_h[det_idx].view(-1, 4),
                                boxes_o[det_idx].view(-1, 4)),
                                scores[det_idx].view(-1)
                            )
                            # all_det_idxs.append(det_idx)
                    if args.vis_img == True:
                        save_path = os.path.join(args.vis_img_path)
                        if os.path.exists(save_path):
                            pass
                        else:
                            os.makedirs(save_path)
                        
                        save_path = os.path.join(save_path, batch[1][0]['filename'])

                        img_result = cv2.imread(os.path.join("../data/hico_20160224_det/images/test2015", batch[1][0]['filename'])).astype(np.float32)
                        
                        h, w, c = img_result.shape
                        h_ratio = h/target['size'][0]
                        w_ratio = w/target['size'][1]
                        
                        
                        # thres = 0.5
                        int_score, int_ind = scores.topk(7)
                        idx = 0
                        f_flag = 0
                        for temp_score, temp_ind in zip(int_score, int_ind):
                            idx += 1
                            # print("hois", hoi)
                            if idx > 9:
                                continue

                            # if int(interactions[temp_ind]) not in hico_unseen_index['unseen_verb'] or \
                            if int(interactions[temp_ind]) not in target['hoi']:
                                continue
                            
                            f_flag = 1

                            color = self.random_color()

                            # human
                            # h_id = hoi['subject_id']
                            
                            x1, y1, x2, y2 = boxes_h[temp_ind]
                            x1 *= w_ratio
                            x2 *= w_ratio
                            y1 *= h_ratio
                            y2 *= h_ratio
                            # h_cls = pred_bboxes[h_id]['category_id']
                            h_name = "person"
                            cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(img_result, '%s' % (h_name), (int(x1), int(y2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
                            
                            # object& action
                            # o_id = HOI_IDX_TO_OBJ_IDX[interactions[temp_ind]]
                            x1, y1, x2, y2 = boxes_o[temp_ind]
                            x1 *= w_ratio
                            x2 *= w_ratio
                            y1 *= h_ratio
                            y2 *= h_ratio
                            # ao_cls = hoi['category_id']
                            ao_name = HICO_INTERACTIONS[int(interactions[temp_ind])]['object']
                            cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            # cv2.putText(img_result, '%s:%.4f' % (o_name, o_cls), (x1, y2), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                            cv2.putText(img_result, '%s' % (ao_name), (int(x1), int(y2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
                            # action
                            # i_cls = hoi['category_id']
                            i_name = HICO_INTERACTIONS[int(interactions[temp_ind])]['action']
                            # cv2.putText(img_result, '%s:%.4f' % (i_name, i_cls),
                            #             (10, 50 * idx_box + 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

                            cv2.putText(img_result, '%s' % (i_name +": " +str(round(temp_score.item(), 3))),
                                    (10, int(50 * idx + 50)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)                    
                        if f_flag == 1:
                            cv2.imwrite( save_path, img_result)

                    meter.append(scores, interactions, labels)  
        return meter.eval()

    def random_color(self):
        rdn = random.randint(1, 1000)
        b = int(rdn * 997) % 255
        g = int(rdn * 4447) % 255
        r = int(rdn * 6563) % 255
        return b, g, r
    
    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache', args = None):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        hoi_dict = {}
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs, batch[1])

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct



            if args.vis_img == True:
                save_path = os.path.join(args.vis_img_path)
                if os.path.exists(save_path):
                    pass
                else:
                    os.makedirs(save_path)
                
                save_path = os.path.join(save_path, batch[1][0]['filename'])

                img_result = cv2.imread(os.path.join("../data/mscoco2014/val2014", batch[1][0]['filename'])).astype(np.float32)
                

            idx = -1
            cnt = -1
            for bh, bo, scr, acti in zip(boxes_h, boxes_o, scores, actions):
                idx += 1
                a_name = dataset.actions[acti].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = scr.item()
                result['_'.join(a_name)] = bo.tolist() + [scr.item()]
                all_results.append(result)
                # pdb.set_trace()

                if args.vis_img == True and scr >= scores.topk(4).values[-1] :  
                    cnt += 1
                    color = self.random_color()

                    x1, y1, x2, y2 = bh
                    h_name = "person"
                    cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(img_result, '%s' % (h_name), (int(x1), int(y2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
                    
                    # object& action
                    x1, y1, x2, y2 = bo
                    o_name = obj_to_name[output['objects'][idx]]
                    cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(img_result, '%s' % (o_name), (int(x1), int(y2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
                    # action

                    cv2.putText(img_result, '%s' % (a_name[0] +": " +str(round(scr.item(), 3))),
                            (10, int(50 * cnt + 50)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)                    

                    cv2.imwrite( save_path, img_result)
    

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir) 
        print('saving cache.pkl to', cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
