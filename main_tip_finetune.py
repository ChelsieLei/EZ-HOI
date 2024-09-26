"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.ops.boxes import box_iou
sys.path.append('detr')
from upt_tip_cache_model_free_finetune_distillself import build_detector
from utils_tip_cache_and_union_finetune import custom_collate, CustomisedDLE, DataFactory
import pdb
from hico_text_label import hico_unseen_index
import vcoco_text_label, hico_text_label

warnings.filterwarnings("ignore")

def tranverse_and_get_hoi_cooccurence(dataset):
    category = dataset.num_interation_cls
    hoi_cooccurence = torch.zeros(category, category)
    for anno in dataset._anno:
        num_gt = len(anno['hoi'])
        for i in range(num_gt):
            for j in range(i+1, num_gt):
                ## need to judge if anno['hoi'][i] and anno['hoi'][j] are the same pair
                h_iou = box_iou(torch.as_tensor(anno['boxes_h'][i:i+1]), torch.as_tensor(anno['boxes_h'][j:j+1]))
                o_iou = box_iou(torch.as_tensor(anno['boxes_o'][i:i+1]), torch.as_tensor(anno['boxes_o'][j:j+1]))
                if min(h_iou.item(), o_iou.item()) > 0.5:
                    if anno['hoi'][i] == anno['hoi'][j]:
                        continue
                    hoi_cooccurence[anno['hoi'][i],anno['hoi'][j]] += 1
                    hoi_cooccurence[anno['hoi'][j],anno['hoi'][i]] += 1
    hoi_cooccurence = hoi_cooccurence.t() / (hoi_cooccurence.sum(dim=-1) + 1e-9)
    hoi_cooccurence = hoi_cooccurence.t()   
    return hoi_cooccurence

def hico_class_corr():
    """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
    class_corr = []
    for i, (k, v) in enumerate(hico_text_label.hico_text_label.items()):
        class_corr.append([i, k[1], k[0]])
    return class_corr

def vcoco_class_corr():
    """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
    class_corr = []
    for i, (k, v) in enumerate(vcoco_text_label.vcoco_hoi_text_label.items()):
        class_corr.append([i, k[1], k[0]])
    return class_corr

def vcoco_object_n_verb_to_interaction(num_object_cls, num_action_cls, class_corr):
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([num_object_cls, num_action_cls], None)
        for i, j, k in class_corr:
            lut[j, k] = i
        return lut.tolist()

def swig_object_n_verb_to_interaction(num_object_cls, num_action_cls, class_corr):
        """
        The interaction classes corresponding to an object-verb pair

        class_corr: List[(hoi_id, object_id, action_id)]

        Returns:
            list[list[407]]
        """
        lut = np.full([num_object_cls, num_action_cls], None)
        for hoi_id, object_id, action_id in class_corr:
            lut[object_id, action_id] = hoi_id
        return lut.tolist()

def swig_object_to_interaction(num_object_cls, _class_corr):
        """
        class_corr: List[(x["id"], x["object_id"], x["action_id"])]
        
        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(num_object_cls)]
        for hoi_id, object_id, action_id in _class_corr:
            obj_to_int[object_id].append(hoi_id)
        return obj_to_int

def swig_object_to_verb(num_object_cls, _class_corr):
        """
        class_corr: List[(x["id"], x["object_id"], x["action_id"])]
        
        Returns:
            list[list]
        """
        obj_to_verb = [[] for _ in range(num_object_cls)]
        for hoi_id, object_id, action_id in _class_corr:
            obj_to_verb[object_id].append(action_id)
        return obj_to_verb

def swig_verb2interaction(num_action_cls, num_interaction_cls, class_corr):
    '''
    Returns: List[hoi_id] = action_id
    '''
    v2i = np.full([num_interaction_cls], None)
    for hoi_id, object_id, action_id in class_corr:
        v2i[hoi_id] = action_id
    return v2i.tolist()

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'
    elif args.clip_model_name == 'ViT-B-32':
        args.clip_model_name = 'ViT-B/32'
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'
    if args.dataset == "swig":
        args.dataset_file = 'swig'
        trainset = build_dataset(image_set='train', args=args)
        testset = build_dataset(image_set='val', args=args)
        class_coor = [(x["id"], x["object_id"], x["action_id"]) for x in SWIG_INTERACTIONS]
        if args.eval: # TODO ????
            class_coor = [(x["id"], x["object_id"], x["action_id"]) for x in SWIG_INTERACTIONS if x["evaluation"] == 1]
        object_n_verb_to_interaction = swig_object_n_verb_to_interaction(num_object_cls=1000, num_action_cls=407, class_corr=class_coor)
        trainset.object_n_verb_to_interaction = object_n_verb_to_interaction
        testset.object_n_verb_to_interaction = object_n_verb_to_interaction
        object_to_interaction = swig_object_to_interaction(num_object_cls=1000, _class_corr=class_coor)
        trainset.object_to_interaction = object_to_interaction
        testset.object_to_interaction = object_to_interaction
        object_to_verb = swig_object_to_verb(num_object_cls=1000, _class_corr=class_coor)
        trainset.object_to_verb = object_to_verb
        testset.object_to_verb = object_to_verb

        verb2interaction = swig_verb2interaction(num_action_cls=407, num_interaction_cls=14130, class_corr=class_coor)
    else:
        trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root, clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type, num_classes=args.num_classes)
        testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root, clip_model_name=args.clip_model_name)
        verb2interaction = None
        # trainset[0][1]: dict_keys(['boxes_h', 'boxes_o', 'hoi', 'object', 'verb', 'orig_size', 'labels', 'size', 'filename'])
        # trainset[0][0]: (torch.Size([3, 814, 640]), torch.Size([3, 224, 224]))
    if args.dataset == 'vcoco':
        class_corr = vcoco_class_corr()
        trainset.dataset.class_corr = class_corr
        testset.dataset.class_corr = class_corr
        object_n_verb_to_interaction = vcoco_object_n_verb_to_interaction(num_object_cls=len(trainset.dataset.objects), num_action_cls=len(trainset.dataset.actions), class_corr=class_corr)
        trainset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction
        testset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction

    # args.hoi_cooccurence = tranverse_and_get_hoi_cooccurence(trainset.dataset)
    if args.training_set_ratio < 0.9:
        print(f'[INFO]: using {args.training_set_ratio} trainset to train!')
        sub_trainset, valset = trainset.dataset.split(args.training_set_ratio)
        trainset.dataset = sub_trainset
        trainset.keep = [i for i in range(len(sub_trainset))]
        
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )

    ## select pic for teaser
    # for imgs, tgts in test_loader:
    #     img, img_crop = imgs[0] ## batchsize==1
    #     tgt = tgts[0]
    #     length = len(tgt['hoi'])
    #     flag = True
    #     for i in range(length):
    #         for j in range(i+1, length):
    #             if box_iou(box_ops.box_cxcywh_to_xyxy(tgt['boxes_h'][i:i+1]), box_ops.box_cxcywh_to_xyxy(tgt['boxes_h'][j:j+1]))[0] < 0.8:
    #                 flag = False
    #     if flag:
    #         pdb.set_trace()
    # for imgs, tgts in test_loader:
    #     img, img_crop = imgs[0] ## batchsize==1
    #     tgt = tgts[0]
    #     if '51610' in tgt['filename']:
    #         pdb.set_trace() 

    # test_loader_of_trainingset = DataLoader(
    #     dataset=trainset,
    #     collate_fn=custom_collate, batch_size=1,
    #     num_workers=args.num_workers, pin_memory=False, drop_last=False,
    #     sampler=torch.utils.data.SequentialSampler(trainset)
    # )

    args.human_idx = 0
    if args.dataset == 'swig':
        object_n_verb_to_interaction = train_loader.dataset.object_n_verb_to_interaction
    else:
        object_n_verb_to_interaction = train_loader.dataset.dataset.object_n_verb_to_interaction

    if args.dataset == 'hicodet':
        if args.num_classes == 117:
            object_to_target = train_loader.dataset.dataset.object_to_verb
        elif args.num_classes == 600:
            object_to_target = train_loader.dataset.dataset.object_to_interaction
        
        if args.zs:
            object_to_target = train_loader.dataset.zs_object_to_target
    elif args.dataset == 'vcoco':
        if args.num_classes == 24:
            object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        elif args.num_classes == 236:
            raise NotImplementedError
    elif args.dataset == 'swig':
        if args.num_classes == 407:
            object_to_target = train_loader.dataset.object_to_verb
        elif args.num_classes == 14130 or args.num_classes == 5539:
            object_to_target = train_loader.dataset.object_to_interaction
        
    print('[INFO]: num_classes', args.num_classes)
    if args.dataset == 'vcoco' or args.dataset == 'swig':
        num_anno = None
    else:
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        if args.num_classes == 117:
            num_anno = torch.as_tensor(trainset.dataset.anno_action)
    # pdb.set_trace()

    upt = build_detector(args, object_to_target, object_n_verb_to_interaction=object_n_verb_to_interaction, clip_model_path=args.clip_dir_vit, num_anno=num_anno, verb2interaction=verb2interaction)

    if args.dataset == 'hicodet' and args.eval:  ## after building model, manually change obj_to_target
        if args.num_classes == 117:
            upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_verb
        else:
            upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_interaction
    if args.pseudo_label:  ## if we generate pseudo label for unseen verbs,
        pdb.set_trace()
        upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_verb

    if os.path.exists(args.resume_part):
        print(f"===>>> Rank {rank}: partially continue from saved checkpoint {args.resume}")
        checkpoint_part = torch.load(args.resume_part, map_location='cpu')
        upt.load_state_dict(checkpoint_part['model_state_dict'])  ## , strict = False

    if os.path.exists(args.resume):
        print(f"===>>> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.dataset == 'swig' and args.eval:
            ckpt = checkpoint['model_state_dict']
            test_hoi_ids = torch.as_tensor([interaction["id"] for interaction in SWIG_INTERACTIONS if interaction["evaluation"] == 1])
            ckpt['clip_head.prompt_learner.token_prefix'] = ckpt['clip_head.prompt_learner.token_prefix'][test_hoi_ids, :, :]
            ckpt['clip_head.prompt_learner.token_suffix'] = ckpt['clip_head.prompt_learner.token_suffix'][test_hoi_ids, :, :]
            upt.load_state_dict(ckpt)
        else:
            if 'e632da11' in args.pretrained and args.dataset == 'hicodet':
                model_dict = checkpoint['model_state_dict']
                model_dict = {k: v for k, v in model_dict.items() if 'detector.class_embed' not in k}
                upt.load_state_dict(model_dict, strict = False)
                # pdb.set_trace()
            else:
                upt.load_state_dict(checkpoint['model_state_dict'], strict=False)  # strict=False
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")
    
    if args.zs and args.fill_zs_verb_type == 1:
        upt.refresh_unseen_verb_cache_mem() ## whether refresh unseen weights after loading weights (during test)
    
    engine = CustomisedDLE(
        upt, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
    )
    if args.vis_tor != 1 and (args.eval or args.cache):
        upt.logit_scale_HO = torch.nn.Parameter(upt.logit_scale_HO * args.vis_tor)
        upt.logit_scale_U = torch.nn.Parameter(upt.logit_scale_U * args.vis_tor)

    if args.tpt:
        engine = CustomisedDLE(
            upt, test_loader_for_tpt,
            max_norm=args.clip_max_norm,
            num_classes=args.num_classes,
            print_interval=args.print_interval,
            find_unused_parameters=True,
            cache_dir=args.output_dir,
        )

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            # print("[!NOTE!]: using test_loader_of_trainingset")
            engine.cache_vcoco(test_loader, args.output_dir, args = args)
        return
    
    if args.eval:
        device = torch.device(args.device)
        upt.eval()
        if args.dataset == 'vcoco':
            raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
        ap = engine.test_hico(test_loader, args)
        # Fetch indices for rare and non-rare classes
        print("ap", ap)
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        # print("rare", ap[rare], len(ap[rare]))
        # print("nonrare", ap[non_rare], len(ap[non_rare]))
        print(
            f"The mAP is {ap.mean()*100:.2f},"
            f" rare: {ap[rare].mean()*100:.2f},"
            f" none-rare: {ap[non_rare].mean()*100:.2f},"
        )
        if args.zs:
            zs_hoi_idx = hico_unseen_index[args.zs_type]
            print(f'>>> zero-shot setting({args.zs_type}!!)')
            ap_unseen = []
            ap_seen = []
            for i, value in enumerate(ap):
                if i in zs_hoi_idx: 
                    ap_unseen.append(value)
                else: 
                    ap_seen.append(value)
            ap_unseen = torch.as_tensor(ap_unseen).mean()
            ap_seen = torch.as_tensor(ap_seen).mean()
            print(
                f"full mAP: {ap.mean()*100:.2f}",
                f"unseen: {ap_unseen*100:.2f}",
                f"seen: {ap_seen*100:.2f}",
            )
            
        return

    for p in upt.detector.parameters():
        p.requires_grad = False
    for n, p in upt.clip_head.named_parameters():
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj'): 
        # if n.startswith('positional_embedding') or n.startswith('ln_post') or n.startswith('proj'): 
            p.requires_grad = True
        elif 'adaptermlp' in n or "prompt_learner" in n:
            p.requires_grad = True
            # print(n)          
        else: 
            p.requires_grad = False


    if args.frozen_classifier != None:
        frozen_name_lst = []
        if 'HO' in args.frozen_classifier:
            frozen_name_lst.append('adapter_HO')
        if 'U' in args.frozen_classifier:
            frozen_name_lst.append('adapter_U')
        if 'T' in args.frozen_classifier:
            frozen_name_lst.append('adapter_union')
        
        for n, p in upt.named_parameters():
            if 'clip_head' in n or 'detector' in n:
                continue
            if n.split('.')[0] in frozen_name_lst:
                p.requires_grad = False
    
    if args.label_learning:
        for n, p in upt.named_parameters():
            if 'clip_head' in n or 'detector' in n:
                continue
            if 'label_' in n:
                p.requires_grad = True
            
    others = [n for n, p in upt.named_parameters()
                    if p.requires_grad and 'clip_head' not in n]

    param_dicts = [
        {
            "params": [p for n, p in upt.clip_head.named_parameters()
                    if p.requires_grad]
        },
        { ## others
            "params": [p for n, p in upt.named_parameters()
                    if p.requires_grad and 'clip_head' not in n],
            "lr": args.lr_head,
        },
    ]
    

    # for n, p in upt.named_parameters():
    #     if p.requires_grad is False and 'clip_head' not in n:
    #         print(n)
    #     if 'fixed_clip_enctxt' in n:
    #         print("wait!", n)
    # pdb.set_trace()
    # print("clip number", sum(p.numel() for n,p in upt.named_parameters() if p.requires_grad is False and 'clip' in n))


    # # Double check
    # enabled = set()
    # for name, param in upt.named_parameters():
    #     if param.requires_grad:
    #         enabled.add(name)
    # print(f"Parameters to be updated: {enabled}")


    n_parameters = sum(p.numel() for p in upt.parameters() if p.requires_grad)
    print('number of leanable params:', n_parameters)
    
    n_parameters = sum(p.numel() for p in upt.parameters())
    print('number of all params:', n_parameters)
    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_vit,
        weight_decay=args.weight_decay
    )
    # optim = torch.optim.SGD(
    #     param_dicts, lr=args.lr_vit,
    #     # weight_decay=args.weight_decay
    # )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch=checkpoint['epoch']
        iteration = checkpoint['iteration']
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    # Override optimiser and learning rate scheduler
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch,iteration=iteration, scaler=scaler)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    # with torch.autograd.set_detect_anomaly(True):

    import json
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

    engine(args.epochs)


@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.human_idx = 0; args.num_classes = 117
    object_to_target = dataset.dataset.object_to_verb
    upt = build_detector(args, object_to_target)
    if args.eval:
        upt.eval()

    image, target = dataset[0]
    outputs = upt([image], [target])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-3, type=float)
    parser.add_argument('--lr-vit', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1233', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--visual_mode', default='vit', type=str)
    # add CLIP model resenet 
    # parser.add_argument('--clip_dir', default='./checkpoints/pretrained_clip/RN50.pt', type=str)
    # parser.add_argument('--clip_visual_layers', default=[3, 4, 6, 3], type=list)
    # parser.add_argument('--clip_visual_output_dim', default=1024, type=int)
    # parser.add_argument('--clip_visual_input_resolution', default=1344, type=int)
    # parser.add_argument('--clip_visual_width', default=64, type=int)
    # parser.add_argument('--clip_visual_patch_size', default=64, type=int)
    # parser.add_argument('--clip_text_output_dim', default=1024, type=int)
    # parser.add_argument('--clip_text_transformer_width', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers', default=12, type=int)
    # parser.add_argument('--clip_text_context_length', default=13, type=int)

    #### add CLIP vision transformer
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)

    ### ViT-L/14@336px START: emb_dim: 768 
    # >>> vision_width: 1024,  vision_patch_size(conv's kernel-size&&stride-size): 14,
    # >>> vision_layers(#layers in vision-transformer): 24 ,  image_resolution:336;
    # >>> transformer_width:768, transformer_layers: 12, transformer_heads:12
    parser.add_argument('--clip_visual_layers_vit', default=24, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=768, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=336, type=int)
    parser.add_argument('--clip_visual_width_vit', default=1024, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=14, type=int)

    # parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=768, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=12, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-L/14@336px----END----
    
    ### ViT-B-16 START
    # parser.add_argument('--clip_visual_layers_vit', default=12, type=list)
    # parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_visual_input_resolution_vit', default=224, type=int)
    # parser.add_argument('--clip_visual_width_vit', default=768, type=int)
    # parser.add_argument('--clip_visual_patch_size_vit', default=16, type=int)

    # # parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_width_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads_vit', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-B-16-----END-----
    parser.add_argument('--clip_text_context_length_vit', default=77, type=int) # 13 -77
    parser.add_argument('--use_insadapter', action='store_true')
    parser.add_argument('--use_distill', action='store_true')
    parser.add_argument('--use_consistloss', action='store_true')
    
    parser.add_argument('--use_mean', action='store_true') # 13 -77
    parser.add_argument('--logits_type', default='HO+U+T', type=str) # 13 -77 # text_add_visual, visual
    parser.add_argument('--num_shot', default='2', type=int) # 13 -77 # text_add_visual, visual
    parser.add_argument('--file1', default='./hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p',type=str)
    parser.add_argument('--prior_type', type=str, default='cbe', choices=['cbe', 'cb', 'ce', 'be', 'c', 'b', 'e'])
    parser.add_argument('--obj_affordance', action='store_true') ## use affordance embedding of objects
    parser.add_argument('--training_set_ratio', type=float, default=1.0)
    parser.add_argument('--frozen_classifier', type=str, default=None)
    parser.add_argument('--zs', action='store_true') ## zero-shot
    parser.add_argument('--hyper_lambda', type=float, default=2.8)
    parser.add_argument('--use_weight_pred', action='store_true')
    parser.add_argument('--zs_type', type=str, default='rare_first', choices=['rare_first', 'non_rare_first', 'unseen_verb', 'unseen_object', 'uc0', 'uc1', 'uc2', 'uc3', 'uc4', 'default'])
    parser.add_argument('--fill_zs_verb_type', type=int, default=0,) # (for init) 0: random; 1: weighted_sum, 
    parser.add_argument('--pseudo_label', action='store_true') 
    parser.add_argument('--tpt', action='store_true') 
    parser.add_argument('--vis_tor', type=float, default=1.0)
    parser.add_argument('--adapter_num_layers', type=int, default=1)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--vis_embed_dim', type=int, default=64)
    
    
    ## prompt learning
    parser.add_argument('--N_CTX', type=int, default=2)  # number of context vectors
    parser.add_argument('--tune_LY', type=int, default=9)  # number of context vectors
    parser.add_argument('--CSC', type=bool, default=False)  # class-specific context
    parser.add_argument('--CTX_INIT', type=str, default='A photo of')  # initialization words
    parser.add_argument('--CLASS_TOKEN_POSITION', type=str, default='end')  # # 'middle' or 'end' or 'front'

    parser.add_argument('--use_templates', action='store_true') 
    parser.add_argument('--LA', action='store_true')  ## Language Aware
    parser.add_argument('--LA_weight', default=0.6, type=float)  ## Language Aware(loss weight)

    parser.add_argument('--feat_mask_type', type=int, default=0,) # 0: dropout(random mask); 1: None
    parser.add_argument('--num_classes', type=int, default=117,) 
    parser.add_argument('--prior_method', type=int, default=0) ## 0: instance-wise, 1: pair-wise, 2: learnable
    parser.add_argument('--vis_prompt_num', type=int, default=50) ##  (prior_method == learnable)
    parser.add_argument('--box_proj', type=int, default=0,) ## 0: None; 1: f_u = ROI-feat + MLP(uni-box)
    parser.add_argument('--adapter_pos', type=str, default='all', choices=['all', 'front', 'end', 'random', 'last'])
    parser.add_argument('--use_multi_hot', action='store_true')
    parser.add_argument('--label_learning', action='store_true')
    parser.add_argument('--label_choice', default='random', choices=['random', 'single_first', 'multi_first', 'single+multi', 'rare_first', 'non_rare_first', 'rare+non_rare'])
    parser.add_argument('--use_mlp_proj', action='store_true')
    
    parser.add_argument('--repeat_factor_sampling', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='apply repeat factor sampling to increase the rate at which tail categories are observed')
    
    ## **************** arguments for deformable detr **************** ##
    parser.add_argument('--d_detr', default=False, type=lambda x: (str(x).lower() == 'true'),)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    # * Transformer
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--vis_img', action="store_true")
    parser.add_argument('--vis_img_path', default = "pred_annotation", type=str)
    ## **************** arguments for deformable detr **************** ##
    
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--clip_type", type=str, default="clip", choices=('clip', 'clip_surgery'))
    parser.add_argument('--self_mem_path', type = str, default=None)
    parser.add_argument('--resume_part', default='', help='Resume from a model')
    parser.add_argument('--multi_cross', default=False, action='store_true')
    parser.add_argument('--fix_mem', default=False, action='store_true')
    parser.add_argument('--txt_mem_init', default=False, action='store_true')
    parser.add_argument('--mem_adpt_self', default=False, action='store_true')
    parser.add_argument('--mem_sa_only', default=False, action='store_true')
    parser.add_argument('--vlmtxt', default='llava', type=str, choices=('llava', 'fix'))
    parser.add_argument('--img_descrip_prompt', default=False, action='store_true')  ### image encoder VLM image description
    parser.add_argument('--txtcls_descrip', default=False, action='store_true')  ### text encoder adapter
    parser.add_argument('--txtcls_pt', default=False, action='store_true')   ### text encoder class specific prompts
    parser.add_argument('--init_txtcls_pt', default=False, action='store_true') 
    parser.add_argument('--img_align', default=False, action='store_true')   ### image encoder final self attention
    parser.add_argument('--unseen_pt_inj', default=False, action='store_true')   ## unseen-class prompts prior transfer
    parser.add_argument('--seperate_pre_pt', default=False, action='store_true')  ### separate class-shared prompts to be learnable in text encoder
    parser.add_argument('--img_clip_pt', default=False, action='store_true')  ### iamge learnabel prompt adapter with CLIP image features
    parser.add_argument('--clip_img_file', default='hicodet_pkl_files/clip336_img_hicodet_train', type=str)
    parser.add_argument('--deunify_pt', default=False, action='store_true')
    parser.add_argument('--de_txtcls', default=False, action='store_true')
    parser.add_argument('--txt_align', default=False, action='store_true')  
    parser.add_argument('--fix_txt_pt', default=False, action='store_true')  
    parser.add_argument('--clip_test', default=False, action='store_true')    ### degrade maple to clip (fix text and visual prompts)
    parser.add_argument('--act_descriptor', default=False, action='store_true')
    parser.add_argument('--use_gt_boxes', default=False, action='store_true')
    parser.add_argument("--pt_begin_layer", type=int, default=0)
    parser.add_argument('--without_unseen_name', default=False, action='store_true')
    parser.add_argument('--origin_ctx', default=False, action='store_true')
    parser.add_argument('--finetune_allcls_utpl', default=False, action='store_true')

    


    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    # mp.spawn(main, nprocs=args.world_size, args=(args,))
    if args.world_size==1:
        main(0,args)
    else:
        mp.spawn(main, nprocs=args.world_size, args=(args,))