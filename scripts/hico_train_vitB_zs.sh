CUDA_VISIBLE_DEVICES=0,1,2,3 python main_tip_finetune.py --world-size 4 \
 --pretrained "checkpoints/detr-r50-hicodet.pth" \
 --output-dir checkpoints/hico_HO_pt_default_vitbase/ \
 --epochs 12  --use_insadapter  --num_classes 117 --use_multi_hot \
 --file1 /mnt/disk2/qinqian/ADA-CM/hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
 --clip_dir_vit /mnt/disk2/qinqian/ADA-CM/checkpoints/pretrained_CLIP/ViT-B-16.pt \
 --batch-size 8  --logits_type "HO"  --port 1236 \
 --selfdistill 1  \
 --txtcls_pt   --img_align  --unseen_pt_inj  --img_clip_pt  \
 --zs --zs_type "unseen_verb" \
 --clip_img_file   hicodet_pkl_files/clipbase_img_hicodet_train \

 