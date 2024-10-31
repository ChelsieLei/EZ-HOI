# [NeurIPS 2024] EZ-HOI: VLM Adaptation via Guided  Prompt Learning for Zero-Shot HOI Detection

## Paper Links

<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.5678-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/)  -->
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit%20Here-4D8C2A?style=for-the-badge)](https://chelsielei.github.io/EZHOI_ProjPage/)


## Dataset 
Follow the process of [UPT](https://github.com/fredzzhang/upt).

The downloaded files should be placed as follows. Otherwise, please replace the default path to your custom locations.
```
|- EZ-HOI
|   |- hicodet
|   |   |- hico_20160224_det
|   |       |- annotations
|   |       |- images
|   |- vcoco
|   |   |- mscoco2014
|   |       |- train2014
|   |       |-val2014
:   :      
```

## Dependencies
1. Follow the environment setup in [UPT](https://github.com/fredzzhang/upt).

2. Follow the environment setup in [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main).

3. run the python file to obtain the pre-extracted CLIP image features
```
python CLIP_hicodet_extract.py
```
Remember to make sure the correct path for annotation files and datasets.

```
|- EZ-HOI
|   |- hicodet_pkl_files
|   |   |- clip336_img_hicodet_test
|   |   |- clip336_img_hicodet_train
|   |   |- clipbase_img_hicodet_test
|   |   |- clipbase_img_hicodet_train
|   |- vcoco_pkl_files
|   |   |- clip336_img_vcoco_train
|   |   |- clip336_img_vcoco_val
:   :      
```


## HICO-DET
### Train on HICO-DET:
```
bash scripts/hico_train_vitB_zs.sh
```

### Test on HICO-DET:
```
bash scripts/hico_test_vitB_zs.sh
```


## Model Zoo

| Dataset | Setting| Backbone  | mAP | Unseen | Seen |
| ---- |  ----  | ----  | ----  | ----  | ----  |
| HICO-DET | [UV](https://drive.google.com/drive/folders/1Uy18k4zzacQXw4ABYi61dxlyQBfyeP4k?usp=drive_link) | ResNet-50+ViT-B  | 32.32|25.10|33.49|
| HICO-DET |UV| ResNet-50+ViT-L  | 36.84 | 28.82|38.15|
| HICO-DET | [RF](https://drive.google.com/drive/folders/1Uy18k4zzacQXw4ABYi61dxlyQBfyeP4k?usp=drive_link)| ResNet-50+ViT-B  | 33.13 |29.02|34.15|
| HICO-DET |RF| ResNet-50+ViT-L  | 36.73|34.24|37.35|
| HICO-DET | [NF](https://drive.google.com/drive/folders/1Uy18k4zzacQXw4ABYi61dxlyQBfyeP4k?usp=drive_link)| ResNet-50+ViT-B  | 31.17|33.66|30.55|
| HICO-DET |NF| ResNet-50+ViT-L  | 34.84|36.33|34.47|
| HICO-DET | [UO](https://drive.google.com/drive/folders/1Uy18k4zzacQXw4ABYi61dxlyQBfyeP4k?usp=drive_link)| ResNet-50+ViT-B  | 32.27|33.28|32.06|
| HICO-DET |UO| ResNet-50+ViT-L  | 36.38|38.17|36.02|

| Dataset | Setting| Backbone  | mAP | Rare | Non-rare |
| ---- |  ----  | ----  | ----  | ----  | ----  |
| HICO-DET |[default](https://drive.google.com/drive/folders/1Uy18k4zzacQXw4ABYi61dxlyQBfyeP4k?usp=drive_link)| ResNet-50+ViT-L  | 38.61|37.70|38.89|

## Citation
If you find our paper and/or code helpful, please consider citing :
```
@inproceedings{
lei2024efficient,
title={Efficient Zero-Shot HOI Detection: Enhancing VLM Adaptation with Innovative Prompt Learning},
author={Lei, Qinqian and Wang, Bo and Robby T., Tan},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```

## Acknowledgement
We gratefully thank the authors from [UPT](https://github.com/fredzzhang/upt) and [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main) for open-sourcing their code.






