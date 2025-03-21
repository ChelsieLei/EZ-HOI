# [NeurIPS 2024] EZ-HOI: VLM Adaptation via Guided  Prompt Learning for Zero-Shot HOI Detection

## Paper Links

[arXiv](https://arxiv.org/pdf/2410.23904) 
[Project Page](https://chelsielei.github.io/EZHOI_ProjPage/)


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

**Reminder**: Please check whether you are using the local CLIP dir provided by our EZ-HOI repo. 
If you have already installed `clip` in their Python env, to avoid the situation that you use the built-in clip package in your python env (i.e., via pip install clip) rather than the local CLIP dir provided by our EZ-HOI repo. Please use the local CLIP by setting the `PYTHONPATH` , 
```
export PYTHONPATH=$PYTHONPATH:"your_path/EZ-HOI/CLIP"
```
So that you can use the local clip **without uninstall the clip of your python env**.


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

4. modify the installed [pocket](https://github.com/fredzzhang/pocket) library as mentioned [here](https://github.com/ChelsieLei/EZ-HOI/issues/2)

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
title={EZ-HOI: VLM Adaptation via Guided Prompt Learning for Zero-Shot HOI Detection},
author={Lei, Qinqian and Wang, Bo and Robby T., Tan},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```

## Acknowledgement
We gratefully thank the authors from [UPT](https://github.com/fredzzhang/upt) and [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main) for open-sourcing their code.






