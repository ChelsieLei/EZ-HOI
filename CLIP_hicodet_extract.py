
import pickle
import clip
import os
import json
from PIL import Image
import torch
import pdb

mode_list = ['train', 'test'] 
clip_mode_list = ['ViT-B/16' , 'ViT-L/14@336px']

for mode in mode_list:
    for clip_mode in clip_mode_list:
        ##### load all file names for the CLIP processing. NOT using the annotation
        if mode == 'train':
            hico_problems = json.load(open("hicodet/trainval_hico.json", 'r'))
        else:
            hico_problems = json.load(open("hicodet/test_hico.json", 'r'))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(clip_mode, device)
        img_path = 'hicodet/hico_20160224_det/images/' + mode+"2015"   ### the dataset path

        file_name_clippart = 'clip336' if clip_mode == 'ViT-L/14@336px' else 'clipbase'
        folder = 'hicodet_pkl_files/'+file_name_clippart+'_img_hicodet_'+mode
        if os.path.exists(folder) is False:
            os.makedirs(folder)

        # img_feats = {}
        for idx, info_hoii in enumerate(hico_problems):
            filename = info_hoii['file_name']
            # temp_det = det[filename][0]
            image = preprocess(Image.open(os.path.join(img_path, filename))).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features = image_features.squeeze(0)[1:]
            # img_feats[filename] = (image_features)
            print("successfully generate image feature for ", filename)
            file = open(os.path.join(folder, filename.split(".")[0]+"_clip.pkl"), 'wb')
            pickle.dump(image_features, file)
            file.close()
