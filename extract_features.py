import os
import random
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.imagenet import ImageNet
from datasets.utils import build_data_loader
import clip
from utils import *
import json

def extract_text_feature(cfg, classnames, clip_model, template):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            
            texts = [t.format(classname) for t in template]
        
            texts_token = clip.tokenize(texts, truncate=True).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    torch.save(clip_weights, cfg['cache_dir'] + "/text_weights.pt")
    return

def extract_multi_scale_feature(cfg, split, clip_model, loader, scale):
    features, labels = [], []
    with torch.no_grad():
        for crop_idx in range(cfg['crop_epoch']):
            features_this = []
            for i, (images, target) in enumerate(tqdm(loader)):
                images = images.cuda()
                image_features = clip_model.encode_image(images)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                features_this.append(image_features)
                if crop_idx == 0:
                    target = target.cuda()
                    labels.append(target)
            features.append(torch.cat(features_this, dim=0))
    features, labels = torch.stack(features, dim=0), torch.cat(labels)
    torch.save(features, cfg['cache_dir'] + "/" + split + "_f"+ "_" + scale + "_.pt")
    label_path = cfg['cache_dir'] + "/" + split + "_l.pt"
    if not os.path.exists(label_path):
        torch.save(labels, label_path)
    return


def extract_ten_crop_feature(cfg, split, clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            features_this = []
            for crops in range(images.shape[1]):
                this_images = images[:,crops,:,:]
                image_features = clip_model.encode_image(this_images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features_this.append(image_features)
                if crops == 0:
                    target = target.cuda()
                    labels.append(target)
            features.append(torch.stack(features_this, dim=0).mean(dim=0))
    features = torch.cat(features, dim=0)
    features /= features.norm(dim=-1, keepdim=True)
    labels = torch.cat(labels)
    torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
    label_path = cfg['cache_dir'] + "/" + split + "_l.pt"
    if not os.path.exists(label_path):
        torch.save(labels, label_path)
    return


if __name__ == '__main__':
    
    clip_model, preprocess = clip.load('RN50')
    clip_model.eval()

    all_dataset = ["caltech101", 'dtd', 'eurosat', 'fgvc', 'food101', 'imagenet', 
                   'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101']
    k_shot = [1]
    data_path = 'DATA'
    this_scale = 0.1
    for set in all_dataset:

        cfg = yaml.load(open('configs/{}.yaml'.format(set), 'r'), Loader=yaml.Loader)
        cache_dir = os.path.join('./caches', cfg['dataset'])
        os.makedirs(cache_dir, exist_ok=True)
        cfg['cache_dir'] = cache_dir
        cfg['crop_epoch'] = 100
        for k in k_shot:
                
            random.seed(1)
            torch.manual_seed(1)
            
            cfg['shots'] = k
            # 10-crop
            test_transform = transforms.Compose([
                transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.TenCrop(size=224),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(crop) for crop in crops])),
            ])
            # multi-scale
            # test_transform = transforms.Compose([
            #     transforms.RandomResizedCrop(size=224, scale=(this_scale, this_scale), interpolation=transforms.InterpolationMode.BICUBIC),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
            
            if set == 'imagenet':
                dataset = ImageNet(cfg['root_path'], cfg['shots'], test_transform)
                val_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
                train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)           
            else:   
                dataset = build_dataset(set, data_path, k)

                val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=test_transform, shuffle=False)
                test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=test_transform, shuffle=False)

        # Extract multi-scale features
        print("\nLoading visual features and labels from val and test set.")
        extract_ten_crop_feature(cfg, "val", clip_model, val_loader)
        # extract_multi_scale_feature(cfg, "val", clip_model, val_loader, this_scale)
        if not set == 'imagenet':
            extract_ten_crop_feature(cfg, "test", clip_model, test_loader)
            # extract_multi_scale_feature(cfg, "test", clip_model, val_loader, this_scale)
    extract_text_feature(cfg, dataset.classnames, clip_model, dataset.template)
            

    