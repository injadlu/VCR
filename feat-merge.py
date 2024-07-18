import torch
import os
import pdb

def mean_merge(feats):
    mean_feat = torch.mean(feats, dim=0)
    mean_feat /= mean_feat.norm(dim=-1, keepdim=True)
    return mean_feat

def load_feature(dir, scale, split):

    feat_dir = dir + "/" + split + "_f_" + scale + ".pt"
    features = torch.load(feat_dir)
    return features

def load_global_feature(dir, split):

    feat_dir = dir + "/" + split + "_f.pt"
    features = torch.load(feat_dir)
    return features

def save_feature(feat, save_dir, split):

    save_path = save_dir + "/" + split + "_f.pt"
    torch.save(feat, save_path)

all_dataset = ["caltech101", 'dtd', 'eurosat', 'fgvc', 'food101', 'imagenet', 
                'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101']
for set in all_dataset:
    # path
    textual_dir = os.path.join('./caches', set)
    feat_dir = os.path.join('./selected', set)
    save_dir = os.path.join('./final', set)
    global_dir = os.path.join('./caches', set)
    # textual features
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Getting textual features as CLIP's classifier.")
    for split in ['test']:
        if set == 'imagenet' and split == 'test':
            continue
        features_all = torch.zeros(load_feature(feat_dir, '1', split).shape).half().cuda()
        for scale in range(1, 11):
            # test features
            print("Getting test features.")
            if scale != 10:
                features = load_feature(feat_dir, str(scale), split)
            else:
                features = load_global_feature(global_dir, split)
            
            ratio = scale / 55.0
            features_all += features * ratio
        features_all /= features_all.norm(dim=-1, keepdim=True)
        save_feature(features_all, save_dir, split)
