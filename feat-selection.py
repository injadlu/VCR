import torch
import yaml
import os
import pdb

def load_text_feature(textual_dir):
    save_path = textual_dir + "/text_weights.pt"
    clip_weights = torch.load(save_path)
    return clip_weights


def select_image_views(view_feat, clip_weights):
    norm_view_feat = view_feat / view_feat.norm(dim=-1, keepdim=True)
    local_logits = norm_view_feat @ clip_weights
    logits_values, _ = torch.topk(local_logits, k=2, dim=-1)
    criterion = logits_values[:,:,0] - logits_values[:,:,1]
    local_idx = torch.argsort(criterion, dim=0, descending=True)[:1]
    selected = torch.take_along_dim(view_feat, local_idx[:,:,None], dim=0).squeeze(0)
    return selected

def load_feature(dir, scale, split):
    feat_dir = dir + "/" + split + "_f_" + scale + "_.pt"
    features = torch.load(feat_dir)

    return features

def save_feature(selected_features, save_dir, scale, split):
    save_path = save_dir + "/" + split +"_f_" + scale + ".pt"
    torch.save(selected_features, save_path)
    return


all_dataset = ["caltech101", 'dtd', 'eurosat', 'fgvc', 'food101', 'imagenet', 
                'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101']
for set in all_dataset:
    # path
    textual_dir = os.path.join('./caches', set)
    feat_dir = os.path.join('./caches', set)
    save_dir = os.path.join('./selected', set)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weight = load_text_feature(textual_dir)

    for scale in range(1, 10):
        print("Getting features.")
        features = load_feature(feat_dir, str(scale), 'val')
        selected_feature = select_image_views(features, clip_weight)
        save_feature(selected_feature, save_dir, str(scale), 'val')
        if not set == 'imagenet':
            features = load_feature(feat_dir, str(scale), 'test')
            selected_feature = select_image_views(features, clip_weight)
            save_feature(selected_feature, save_dir, str(scale), 'test')