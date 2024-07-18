import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets.imagenet import ImageNet
import clip
from utils import *
import pdb

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='yaml format')
    parser.add_argument('--shot', default=1, type=int)
    args = parser.parse_args()

    return args


def run_training_free(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    clip_logits = 100. * test_features @ clip_weights
    # Training-free
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    training_free_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(training_free_logits, test_labels)
    print("**** Training-free accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)

    clip_logits = 100. * test_features @ clip_weights
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    training_free_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(training_free_logits, test_labels)
    print("**** Training-free accuracy: {:.2f}. ****\n".format(acc))
    out_dir = 'output/' + cfg['dataset'] + "/" + str(cfg['shots'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = out_dir + '/training-free.log'
    out_file = open(out_path, mode='a', encoding='utf-8')
    print(cfg['shots'], file=out_file)
    print("Training-free accuracy: {:.2f}\n".format(acc), file=out_file)
    out_file.close()


def run_training_need(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            training_need_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(training_need_logits, target)

            acc = cls_acc(training_need_logits, target)
            correct_samples += acc / 100 * len(training_need_logits)
            all_samples += len(training_need_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        training_need_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(training_need_logits, test_labels)

        print("**** Training-need accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** Training-need best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)
    print("\n-------- Evaluating on the test set. --------")
   
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    training_need_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(training_need_logits, test_labels)
    print("**** Training-need accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))
    out_dir = 'output/' + cfg['dataset'] + "/" + str(cfg['shots'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = out_dir + '/training-need.log'
    out_file = open(out_path, mode='a', encoding='utf-8')
    print(cfg['shots'], file=out_file)
    print("Training-need accuracy: {:.2f}\n".format(acc), file=out_file)
    out_file.close()

def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['shots'] = args.shot
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(cfg)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "val")

    # ------------------------------------------ Training-free ------------------------------------------
    run_training_free(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)

    # ------------------------------------------ Training-need ------------------------------------------
    run_training_need(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F)


if __name__ == '__main__':
    main()