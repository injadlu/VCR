import os
import math
import random
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms
import pdb
from torchvision.transforms import RandomResizedCrop
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
from PIL import Image

class RandomResizedCrop_revise(RandomResizedCrop):
    """
    Modified from torchvision, return positions of cropping boxes
    """
    def __init__(self, size, scale):
        super().__init__(size = size, scale=scale)
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias), (i,j,h,w)

META_FILE = "meta.bin"

def load_meta_file(root, file = None):
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    return torch.load(file)

class ImageNet(ImageFolder):

    dataset_dir = 'imagenet'

    def __init__(self, root):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split = "val"
        self.split_folder = os.path.join(self.image_dir, self.split)
        super().__init__(self.split_folder)

        self.crop_func = RandomResizedCrop_revise(size=224, scale=(0.05, 1.0))
        self.crop_num = 100
        self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        
        wnid_to_classes = load_meta_file(self.image_dir)[0]
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.idx_to_cls = {idx: cls for idx, clss in enumerate(self.classes) for cls in clss}
        self.cls_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    def __getitem__(self, index):
        path, label= self.imgs[index]
        image=Image.open(path).convert('RGB')
        patch_list=[]
        positions= []
        for _ in range(self.crop_num):
            image_, position = self.crop_func(image)
            positions.append(position)
            patch_list.append(self.transform(image_))
        patch_list=torch.stack(patch_list,dim=0)

        return patch_list, positions, label
    
    def get_imgs(self):
        return self.imgs
    
    def get_class_idx(self):
        return self.idx_to_cls, self.cls_to_idx
