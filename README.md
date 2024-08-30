# VCR: Rethinking Visual Content Refinement in Low-Shot CLIP Adaptation
Implementation of our paper: [Rethinking Visual Content Refinement in Low-Shot CLIP Adaptation](https://arxiv.org/pdf/2407.14117)

## Introduction
In this paper, we systematically analyze current CLIP adaptation methods and discover a perceived bias issue in these adaptations. Specifically, the perceived bias issue can be manifested into 2 aspects, namely **Component** and **Environmental** bias, as depicted in the following figure. To settle this issue, we propose the **V**isual **C**ontent **R**efinement(**VCR**) method. Concretely, given an image, we firstly decompose it into multiple scales, where each scale contains sufficient local views, then we refine the content at each scale, and finally we construct its refined representation to boost further adaptation methods.<br>
<div align=center>
  <img src="https://github.com/injadlu/VCR/blob/main/Figure-1.svg">
</div>

## Overview
An overview of our **V**isual **C**ontent **R**efinement(**VCR**).
<div align=center>
  <img src="https://github.com/injadlu/VCR/blob/main/Pipeline.svg">
</div>

## Running
the running can be categories into following steps:
1. run  feat-extraction.py 
2. run  feat-selection.py
3. run  feat-merge.py
4. run  main_imagenet.py for ImageNet, and main.py for other 10 datasets

## contact:
**Any problems please contact me at jackie64321@gmail.com**

## References
```
@article{lu2024rethinking,
  title={Rethinking Visual Content Refinement in Low-Shot CLIP Adaptation},
  author={Lu, Jinda and Wang, Shuo and Hao, Yanbin and Liu, Haifeng and Wang, Xiang and Wang, Meng},
  journal={arXiv preprint arXiv:2407.14117},
  year={2024}
}
```
