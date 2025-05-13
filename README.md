# MeshMamba3D: A State-Space Model Enabling Meshes to be Learned like Point Clouds
<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/abs/2305.14314" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2411.10499-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://hithqd.github.io/projects/PointRWKV/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href='http://www.apache.org/licenses/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Licence-Apache_2.0-orange' alt='webpage'>
  </a>
</div>

## Abstract
Transformers have demonstrated powerful capabilities in mesh learning tasks, but their quadratic complexity limits their scalability to complex mesh structures, placing a significant burden on computational resources. Recently, RWKV, a novel deep sequence model, has shown efficient sequence modeling potential in NLP tasks. In this study, we propose ​MeshRWKV, a linear-complexity model derived from the RWKV model, specifically adapted and optimized for mesh learning tasks. Specifically, taking embedded mesh patches as input, we first introduce improved multi-headed matrix-valued states and a dynamic attention recurrence mechanism within the MeshRWKV blocks to explore global structural features of the mesh. To simultaneously capture local geometric details, we design a parallel branch that efficiently encodes the local topology of the mesh through adjacency-based graph convolutional networks and a graph stabilizer. Additionally, we design MeshRWKV as a multi-scale framework for hierarchical feature learning of meshes, supporting downstream tasks such as mesh segmentation, classification, and reconstruction. Extensive experiments on various mesh learning tasks demonstrate that our proposed MeshRWKV outperforms Transformer- and Mamba-based models while significantly reducing floating-point operations (FLOPs) by approximately 40%, showcasing its potential as a foundational model for efficient mesh learning.
## Overview
![image](../main/asset/overview.png) 

## Dataset
Here, we provide the download links of the datasets for pre-train, classification and segmentation. 

- ModelNet40 [here](https://drive.google.com/file/d/1Cf5zQqN-kAXF7OiZZ0hNNPT59J-Ijy-i/view?usp=sharing)
- Humanbody [here](https://drive.google.com/file/d/1XaqMC8UrIZ_N77gN83PI3VK03G5IJskt/view?usp=sharing)
- COSEG-aliens [here](https://drive.google.com/file/d/12QCv2IUySoSzxeuvERGzgmE7YY3QzjfW/view?usp=sharing)
- ShapeNet [here](https://shapenet.org)（we also provide the processed ShapeNet dataset as [here](https://pan.baidu.com/s/1w044bIgiCMY0WXD9QviUJg?pwd=ufb9)）

## Model Zoo
| Task | Dataset | Acc.(Scratch) | Download (Scratch) | Acc.(pretrain) | Download (Finetune) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Pre-training | ShapeNet |                                                             | - | [model](https://drive.google.com/file/d/1QXB1msBljSOPJhx5sGYpueOdCrY0yaCO/view?usp=sharing) |
| Classification | ModelNet40 | 94.66% | [model](https://drive.google.com/file/d/1iMN-iAGjKWAUpAoIOqaS9e_CI_wk5nhE/view?usp=sharing) | 96.16% | [model](https://drive.google.com/file/d/11iBDSwdTIpHldUGWIsFp9orbCwNf69fB/view?usp=sharing) |
| Classification | ScanObjectNN | 92.88% | [model](https://drive.google.com/file/d/1DQx_5t9DNSIT11zLh1LZWJ5I3zgDXfmM/view?usp=sharing) | 93.05% | [model](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_objbg_pretrain.pth) |
| Part Segmentation | ShapeNetPart | - | - | 90.26% mIoU | [model](https://drive.google.com/file/d/1hQnB8uGzFGXUWXzM9ihjobIE-O9h9c2v/view?usp=sharing) |

## Qualitative Results

## Acknowledgement
Mesh-RWKV is built with reference to the code of the following projects: [Mamba3D](https://github.com/lzhengning/SubdivNet), [MeshMAE](https://github.com/liang3588/MeshMAE)). Thanks for their awesome work!
