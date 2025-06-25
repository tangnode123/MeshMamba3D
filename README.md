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
The non-uniformity, irregularity, and complex topological structures of 3D mesh data pose significant challenges for deep learning. Existing methods predominantly rely on specialized mesh convolutional algorithms or graph-structured modeling, often requiring the design of complex operators. Given that point cloud models are insensitive to input order, can efficient mesh learning be achieved by treating mesh faces as points and feeding them into a point cloud model, with local relationships constructed via KNN? First, this paper theoretically analyzes the commonalities between point clouds and meshes, as well as the effectiveness of KNN in capturing local topological relationships in meshes. Second, to address the non-uniformity of meshes, a graph farthest point sampling method is proposed to optimize the sampling strategy. Finally, based on the Mamba3D model, MeshMamba3D is developed. To address the limitation of single-scale local geometric features in Mamba3D, a multi-scale Local Norm Pooling module is introduced to balance feature attention across different scales. Experimental results show that MeshMamba3D achieves a state-of-the-art accuracy of 94.81% on the Manifold40 dataset, validating the feasibility and effectiveness of applying point cloud methods to mesh learning.
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
| Pre-training | ShapeNet |                                                             | - | [model](https://drive.google.com/file/d/1iSDbm-05w46UQoQGdpo8tYCtBoPkkb-a/view?usp=drive_link) |
| Classification | ModelNet40 | 94.17% | [model](https://drive.google.com/file/d/1EmNXXM58rSUmb8qfZ2DXsqfPqtTz84jy/view?usp=drive_link) | 94.45% | [model](https://drive.google.com/file/d/1Gwy8EN4zaSOHXCRW0WnZ4TVLANvBBED4/view?usp=drive_link) |
| Classification | ScanObjectNN | 92.88% | [model](https://drive.google.com/file/d/1DQx_5t9DNSIT11zLh1LZWJ5I3zgDXfmM/view?usp=sharing) | 93.05% | [model](https://github.com/LMD0311/PointMamba/releases/download/ckpts/scan_objbg_pretrain.pth) |
| Part Segmentation | ShapeNetPart | - | - | 90.26% mIoU | [model](https://drive.google.com/file/d/1hQnB8uGzFGXUWXzM9ihjobIE-O9h9c2v/view?usp=sharing) |

## Qualitative Results

## Acknowledgement
Mesh-RWKV is built with reference to the code of the following projects: [Mamba3D](https://github.com/lzhengning/SubdivNet), [MeshMAE](https://github.com/liang3588/MeshMAE)). Thanks for their awesome work!
