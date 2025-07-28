# MeshMamba3D: A State-Space Model Enabling Meshes to be Learned like Point Clouds

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/abs/2305.14314" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2411.10499-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://github.com/tangnode123/MeshMamba3D' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href='http://www.apache.org/licenses/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Licence-Apache_2.0-orange' alt='webpage'>
  </a>
</div>

## Abstract
The inherent non-uniformity, irregular connectivity, and complex topological structures of 3D mesh data present fundamental challenges for deep learning approaches. Existing methods predominantly rely on specialized mesh convolutional algorithms or graph-structured modeling, often requiring the design of complex operators. Given that point cloud models are insensitive to input order, can efficient mesh learning be achieved by treating mesh faces as points and feeding them into a point cloud model, with local relationships constructed via KNN? First, this paper theoretically analyzes the commonalities between point clouds and meshes, as well as the effectiveness of KNN in capturing local topological relationships in meshes. Second, to address the non-uniformity of meshes, a graph farthest point sampling method is proposed to optimize the sampling strategy. Finally, based on the Mamba3D model, MeshMamba3D is developed. To address the limitation of single-scale local geometric features in Mamba3D, a Multi-Scale Local Norm Pooling block is introduced to balance feature attention across different scales. Experimental validation demonstrates that MeshMamba3D attains a state-of-the-art accuracy of 94.81% on the Manifold40 dataset, confirming the viability and efficacy of adapting point cloud approaches to mesh learning.
## Overview
![image](../main/asset/Overview.png)
## Dataset
Here, we provide the download links of the datasets for pre-train, classification and segmentation. 

- ModelNet40 ([ManiFold40](https://drive.google.com/file/d/1K5jrJlzx7HpDmx8I3ishLld37bVDRi2Z/view?usp=drive_link)) 
- SHREC11 ([spilt10](https://drive.google.com/file/d/1lBT2mE6VQq4hQQiYnqT8Ro279gogPz35/view?usp=drive_link), [spilt16](https://drive.google.com/file/d/1kfpXM-YSiFxyOt4Ra-1JJZdC_9HQ7wi7/view?usp=drive_link)) 
- [Cube Engraving](https://drive.google.com/file/d/1ff5IpJD_AWgisSyjW536-xpFNGO5UQaX/view?usp=drive_link)
- COSEG ([Tele-aliens](https://drive.google.com/file/d/1i1y1pJ8L1p4u941O9z7DCbSPcQvp5J8x/view?usp=drive_link), [Chairs](https://drive.google.com/file/d/16FLKaD00EicnSuhFyzF32Pvfn-qGHdYK/view?usp=drive_link), [Vases](https://drive.google.com/file/d/1jOWIbkp7PW41V5WaXMF_A5H0CNAVPPT9/view?usp=drive_link))
- [ShapeNet](https://shapenet.org)（we also provide the processed ShapeNet dataset as [here](https://drive.google.com/file/d/198ccVlqSNHPzXGCGLVJ59TNKAb65L_xP/view?usp=drive_link)）

## Model Zoo
| Task | Dataset | Accuracy | Download |
| :---- | :---- | :---- | :---- |
| Pre-training | ShapeNet | - | [ckpt](https://drive.google.com/file/d/1iSDbm-05w46UQoQGdpo8tYCtBoPkkb-a/view?usp=drive_link) |
| Classification | ModelNet40 (ManiFlod40)  | 94.81% | [ckpt](https://drive.google.com/file/d/11RkPjKmUky8XGrOrB5ww_kyNwiwT2I44/view?usp=drive_link) |
| Classification | Cube Engraving  | 96.21% | [ckpt](https://drive.google.com/file/d/1AQaNCdsGa5aAqkMrW-4_r2ybtMrz_9oD/view?usp=drive_link) |
| Classification | SHREC11-Split10 | 99.00% | [ckpt](https://drive.google.com/file/d/1BTAM4JvPqJJOziii4LsvO47ysA_FibDx/view?usp=drive_link) |
| Classification | SHREC11-Split16 | 100.00% | [ckpt](https://drive.google.com/file/d/1OuPiEyOWAApNxiOQDRVwNLd8WvP10yRw/view?usp=drive_link) |
| Segmentation | COSEG-Tele-aliens | 91.42% | [ckpt](https://drive.google.com/file/d/1VW2Hcu45pWt2dZEw9tu4YQso9nFqUKKQ/view?usp=drive_link) |
| Segmentation | COSEG-Chairs | 95.49% | [ckpt](https://drive.google.com/file/d/1CGfEozfPQGL0K3XDqWgjdkZBQ670YGsd/view?usp=drive_link) |
| Segmentation | COSEG-Vases | 90.08% | [ckpt](https://drive.google.com/file/d/1vGKsk2YCfR1Hk4QRlZrD-yS-mTQQXjZO/view?usp=drive_link) |

## Requirements
Tested on:
PyTorch == 2.0.1;
python == 3.8.17;
CUDA == 11.7
```
pip install -r requirements.txt
```

```
# Chamfer Distance
cd ./model/chamfer_dist
python setup.py install --user
# GFFS
cd ./extension/ffs_cuda
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Mamba install
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.1
```
## Training
Pretaining:
```
python pretrain.py cfg/pretrain.yaml
```
Classification Task:
```
python train_cls.py cfg/***.yaml
```
Segmentation Task:
```
python train_seg.py cfg/***.yaml
```
## Acknowledgement
MeshMamba3D is built with reference to the code of the following projects: ([Mamba3D](https://github.com/xhanxu/Mamba3D), [MeshMAE](https://github.com/liang3588/MeshMAE)). Thanks for their awesome work!
