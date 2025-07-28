import numpy as np
import torch
import fps_cuda
from pointnet2_ops import pointnet2_utils

def mffs(batch_face_adjacency, num_samples=512):
    with torch.no_grad():  
        ffs=fps_cuda.forward(batch_face_adjacency, num_samples)
    return ffs


def gather_select_features(features, indices):
    
    indices = indices.long() if indices.dtype != torch.int64 else indices
    
    
    expanded_indices = indices.unsqueeze(-1).expand(-1, -1, features.size(-1))
    
    
    selected_features = torch.gather(features, 1, expanded_indices)
    return selected_features
    
def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data