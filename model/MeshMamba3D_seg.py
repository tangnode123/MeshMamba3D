import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import misc
import numpy as np

from knn_cuda import KNN
from timm.models.layers import DropPath, PatchEmbed
from model.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import random

from model.bimamba_ssm.modules.mamba_simple import Mamba
from model.bimamba_ssm.utils.generation import GenerationMixin
from model.bimamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from timm.models.layers import DropPath, trunc_normal_

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(10, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 10
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 10)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)
        
class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, x):
        '''
            input: B N 13
            ---------------------------
            output: 
                - neighborhood: B G M 13
                - center : B G 3
        '''
        batch_size, num_points, _ = x.shape
        
        # ffs the centers index out
        pts=x[:,:,4:7].contiguous()
        center_xyz = misc.fps(pts, self.num_group)
        # knn to get the neighborhood
        _, idx = self.knn(pts, center_xyz)
        
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        neighborhood = x.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, -1).contiguous()
        
        neighborhood[:, :, :, 4:7] = neighborhood[:, :, :, 4:7] - center_xyz.unsqueeze(2)
        
        return neighborhood, center_xyz   
        
        
class Group_gffs(nn.Module):  # GFFS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, x, adj):
        '''
            input: B N 13
            ---------------------------
            output: 
                - neighborhood: B G M 13
                - center : B G 3
        '''
        batch_size, num_points, _ = x.shape
        
        # ffs the centers index out
        pts=x[:,:,4:7].contiguous()
        center_index = misc.mffs(adj, self.num_group)
        center_xyz=misc.gather_select_features(x, center_index)[:,:,4:7]
        
        _, idx = self.knn(pts, center_xyz)
        
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        neighborhood = x.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, -1).contiguous()
        
        neighborhood[:, :, :, 4:7] = neighborhood[:, :, :, 4:7] - center_xyz.unsqueeze(2)
        
        return neighborhood, center_xyz


    
class GroupFeature(nn.Module):  # FPS + KNN
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size  # the first is the point itself
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, feat):
        '''
            input: 
                xyz: B N 3
                feat: B N C
            ---------------------------
            output: 
                neighborhood: B N K 3
                feature: B N K C
        '''
        batch_size, num_points, _ = xyz.shape # B N 3 : 1 128 3
        C = feat.shape[-1]

        center = xyz
        # knn to get the neighborhood
        _, idx = self.knn(xyz, xyz) # B N K : get K idx for every center
        assert idx.size(1) == num_points # N center
        assert idx.size(2) == self.group_size # K knn group
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :] # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous() # 1 128 8 3
        neighborhood_feat = feat.contiguous().view(-1, C)[idx, :] # BxNxK C 128x8 384   128*26*8
        assert neighborhood_feat.shape[-1] == feat.shape[-1]
        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size, feat.shape[-1]).contiguous() # 1 128 8 384
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        
        return neighborhood, neighborhood_feat
    

class K_Norm(nn.Module):
    def __init__(self, out_dim, k_group_size, alpha, beta):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))

    def forward(self, lc_xyz, lc_x):
        #get knn xyz and feature 
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x) # B G K 3, B G K C

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2) # B G 1 C
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz) # B G 1 3

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5) # B G K 3

        B, G, K, C = knn_x.shape

        # Feature Expansion
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1) # B G K 2C

        # Affine
        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat 
        
        # Geometry Extraction
        knn_x_w = knn_x.permute(0, 3, 1, 2) # B 2C G K

        return knn_x_w

# Pooling
class K_Pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        e_x = torch.exp(knn_x_w) # B 2C G K
        up = (knn_x_w * e_x).mean(-1) # # B 2C G
        down = e_x.mean(-1)
        lc_x = torch.div(up, down)
        # lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1) # B 2C G K -> B 2C G
        return lc_x
    
class Post_ShareMLP(nn.Module):
    def __init__(self, in_dim, out_dim, permute=True):
        super().__init__()
        self.share_mlp = torch.nn.Conv1d(in_dim, out_dim, 1)
        self.permute = permute
    
    def forward(self, x):
        # x: B 2C G mlp-> B C G  permute-> B G C
        if self.permute:
            return self.share_mlp(x).permute(0, 2, 1)
        else:
            return self.share_mlp(x)
    
# K_Norm + K_Pool + Shared MLP
class LNPBlock(nn.Module):
    def __init__(self, lga_out_dim, k_group_size, alpha, beta, mlp_in_dim, mlp_out_dim, num_group=128, act_layer=nn.SiLU, drop_path=0., norm_layer=nn.LayerNorm,):
        super().__init__()
        '''
        lga_out_dim: 2C
        mlp_in_dim: 2C
        mlp_out_dim: C
        x --->  (lga -> pool -> mlp -> act) --> x

        '''
        self.num_group = num_group
        self.lga_out_dim = lga_out_dim

        self.lga = K_Norm(self.lga_out_dim, k_group_size, alpha, beta)
        self.kpool = K_Pool()
        self.mlp = Post_ShareMLP(mlp_in_dim, mlp_out_dim)
        self.pre_norm_ft = norm_layer(self.lga_out_dim)

        self.act = act_layer()
        
    
    def forward(self, center, feat):
        # feat: B G+1 C
        B, G, C = feat.shape
        # cls_token = feat[:,0,:].view(B, 1, C)
        # feat = feat[:,1:,:] # B G C

        lc_x_w = self.lga(center, feat) # B 2C G K 
        
        lc_x_w = self.kpool(lc_x_w) # B 2C G : 1 768 128

        # norm([2C])
        lc_x_w = self.pre_norm_ft(lc_x_w.permute(0, 2, 1)) #pre-norm B G 2C
        lc_x = self.mlp(lc_x_w.permute(0, 2, 1)) # B G C : 1 128 384
        
        lc_x = self.act(lc_x)
        
        # lc_x = torch.cat((cls_token, lc_x), dim=1) # B G+1 C : 1 129 384
        return lc_x

class GlobalAffine(nn.Module):
    def __init__(self, gourp_num, out_dim, act_layer=nn.SiLU):
        super().__init__()
        self.group_num=gourp_num
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, out_dim))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.shared_mlp = torch.nn.Conv1d(gourp_num, gourp_num, 1)
        self.act = act_layer()

    def forward(self, features):
        
        # 全局归一化
        mean = features.mean(dim=1, keepdim=True)  # [B, 1, C]
        std = features.std(dim=1, keepdim=True)    # [B, 1, C]
        normalized = (features - mean) / (std + 1e-5)
        
        # 全局仿射
        x = self.affine_alpha * normalized + self.affine_beta
        

        # 全局MLP
        x = self.shared_mlp(x)
        x = self.act(x)

        return x


class MSLNPBlock(nn.Module):
    def __init__(self, lga_out_dim, alpha, beta, mlp_in_dim, mlp_out_dim, 
                 k_group_size=[4, 8, 16], num_group=128, act_layer=nn.SiLU, 
                 drop_path=0., norm_layer=nn.LayerNorm, mask_ratio=0.6, mask=True):
        super().__init__()
        self.group_num = int((1-mask_ratio) * num_group)+1 if mask else num_group
        self.lnps = nn.ModuleList([
            LNPBlock(lga_out_dim, k, alpha, beta, mlp_in_dim, mlp_out_dim, num_group, act_layer)
            for k in k_group_size
        ])
        
        # 新增可学习权重参数
        self.fusion_weights = nn.Parameter(torch.ones(len(k_group_size)) / len(k_group_size))
        self.fusion_softmax = nn.Softmax(dim=0)
        
        self.gaf = GlobalAffine(self.group_num, mlp_out_dim, act_layer)
        self.mlp = nn.Conv1d(self.group_num*2, self.group_num, 1)

    def forward(self, center, feat):
        B, G, C = feat.shape
        # cls_token = feat[:, 0:1, :]
        # feat = feat[:, 1:, :]
        
        # 多尺度特征提取
        lnp_outputs = [lnp(center, feat) for lnp in self.lnps]
        
        # 自学习加权融合 (改进部分)
        weights = self.fusion_softmax(self.fusion_weights)  # [K]
        weighted_outputs = [w * out for w, out in zip(weights, lnp_outputs)]
        lnp = torch.stack(weighted_outputs, dim=0).sum(dim=0)  # [B, G, C]
        
        ga = self.gaf(feat)
        lnp = torch.cat([ga, lnp], dim=1)
        x = self.mlp(lnp)
        # x = torch.cat([cls_token, x], dim=1)   
        return x


class Mamba3DBlock(nn.Module):
    def __init__(self, 
                dim, 
                mlp_ratio=4., 
                drop=0., 
                drop_path=0., 
                act_layer=nn.SiLU, 
                norm_layer=nn.LayerNorm,
                k_group_size=[4,8,16], 
                alpha=100, 
                beta=1000,
                num_group=128,
                num_heads=6,
                bimamba_type="v2",
                mask_ratio=0.6,
                mask=False
                ):
        super().__init__()
        # self.norm1 = DyT(dim, init_alpha=0.5)
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = DyT(dim, init_alpha=0.5)
        self.norm2 = norm_layer(dim)
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.mask_ratio=mask_ratio
        
        self.num_heads = num_heads
        
        # self.lfa = LNPBlock(lga_out_dim=dim*2, 
        #             k_group_size=self.k_group_size, 
        #             alpha=alpha, 
        #             beta=beta, 
        #             mlp_in_dim=dim*2, 
        #             mlp_out_dim=dim, 
        #             num_group=self.num_group,
        #             act_layer=act_layer,
        #             drop_path=drop_path,
        #             # num_heads=self.num_heads, # uncomment this line if use attention
        #             norm_layer=norm_layer,
        #             )
        self.lfa = MSLNPBlock(lga_out_dim=dim*2, 
                    alpha=alpha, 
                    beta=beta, 
                    mlp_in_dim=dim*2, 
                    mlp_out_dim=dim, 
                    k_group_size=self.k_group_size, 
                    num_group=self.num_group,
                    act_layer=act_layer,
                    drop_path=drop_path,
                    # num_heads=self.num_heads, # uncomment this line if use attention
                    norm_layer=norm_layer,
                    mask_ratio=self.mask_ratio,
                    mask=mask
                    )

        self.mixer = Mamba(dim, bimamba_type=bimamba_type)

    def shuffle_x(self, x, shuffle_idx):
        pos = x[:, None, 0, :]
        feat = x[:, 1:, :]
        shuffle_feat = feat[:, shuffle_idx, :]
        x = torch.cat([pos, shuffle_feat], dim=1)
        return x

    def mamba_shuffle(self, x):
        G = x.shape[1] - 1 #
        shuffle_idx = torch.randperm(G)
        # shuffle_idx = torch.randperm(int(0.4*self.num_group+1)) # 1-mask
        x = self.shuffle_x(x, shuffle_idx) # shuffle

        x = self.mixer(self.norm2(x)) # layernorm->mamba

        x = self.shuffle_x(x, shuffle_idx) # un-shuffle
        return x

    def forward(self, center, x):
        # x + norm(x)->lfa(x)->dropout
        x = x + self.drop_path(self.lfa(center,self.norm1(x))) # x: 32 129 384. center: 32 128 3

        # x + norm(x)->mamba(x)->dropout
        # x = x + self.drop_path(self.mamba_shuffle(x))
        x = x + self.drop_path(self.mixer(self.norm2(x)))
    
        return x


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
    

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class MeshMamba3DEncoder_Seg(nn.Module):
    def __init__(self, k_group_size=8, embed_dim=768, depth=4, drop_path_rate=0., num_group=128, num_heads=6, bimamba_type="v2",):
        super().__init__()
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.num_heads = num_heads
        self.dim=embed_dim
        self.blocks = nn.ModuleList([Mamba3DBlock(
                dim=self.dim, #
                k_group_size = self.k_group_size,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate, #
                num_group=self.num_group,
                num_heads=self.num_heads,
                bimamba_type=bimamba_type,
                mask_ratio=0.,
                mask=False
        )
            for i in range(depth)])

    def forward(self, center, x, pos):
        '''
        INPUT:
            x: patched point cloud and encoded, B G+1 C, 8 128+1=129 384
            pos: positional encoding, B G+1 C, 8 128+1=129 384
        OUTPUT:
            x: x after transformer block, keep dim, B G+1 C, 8 128+1=129 384
            
        NOTE: Remember adding positional encoding for every block, 'cause ptc is sensitive to position
        '''
        # TODO: pre-compute knn (GroupFeature)
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(center, x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list

class MeshMamba3D_Seg(nn.Module):
    def __init__(self, trans_dim=384, depth=12, drop_path_rate=0.2, cls_dim=4, num_heads=6, 
                 group_size=32, num_group=128, encoder_dims=384, ordering=False, center_local_k=[4,8,16,32], 
                 bimamba_type="v4"):
        super().__init__()
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.cls_dim = cls_dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.k_group_size = center_local_k
        self.bimamba_type = bimamba_type
        
        self.group_divider = Group_gffs(num_group=self.num_group, group_size=self.group_size)
        
        self.encoder_dims = encoder_dims
        
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = MeshMamba3DEncoder_Seg(
            embed_dim=self.trans_dim,
            k_group_size=self.k_group_size,
            depth=self.depth,
            drop_path_rate=dpr,
            num_group=self.num_group,
            num_heads=self.num_heads,
            bimamba_type=self.bimamba_type,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 10,
                                                        mlp=[self.trans_dim * 4, 1024])
        self.convs1 = nn.Conv1d(3328, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)

        self.relu = nn.ReLU()
    def forward(self, pts, adj):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)  # B N 3
        keep_indices = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts,adj)
        faces_feat=pts[:,:,keep_indices]
        pts=pts[:,:,4:7]
        neighborhood=neighborhood[:,:,:,keep_indices]
        
        group_input_tokens = self.encoder(neighborhood)  # B G N

        pos = self.pos_embed(center)
        x=group_input_tokens
        # transformer
        feature_list = self.blocks(center,x, pos)
        feature_list = [self.norm(y).transpose(-1, -2).contiguous() for y in feature_list]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
        x_max = torch.max(x,2)[0]
        x_avg = torch.mean(x,2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1) #1152*2 + 64
        

        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), faces_feat.transpose(-1, -2), x) #·
        x = torch.cat((f_level_0,x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        return x
