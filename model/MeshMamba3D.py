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
        # center_xyz = misc.fps(pts, self.num_group)
        # knn to get the neighborhood
        _, idx = self.knn(pts, center_xyz)
        # adj = misc.Graph_Construction(idx)
        
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        
        # 重组邻域(包含所有13维特征)
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
    def __init__(self, lga_out_dim, k_group_size, alpha, beta, mlp_in_dim, mlp_out_dim, num_group=128, act_layer=nn.SiLU, drop_path=0., norm_layer=nn.LayerNorm):
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
        self.mask=mask
        self.group_num = int((1-mask_ratio) * num_group)+1 if mask else num_group
        self.lnps = nn.ModuleList([
            LNPBlock(lga_out_dim, k, alpha, beta, mlp_in_dim, mlp_out_dim, num_group, act_layer)
            for k in k_group_size
        ])
        
        # # 新增可学习权重参数
        self.fusion_weights = nn.Parameter(torch.ones(len(k_group_size)) / len(k_group_size))
        self.fusion_softmax = nn.Softmax(dim=0)
        
        self.gaf = GlobalAffine(self.group_num, mlp_out_dim, act_layer)
        self.mlp = nn.Conv1d(self.group_num*2, self.group_num, 1)

    def forward(self, center, feat):
        B, G, C = feat.shape
        cls_token = feat[:, 0:1, :]
        feat = feat[:, 1:, :]
        
        
        lnp_outputs = [lnp(center, feat) for lnp in self.lnps]
        
        
        weights = self.fusion_softmax(self.fusion_weights)  # [K]
        weighted_outputs = [w * out for w, out in zip(weights, lnp_outputs)]
        lnp = torch.stack(weighted_outputs, dim=0).sum(dim=0)  # [B, G, C]
        
        ga = self.gaf(feat)
        lnp = torch.cat([ga, lnp], dim=1)
        x = self.mlp(lnp)
        x = torch.cat([cls_token, x], dim=1)   
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
class Mamba3DEncoder(nn.Module):
    def __init__(self, k_group_size=[4,8,16], embed_dim=384, depth=4, drop_path_rate=0., num_group=128, num_heads=6, bimamba_type="v2",mask_ratio=0.6, mask=True):
        super().__init__()
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.num_heads = num_heads
        self.mask_ratio=mask_ratio
        self.blocks = nn.ModuleList([
            Mamba3DBlock(
                dim=embed_dim, #
                k_group_size = self.k_group_size,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate, #
                num_group=self.num_group,
                num_heads=self.num_heads,
                bimamba_type=bimamba_type,
                mask_ratio=self.mask_ratio,
                mask=mask
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
        for _, block in enumerate(self.blocks):
              x = block(center, x + pos)
        return x
    
class MambaDecoder(nn.Module):
    def __init__(self, k_group_size=[4,8,16], embed_dim=768, depth=4, drop_path_rate=0., num_group=128, num_heads=6, bimamba_type="v2",mask_ratio=0.6):
        super().__init__()
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.num_heads = num_heads
        self.mask_ratio=mask_ratio
        self.blocks = nn.ModuleList([
            Mamba3DBlock(
                dim=embed_dim, #
                k_group_size = self.k_group_size,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate, #
                num_group=self.num_group,
                num_heads=self.num_heads,
                bimamba_type=bimamba_type,
                mask_ratio=self.mask_ratio,
                mask=False
                )
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self,center, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
              x = block(center, x + pos)
        x = self.head(self.norm(x[:, -return_token_num:]))
        return x

    
class MaskMamba(nn.Module):
    def __init__(self, mask_ratio=0.6, mask_type='rand', encoder_dims=384, k_group_size=[4,8,16], num_groups=128,
                trans_dim=384, depth=12, drop_path_rate=0.1, num_heads=6, bimamba_type='v2'):
        super().__init__()
        self.mask_ratio = mask_ratio 
        self.trans_dim = trans_dim
        self.depth = depth 
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads
        self.num_groups= num_groups
        # embedding
        self.encoder_dims =  encoder_dims
        self.k_group_size = k_group_size
        self.bimamba_type=bimamba_type
        self.encoder=Encoder(encoder_channel=self.encoder_dims)

        self.mask_type=mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim)
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = Mamba3DEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            num_group=self.num_groups,
            k_group_size=self.k_group_size,
            bimamba_type=self.bimamba_type,
            mask_ratio=self.mask_ratio,
            mask=True
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def _make_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos
    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        

        return overall_mask.to(center.device) # B G
    
    def forward(self,  neighborhood, center, noaug = False):
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug)
        else:
            bool_masked_pos = self._make_center_block(center, noaug)
        group_input_tokens=self.encoder(neighborhood)
        batch_size, seq_len, C = group_input_tokens.shape

        x_vis=group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        mask_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(mask_center)

        x_vis=self.blocks(mask_center, x_vis, pos)
        x_vis=self.norm(x_vis)
        return x_vis, bool_masked_pos
    
class MeshMamba3D_MAE(nn.Module):
    def __init__(self, mask_ratio=0.6, mask_type='rand', trans_dim=384, depth=12, drop_path_rate=0.1,
                 num_heads=6, group_size=32, num_group=128, encoder_dims=384, loss='cdl2', decoder_depth=4,
                 decoder_num_heads=6, decoder_dims=384, decoder_with_lfa=False, ordering=False, center_local_k=[4,8,16], 
                 bimamba_type='v4',weight=0.5):
        super().__init__()
        self.trans_dim = trans_dim
        self.MAE_encoder = MaskMamba(mask_ratio=mask_ratio, mask_type=mask_type, encoder_dims=encoder_dims, k_group_size=center_local_k, 
                                     num_groups=num_group,trans_dim=self.trans_dim, depth=depth, drop_path_rate=drop_path_rate, num_heads=num_heads,bimamba_type=bimamba_type)
        self.group_size = group_size
        self.num_group = num_group
        self.drop_path_rate = drop_path_rate
        self.weight = weight
        self.mask_token=nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.decoder_dims = decoder_dims
        dpr=[x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = MambaDecoder(
            k_group_size=center_local_k,
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_group=self.num_group,
            num_heads=self.decoder_num_heads,
        )
        self.group_divider = Group_gffs(num_group = self.num_group, group_size = self.group_size)
        self.pred_ver = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )
        self.pred_feats = nn.Sequential(
            nn.Conv1d(self.trans_dim,10*self.group_size, 1)
        )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = loss
        self.build_loss_func(self.loss)
    def build_loss_func(self, loss_type):
        if loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl1':
            self.loss_func = ChamferDistanceL1().cuda()
        else:
            raise NotImplementedError
    def forward(self, pts, adj):
        pts=pts.permute(0, 2, 1).contiguous()
        keep_indices = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]
        neighborhood, center= self.group_divider(pts,adj)
        neighborhood_center=neighborhood[:,:,:,4:7]
        neighborhood=neighborhood[:,:,:,keep_indices]
        
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B,_,C = x_vis.shape

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token=self.mask_token.expand(B,N,-1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(center, x_full, pos_full, N)
        
        B, M, C = x_rec.shape
        
        rebuild_points = self.pred_ver(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B , M, -1, 3) 
        rebuild_feats = self.pred_feats(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B ,M, -1, 10)
        gt_feats = neighborhood[mask].reshape(B,M,-1,10)

        gt_shape=neighborhood_center[mask].reshape(B,M,-1,3)
        
        feats_loss=F.mse_loss(rebuild_feats,gt_feats)
        shape_loss=self.loss_func(rebuild_points,gt_shape)
        loss1=feats_loss+self.weight*shape_loss
        return loss1
        
class MeshMamba3D(nn.Module):
    def __init__(self, trans_dim=384, depth=12, drop_path_rate=0.2, cls_dim=40, num_heads=6, 
                 group_size=32, num_group=128, encoder_dims=384, ordering=False, center_local_k=[4,8,16], 
                 bimamba_type="v4"):
        super().__init__()

        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.cls_dim = cls_dim
        self.num_heads = num_heads

        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims

        self.group_divider = Group_gffs(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim)
        )

        self.ordering = ordering
        
        self.k_group_size = center_local_k # default=8

        self.bimamba_type = bimamba_type

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        #define the encoder
        self.blocks = Mamba3DEncoder(
            embed_dim=self.trans_dim,
            k_group_size=self.k_group_size,
            depth=self.depth,
            drop_path_rate=dpr,
            num_group=self.num_group,
            num_heads=self.num_heads,
            bimamba_type=self.bimamba_type,
            mask=False
        )
        #embed_dim=768, depth=4, drop_path_rate=0.

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, adj):
        pts=pts.permute(0, 2, 1).contiguous()
        keep_indices = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]
        # neighborhood, center= self.group_divider(pts) # B G K 3
        neighborhood, center= self.group_divider(pts, adj) # B G K 3
        neighborhood=neighborhood[:,:,:,keep_indices]
        group_input_tokens = self.encoder(neighborhood)  # B G C


        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center) # B G C

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(center, x, pos) # enter transformer blocks
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0] + x[:, 1:].mean(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret  


