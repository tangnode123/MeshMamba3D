import numpy as np
import trimesh
from collections import deque
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path


def augment_points(pts):
    # scale
    pts = pts * np.random.uniform(0.8, 1.25)

    # translation
    translation = np.random.uniform(-0.1, 0.1)
    pts = pts + translation

    return pts


def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh.vertices = rotation.apply(mesh.vertices)
    return mesh


def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh


def mesh_normalize(mesh: trimesh.Trimesh):
    vertices = mesh.vertices - mesh.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh.vertices = vertices
    return mesh


def load_mesh(path, normalize=False, augments=[], request=[]):
    mesh = trimesh.load_mesh(path, process=False)

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)

    if normalize:
        mesh = mesh_normalize(mesh)

    F = mesh.faces
    V = mesh.vertices
    Fs = F.shape[0]

    # 将面索引转换为顶点坐标 [face_num, 3, 3]
    F_vertices = V[F]  # 这会得到一个形状为 [face_num, 3, 3] 的数组

    face_center = F_vertices.mean(axis=1)  # 计算每个面的中心点
    F_vertices = F_vertices.reshape(Fs, 9)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])
    
    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)
    adj=mesh.face_adjacency
    rows = adj[:, 0]
    cols = adj[:, 1]
    data = np.ones(len(rows))
    adj_matrix = coo_matrix((data, (rows, cols)), shape=(Fs, Fs))
    
    dist_matrix = shortest_path(adj_matrix, directed=False, unweighted=True)
    return F_vertices, feats, Fs, dist_matrix

class ClassificationDataset(Dataset):
    def __init__(self, dataroot, train=True, augment=False, in_memory=False):
        super().__init__()
        self.augment = augment
        self.in_memory = in_memory
        self.dataroot = Path(dataroot)
        self.augments = []
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs']

        self.mesh_paths = []
        self.labels = []
        self.dist_matrices_chace = {}
        self.browse_dataroot()

    def browse_dataroot(self):
        self.shape_classes = sorted([x.name for x in self.dataroot.iterdir() if x.is_dir()])
        for obj_class in self.dataroot.iterdir():
            if obj_class.is_dir():
                label = self.shape_classes.index(obj_class.name)
                for obj_path in (obj_class / self.mode).iterdir():
                    if obj_path.is_file():
                        self.mesh_paths.append(obj_path)
                        self.labels.append(label)
        self.mesh_paths = np.array(self.mesh_paths)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        faces, feats, Fs, adj= load_mesh(self.mesh_paths[idx],
                                     normalize=True,
                                     augments=self.augments,
                                     request=self.feats)
        label = self.labels[idx]
        return {
            'faces': torch.from_numpy(faces.copy()).float(),
            'feats': torch.from_numpy(feats.copy()).float(),
            'Fs': Fs,
            'adj': torch.from_numpy(adj).int(), 
            'label': label,
            'path': str(self.mesh_paths[idx])
        }
class ShapeNetDataset(Dataset):
    def __init__(self, dataroot, augment=False, in_memory=False):
        super().__init__()
        self.augment = augment
        self.in_memory = in_memory
        self.dataroot = Path(dataroot)
        self.augments = []
        self.feats = ['area', 'face_angles', 'curvs']
        
        self.mesh_paths = []
        self.labels = []
        self.browse_dataroot()

    def browse_dataroot(self):
        self.shape_classes = sorted([x.name for x in self.dataroot.iterdir() if x.is_dir()])
        for obj_class in self.dataroot.iterdir():
            for obj_path in obj_class.iterdir():
                if obj_path.is_file():
                    self.mesh_paths.append(obj_path)
                    self.labels.append(self.shape_classes.index(obj_class.name))
        self.mesh_paths = np.array(self.mesh_paths)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        faces, feats, Fs, adj = load_mesh(self.mesh_paths[idx],
                                     normalize=True,
                                     augments=self.augments,
                                     request=self.feats)
        label = self.labels[idx]
        return faces,feats,Fs,label,str(self.mesh_paths[idx]), torch.from_numpy(adj).int()

def load_segment(path):
    with open(path) as f:
        segment = json.load(f)
    sub_labels = np.array(segment['sub_labels'])
    # sub_labels=np.loadtxt(path,dtype=np.int64)

    return sub_labels
       # /root/onethingai-tmp/Mesh/human_seg-1024 
class SegmentationDataset(Dataset):
    def __init__(self, dataroot, train=True, augments=False, in_memory=False):
        super().__init__()
        self.augment = augments
        self.in_memory = in_memory
        self.dataroot = Path(dataroot)
        self.augments = []
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs', 'center', 'normal']

        self.mesh_paths = []
        self.raw_paths=[]
        self.seg_paths = []
        # self.labels = []
        self.browse_dataroot()

    def browse_dataroot(self):
        for obj_path in (Path(self.dataroot)/ self.mode).iterdir():
            if obj_path.suffix == '.obj':
                obj_name = obj_path.stem
                seg_path = obj_path.parent / (obj_name + '.json')
                self.mesh_paths.append(str(obj_path))
                self.seg_paths.append(str(seg_path))
        self.mesh_paths = np.array(self.mesh_paths)
        self.seg_paths = np.array(self.seg_paths)

    def __len__(self):
        return len(self.mesh_paths) 
    def __getitem__(self, idx):
        faces, feats, Fs, adj= load_mesh(self.mesh_paths[idx],
                                     normalize=True,
                                     augments=self.augments,
                                     request=self.feats)
        labels = load_segment(self.seg_paths[idx])
        return {
            'faces': torch.from_numpy(faces.copy()).float(),
            'feats': torch.from_numpy(feats.copy()).float(),
            'Fs': Fs,
            'adj': torch.from_numpy(adj).int(), 
            'label': labels
        }
    


    # from tqdm import tqdm  # 导入 tqdm

    # # 假设 train_cls_loader 是你的 DataLoader
    # for i, (faces, feats, Fs, label, mesh_path) in enumerate(tqdm(train_cls_loader, desc="检查 Fs 值", dynamic_ncols=True)):
    #     if Fs[0].item() != 1024:
    #         print(f"\n❌ Fs[0] 的值不等于 1024，实际值为: {Fs[0].item()}")
    #         print(f"文件路径: {mesh_path}")
    # 将feats转化为tensor，feats是简单的列表

