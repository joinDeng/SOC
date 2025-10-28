import json
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T

class SpaceGraphDataset(Dataset):
    """
    优化的图数据集类，支持多种构图策略和多任务学习
    """
    def __init__(self, h5_file, index_file, window_size=1440, subwindow_size=64, 
                 stride=32, graph_type='temporal', transform=None, use_cache=True):
        """
        Args:
            h5_file: HDF5文件路径
            index_file: 样本索引文件
            window_size: 主窗口大小
            subwindow_size: 子窗口大小（节点特征序列长度）
            stride: 子窗口滑动步长
            graph_type: 构图类型 ['temporal', 'temporal_bidirectional', 'similarity']
            transform: 数据增强变换
            use_cache: 是否缓存处理后的图数据
        """
        self.h5_file = h5py.File(h5_file, 'r')
        self.samples = json.load(open(index_file))
        self.window_size = window_size
        self.subwindow_size = subwindow_size
        self.stride = stride
        self.graph_type = graph_type
        self.transform = transform
        self.use_cache = use_cache
        
        # 缓存处理后的图数据
        self.cache = {}
        
        # 标签映射
        self.orbit_map = {'LEO': 0, 'MEO': 1, 'HEO': 2}
        self.cat_map = {'payload': 0, 'rocket body': 1, 'debris': 2}
        self.rcs_map = {'small': 0, 'medium': 1, 'large': 2}
        
        # 预计算样本有效性
        self.valid_indices = []
        for idx, rec in enumerate(self.samples):
            nid, s, e = rec['norad_id'], rec['start_idx'], rec['end_idx']
            if nid in self.h5_file and e - s >= self.subwindow_size:
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def _create_temporal_edges(self, num_nodes, bidirectional=True):
        """创建时序边"""
        if bidirectional:
            # 双向时序边：i↔i+1
            edges = []
            for i in range(num_nodes - 1):
                edges.append([i, i+1])
                edges.append([i+1, i])
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # 单向时序边：i→i+1
            edge_index = torch.empty((2, num_nodes-1), dtype=torch.long)
            edge_index[0] = torch.arange(num_nodes-1)
            edge_index[1] = torch.arange(1, num_nodes)
            return edge_index

    def _create_similarity_edges(self, node_features, threshold=0.8):
        """基于特征相似性创建边"""
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(node_features)
        edges = []
        n = len(node_features)
        
        for i in range(n):
            for j in range(i+1, n):
                if similarities[i, j] > threshold:
                    edges.append([i, j])
                    edges.append([j, i])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    def _extract_time_features(self, t):
        """提取丰富的时间特征"""
        dt = (t - t[0]) / 86400.0  # 转换为天
        
        # 多种周期性特征
        seasonal_sin = np.sin(2 * np.pi * dt / 365.0)
        seasonal_cos = np.cos(2 * np.pi * dt / 365.0)
        
        # 周周期
        weekly_sin = np.sin(2 * np.pi * dt / 7.0)
        weekly_cos = np.cos(2 * np.pi * dt / 7.0)
        
        # 日周期（如果数据有日内变化）
        daily_sin = np.sin(2 * np.pi * dt)
        daily_cos = np.cos(2 * np.pi * dt)
        
        time_features = np.column_stack([
            dt, seasonal_sin, seasonal_cos, weekly_sin, weekly_cos, daily_sin, daily_cos
        ])
        return time_features

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        
        # 检查缓存
        if self.use_cache and original_idx in self.cache:
            return self.cache[original_idx]
        
        rec = self.samples[original_idx]
        nid, s, e = rec['norad_id'], rec['start_idx'], rec['end_idx']
        grp = self.h5_file[nid]
        
        # 确保窗口大小一致
        actual_length = min(e - s, self.window_size)
        s = s if e - s <= self.window_size else e - self.window_size
        
        # 读取数据
        pos = grp['pos'][s:e]
        vel = grp['vel'][s:e]
        ncf = grp['ncf'][s:e]
        t = grp['t'][s:e]
        
        # 提取时间特征
        time_features = self._extract_time_features(t)
        
        # 组合序列特征 [T, 19]
        seq = np.concatenate([
            pos, vel, ncf, time_features
        ], axis=1)
        
        # 标签映射
        grid = rec['grid']
        y_orbit = self.orbit_map.get(grid['orbit_class'], 0)
        y_cat = self.cat_map.get(grid['final_cat'], 0)
        y_rcs = self.rcs_map.get(grid['final_rcs'], 0)
        
        # 创建节点：滑动子窗口
        nodes = []
        valid_starts = []
        
        for start in range(0, len(seq) - self.subwindow_size + 1, self.stride):
            subseq = seq[start:start + self.subwindow_size]
            nodes.append(subseq)
            valid_starts.append(start)
        
        if not nodes:
            # 如果序列太短，使用零填充
            padded_seq = np.zeros((self.subwindow_size, seq.shape[1]))
            actual_len = min(len(seq), self.subwindow_size)
            padded_seq[:actual_len] = seq[:actual_len]
            nodes = [padded_seq]
            valid_starts = [0]
        
        nodes = torch.tensor(np.stack(nodes), dtype=torch.float32)  # [N, subwindow_size, 19]
        
        # 创建边
        num_nodes = len(nodes)
        
        if self.graph_type == 'temporal':
            edge_index = self._create_temporal_edges(num_nodes, bidirectional=False)
        elif self.graph_type == 'temporal_bidirectional':
            edge_index = self._create_temporal_edges(num_nodes, bidirectional=True)
        elif self.graph_type == 'similarity':
            similarity_edges = self._create_similarity_edges(node_features, threshold=0.8)
            # 可以在这里添加其他类型的边
            edge_index = similarity_edges
        else:
            edge_index = self._create_temporal_edges(num_nodes, bidirectional=True)
        
        # 创建PyG Data对象
        data = Data(
            x=nodes,                    # 节点特征 [N, subwindow_size, 19]
            edge_index=edge_index,      # 边索引 [2, E]
            y_cat=torch.tensor(y_cat, dtype=torch.long),
            y_rcs=torch.tensor(y_rcs, dtype=torch.long),
            y_orbit=torch.tensor(y_orbit, dtype=torch.long),
            num_nodes=num_nodes,
            norad_id=nid,
            window_start=s,
            window_end=e
        )
        
        # 数据增强
        if self.transform:
            data = self.transform(data)
        
        # 缓存结果
        if self.use_cache:
            self.cache[original_idx] = data
        
        return data

    def close(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
    
    def __del__(self):
        self.close()

# 图数据增强变换
class GraphNoiseTransform:
    """为节点特征添加噪声"""
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std
    
    def __call__(self, data):
        if self.noise_std > 0:
            noise = torch.randn_like(data.x) * self.noise_std
            data.x = data.x + noise
        return data

class EdgeDropTransform:
    """随机丢弃边"""
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob
    
    def __call__(self, data):
        if self.drop_prob > 0 and data.edge_index.size(1) > 0:
            keep_mask = torch.rand(data.edge_index.size(1)) > self.drop_prob
            data.edge_index = data.edge_index[:, keep_mask]
        return data