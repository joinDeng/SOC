#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的空间目标数据集类
"""
import h5py
import torch
import json
import numpy as np
from torch.utils.data import Dataset

class SpaceDataset(Dataset):
    def __init__(self, h5_path, window_size=None, transform=None):
        """
        Args:
            h5_path: HDF5文件路径
            window_size: 可选，如果指定则对长序列进行窗口分割
            transform: 数据增强变换
        """
        self.h5_path = h5_path
        self.window_size = window_size
        self.transform = transform
        
        # 延迟打开文件，避免多进程问题
        self._file = None
        self._keys = None
        
    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        return self._file
    
    @property
    def keys(self):
        if self._keys is None:
            self._keys = list(self.file.keys())
        return self._keys
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        grp = self.file[key]
        
        # 读取数据
        t = torch.tensor(grp['t'][:], dtype=torch.float32)
        pos = torch.tensor(grp['pos'][:], dtype=torch.float32)
        vel = torch.tensor(grp['vel'][:], dtype=torch.float32)
        ncf = torch.tensor(grp['ncf'][:], dtype=torch.float32)
        
        # 解析网格信息
        grid_info = json.loads(grp.attrs['grid'])
        
        # 窗口分割（如果需要）
        if self.window_size and len(t) > self.window_size:
            start_idx = torch.randint(0, len(t) - self.window_size, (1,)).item()
            t = t[start_idx:start_idx + self.window_size]
            pos = pos[start_idx:start_idx + self.window_size]
            vel = vel[start_idx:start_idx + self.window_size]
            ncf = ncf[start_idx:start_idx + self.window_size]
        
        # 数据增强
        if self.transform:
            t, pos, vel, ncf = self.transform(t, pos, vel, ncf)
        
        # 创建样本字典
        sample = {
            't': t,
            'pos': pos,
            'vel': vel, 
            'ncf': ncf,
            'grid': grid_info,
            'length': len(t),
            'key': key
        }
        
        return sample
    
    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def __del__(self):
        self.close()


def collate_fn(batch):
    """
    改进的collate函数，正确处理批次数据
    """
    # 按序列长度排序（用于pack_padded_sequence）
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)

    # 提取各种数据
    sequences = []
    lengths = []
    grid_infos = []

    for sample in batch:
        # 合并特征 [T, 12]: pos(3) + vel(3) + ncf(3) + 时间特征(3)
        pos = sample['pos']
        vel = sample['vel']
        ncf = sample['ncf']
        t = sample['t']

        # 计算时间特征
        dt = (t - t[0]) / 86400.0  # 转换为天
        day_angle = 2 * np.pi * dt / 365.0
        day_sin = torch.sin(day_angle)
        day_cos = torch.cos(day_angle)

        # 合并所有特征
        features = torch.cat([pos, vel, ncf,
                             dt.unsqueeze(-1),
                             day_sin.unsqueeze(-1),
                             day_cos.unsqueeze(-1)], dim=-1)

        sequences.append(features)
        lengths.append(sample['length'])
        grid_infos.append(sample['grid'])

    # 填充序列
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0.0
    )

    # 轨道类型编码
    orbit_map = {'LEO': 0, 'MEO': 1, 'HEO': 2}
    orbit_types = torch.tensor([
        orbit_map[info['orbit_class']] for info in grid_infos
    ], dtype=torch.long)

    # 类别标签编码
    cat_map = {'payload': 0, 'rocket body': 1, 'debris': 2}
    labels = torch.tensor([
        cat_map[info['final_cat']] for info in grid_infos
    ], dtype=torch.long)

    # RCS大小编码
    rcs_map = {'small': 0, 'medium': 1, 'large': 2}
    rcs_sizes = torch.tensor([
        rcs_map.get(info['final_rcs'], 1) for info in grid_infos  # 默认medium
    ], dtype=torch.long)

    return {
        'sequences': padded_sequences,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'orbit_labels': orbit_types,
        'cat_labels': labels,
        'rcs_labels': rcs_sizes,
        'grid_infos': grid_infos
    }


# def collate_fn(batch):
#     """处理变长序列的collate函数"""
#     features = [item['features'] for item in batch]
#     sequence_lengths = [item['sequence_length'] for item in batch]
#     max_len = max(sequence_lengths)
    
#     # 填充特征序列
#     padded_features = []
#     for feat, seq_len in zip(features, sequence_lengths):
#         if len(feat) < max_len:
#             pad_size = max_len - len(feat)
#             padded = torch.cat([feat, torch.zeros(pad_size, feat.shape[1])])
#         else:
#             padded = feat[:max_len]
#         padded_features.append(padded)
    
#     # 堆叠所有张量
#     batch_features = torch.stack(padded_features)  # [B, T, F]
#     batch_t = torch.stack([item['t'][:max_len] if len(item['t']) >= max_len 
#                           else torch.cat([item['t'], torch.zeros(max_len - len(item['t']))])
#                           for item in batch])
    
#     # 标签
#     orbit_labels = torch.tensor([item['orbit_label'] for item in batch], dtype=torch.long)
#     cat_labels = torch.tensor([item['cat_label'] for item in batch], dtype=torch.long)
#     rcs_labels = torch.tensor([item['rcs_label'] for item in batch], dtype=torch.long)
#     sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long)
    
#     # 创建注意力mask
#     attention_mask = torch.arange(max_len).unsqueeze(0) < sequence_lengths.unsqueeze(1)
    
#     return {
#         'features': batch_features,
#         't': batch_t,
#         'orbit_labels': orbit_labels,
#         'cat_labels': cat_labels,
#         'rcs_labels': rcs_labels,
#         'sequence_lengths': sequence_lengths,
#         'attention_mask': attention_mask
#     }


# 数据增强变换
class RandomNoise:
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std
    
    def __call__(self, t, pos, vel, ncf):
        pos = pos + torch.randn_like(pos) * self.noise_std
        vel = vel + torch.randn_like(vel) * self.noise_std
        ncf = ncf + torch.randn_like(ncf) * self.noise_std
        return t, pos, vel, ncf

class TimeMask:
    def __init__(self, mask_ratio=0.1):
        self.mask_ratio = mask_ratio
    
    def __call__(self, t, pos, vel, ncf):
        seq_len = len(t)
        mask_len = int(seq_len * self.mask_ratio)
        if mask_len > 0:
            start_idx = torch.randint(0, seq_len - mask_len, (1,)).item()
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[start_idx:start_idx + mask_len] = True
            pos[mask] = 0
            vel[mask] = 0
            ncf[mask] = 0
        return t, pos, vel, ncf
