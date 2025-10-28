#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的1D-CNN多尺度+残差空间目标分类模型
支持多任务学习：类别分类、RCS大小分类、轨道类型分类
"""
import os
import json
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from space_dataset import SpaceDataset
import torch.nn.functional as F
from collections import Counter

# ---------- 改进的模型组件 ----------
class ImprovedResBlock1D(nn.Module):
    """改进的残差块，支持下采样和通道变化"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, 
                              padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, 
                              dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # 跳跃连接
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class MultiScaleFeatureExtractor(nn.Module):
    """改进的多尺度特征提取器"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        # 确保通道数能被3整除
        assert out_ch % 4 == 0, "out_ch must be divisible by 4"
        branch_ch = out_ch // 4
        
        # 不同尺度的卷积核
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(inplace=True)
        )
        
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(inplace=True)
        )
        
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 7, padding=3, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(inplace=True)
        )

        self.branch9 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 9, padding=4, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        b9 = self.branch9(x)
        
        out = torch.cat([b3, b5, b7, b9], dim=1)
        out = self.fusion(out)
        return out

class AttentionPooling(nn.Module):
    """注意力池化替代全局平均池化"""
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_dim // 2, 1, 1),
            nn.Softmax(dim=2)
        )
        
    def forward(self, x):
        # x: [B, C, T]
        attn_weights = self.attention(x)  # [B, 1, T]
        out = torch.sum(x * attn_weights, dim=2)  # [B, C]
        return out, attn_weights

class MultiTaskCNNClassifier(nn.Module):
    """多任务CNN分类器"""
    def __init__(self, input_dim=12, base_ch=64, num_blocks=4, 
                 dropout=0.3, use_attention=True):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, base_ch, 7, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 多尺度特征提取
        self.multiscale = MultiScaleFeatureExtractor(base_ch, base_ch)
        
        # 残差块序列
        self.res_blocks = nn.ModuleList()
        current_ch = base_ch
        
        for i in range(num_blocks):
            # 每两个块进行一次下采样
            stride = 2 if i % 2 == 1 else 1
            next_ch = current_ch * 2 if stride == 2 else current_ch
            
            block = ImprovedResBlock1D(
                current_ch, next_ch, stride=stride, dropout=dropout
            )
            self.res_blocks.append(block)
            current_ch = next_ch
        
        # 池化层
        self.use_attention = use_attention
        if use_attention:
            self.pooling = AttentionPooling(current_ch)
        else:
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # 多任务分类头
        # 目标类别分类 (payload, rocket body, debris)
        self.classifier_category = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(current_ch, current_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(current_ch // 2, 3)  # 3个类别
        )
        
        # RCS大小分类
        self.classifier_rcs = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(current_ch, current_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(current_ch // 2, 3)  # small, medium, large
        )
        
        # 轨道类型分类
        self.classifier_orbit = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(current_ch, current_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(current_ch // 2, 3)  # LEO, MEO, GEO
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [B, F, T]
        # batch_size, seq_len, input_dim = x.shape
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.multiscale(x)
        
        # 通过残差块
        for block in self.res_blocks:
            x = block(x)
        
        # 池化
        if self.use_attention:
            features, attn_weights = self.pooling(x)
        else:
            features = self.pooling(x).squeeze(-1)
            attn_weights = None
        
        # 多任务输出
        output_category = self.classifier_category(features)
        output_rcs = self.classifier_rcs(features)
        output_orbit = self.classifier_orbit(features)
        
        return {
            'cat': output_category,
            'rcs': output_rcs,
            'orbit': output_orbit,
            'features': features,
            'attention': attn_weights
        }

# ---------- 改进的数据辅助函数 ----------
def improved_collate_fn(batch):
    """
    改进的collate函数，正确处理多任务标签
    """
    # 按序列长度排序
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    # 提取序列数据
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
        features = torch.cat([
            pos, vel, ncf, 
            dt.unsqueeze(-1), 
            day_sin.unsqueeze(-1), 
            day_cos.unsqueeze(-1)
        ], dim=-1)
        
        sequences.append(features.transpose(0, 1))  # [F, T] 用于1D卷积
        lengths.append(sample['length'])
        grid_infos.append(sample['grid'])
    
    # 填充序列 [B, F, T]
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0.0
    ).transpose(1, 2)  # 转置为 [B, F, T]
    
    # 创建掩码
    max_length = padded_sequences.size(2)
    mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    # 多任务标签编码
    cat_map = {'payload': 0, 'rocket body': 1, 'debris': 2}
    rcs_map = {'small': 0, 'medium': 1, 'large': 2}
    orbit_map = {'LEO': 0, 'MEO': 1, 'HEO': 2}
    
    cat_labels = torch.tensor([
        cat_map[info['final_cat']] for info in grid_infos
    ], dtype=torch.long)
    
    rcs_labels = torch.tensor([
        rcs_map.get(info['final_rcs'], 1) for info in grid_infos  # 默认medium
    ], dtype=torch.long)
    
    orbit_labels = torch.tensor([
        orbit_map[info['orbit_class']] for info in grid_infos
    ], dtype=torch.long)
    
    return {
        'sequences': padded_sequences,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'mask': mask,
        'cat_labels': cat_labels,
        'rcs_labels': rcs_labels,
        'orbit_labels': orbit_labels,
        'grid_infos': grid_infos
    }