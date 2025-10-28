#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的双层LSTM空间目标分类模型
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
from collections import Counter
import torch.nn.functional as F

class MultiTaskLSTMClassifier(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, 
                 num_orbit_classes=3, num_cat_classes=3, num_rcs_classes=3,
                 dropout=0.2):
        super().__init__()
        
        # 计算LSTM输入维度
        lstm_input_dim = input_dim
        
        # LSTM层
        self.lstm = nn.LSTM(
            lstm_input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=False
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 多任务分类头
        # 目标类别分类 (payload, rocket body, debris)
        self.classifier_category = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, num_cat_classes)  # 3个类别
        )
        
        # RCS大小分类
        self.classifier_rcs = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, num_rcs_classes)  # 3个类别
        )
        
        # 轨道类型分类
        self.classifier_orbit = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, num_orbit_classes)  # 3个类别
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(lstm_input_dim)
        
    def forward(self, sequences, lengths):
        batch_size, seq_len, input_dim = sequences.shape
        
        # 层归一化
        sequences = self.layer_norm(sequences)
        
        # 打包序列以提高效率
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM前向传播
        packed_output, (hidden, _) = self.lstm(packed_sequences)
        
        # 解包输出
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        # 注意力机制
        attention_weights = torch.softmax(
            self.attention(output).squeeze(-1), dim=-1
        )
        
        # 加权求和
        context_vector = torch.sum(
            output * attention_weights.unsqueeze(-1), dim=1
        )
        
        # # 分类
        # logits = self.classifier(context_vector)
        
        # return logits, attention_weights

        # 多任务输出
        orbit_logits = self.classifier_orbit(context_vector)
        cat_logits = self.classifier_category(context_vector)
        rcs_logits = self.classifier_rcs(context_vector)
        
        return {
            'orbit': orbit_logits,
            'cat': cat_logits,
            'rcs': rcs_logits
        }


