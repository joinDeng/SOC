import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from improved_gat import RoPETransformerEncoder, SinCosPETimeTransEncoder

class SimpleSpectrumEncoder(nn.Module):
    """简化的频谱特征提取器 - 专注于主要频率成分"""
    def __init__(self, n_fft=64, hop_length=16, topk_freq=6, spec_dim=8):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.topk_freq = topk_freq
        self.spec_dim = spec_dim
        
        # 简单的频谱特征压缩网络
        self.spec_encoder = nn.Sequential(
            nn.Linear(topk_freq, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, spec_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, channels]
        返回: 压缩后的频谱特征 [batch_size, spec_dim]
        """
        batch_size, seq_len, channels = x.shape
        
        # 转置以便处理
        x_permuted = x.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        
        spec_features = []
        
        for i in range(batch_size):
            channel_specs = []
            
            # 对每个通道计算频谱
            for ch in range(channels):
                signal = x_permuted[i, ch]  # [seq_len]
                
                # 计算STFT
                stft_result = torch.stft(
                    signal, n_fft=self.n_fft, hop_length=self.hop_length,
                    window=torch.hann_window(self.n_fft).to(x.device),
                    return_complex=True, center=False
                )  # [freq_bins, time_frames]
                
                # 幅度谱
                magnitude = torch.abs(stft_result)  # [freq_bins, time_frames]
                
                # 时间维度平均，得到频率能量分布
                freq_energy = magnitude.mean(dim=1)  # [freq_bins]
                
                # 选择能量最高的频率
                top_energies, _ = torch.topk(
                    freq_energy, k=min(self.topk_freq, len(freq_energy))
                )
                channel_specs.append(top_energies)
            
            # 跨通道平均
            combined_spec = torch.stack(channel_specs).mean(dim=0)  # [topk_freq]
            
            # 压缩特征
            compressed_spec = self.spec_encoder(combined_spec.unsqueeze(0))  # [1, spec_dim]
            spec_features.append(compressed_spec)
        
        return torch.cat(spec_features, dim=0)  # [batch_size, spec_dim]


class EfficientSpectrumEncoder(nn.Module):
    """高效频谱编码器 - 计算一次，多处使用"""
    def __init__(self, n_fft=64, hop_length=9, topk_freq=8, spec_dim=8, 
                 use_channel_attention=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.topk_freq = topk_freq
        self.spec_dim = spec_dim
        self.use_channel_attention = use_channel_attention
        
        # 通道注意力机制
        if use_channel_attention:
            self.channel_attention = nn.Sequential(
                nn.Linear(topk_freq, 32),
                nn.ReLU(),
                nn.Linear(32, hop_length),
                nn.Softmax(dim=-1)
            )
        
        # 频谱特征压缩网络
        self.spec_compressor = nn.Sequential(
            nn.Linear(topk_freq, 32),
            nn.LayerNorm(32), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, spec_dim),
            nn.LayerNorm(spec_dim),
            nn.Tanh()  # 限制输出范围
        )
        
        # 频谱相似性投影
        self.similarity_proj = nn.Sequential(
            nn.Linear(spec_dim, 32),
            nn.GELU(),
            nn.Linear(32, spec_dim)
        )
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, 12] 或 [num_nodes, seq_len, 12]
        返回: 频谱特征和相似性特征
        """
        batch_size, seq_len, num_channels = x.shape
        
        # 转置以便处理通道维度
        x_permuted = x.permute(0, 2, 1)  # [batch_size, 12, seq_len]
        
        spec_features = []
        dominant_freqs = []
        
        for i in range(batch_size):
            channel_specs = []
            channel_dom_freqs = []
            
            # for ch in range(num_channels):
            # 提取 pos, vel, 和ncf 的频谱特征
            for ch in range(9):  
                # 计算单通道STFT
                signal = x_permuted[i, ch]  # [seq_len]
                
                # 使用torch.stft
                stft_result = torch.stft(
                    signal, n_fft=self.n_fft, hop_length=self.hop_length,
                    window=torch.hann_window(self.n_fft).to(x.device),
                    return_complex=True
                )  # [freq_bins, time_frames]
                
                # 计算幅度谱
                magnitude = torch.abs(stft_result)  # [freq_bins, time_frames]
                
                # 时间维度平均，得到频率分布
                freq_energy = magnitude.mean(dim=1)  # [freq_bins]
                
                # 选择能量最高的topk频率
                top_energies, top_indices = torch.topk(
                    freq_energy, k=min(self.topk_freq, len(freq_energy))
                )
                
                channel_specs.append(top_energies)
                channel_dom_freqs.append(top_indices.float() / self.n_fft)  # 归一化频率
            
            # 通道聚合
            if self.use_channel_attention:
                # 使用注意力权重聚合通道
                all_specs = torch.stack(channel_specs)  # [9, topk_freq]
                attention_weights = self.channel_attention(all_specs)  # [9, 9] - 自注意力
                weighted_specs = torch.matmul(attention_weights, all_specs)  # [9, topk_freq]
                combined_spec = weighted_specs.mean(dim=0)  # [topk_freq]
            else:
                # 简单平均
                combined_spec = torch.stack(channel_specs).mean(dim=0)  # [topk_freq]
            
            # 压缩频谱特征
            compressed_spec = self.spec_compressor(combined_spec.unsqueeze(0))  # [1, spec_dim]
            spec_features.append(compressed_spec)
            
            #  dominant frequencies for similarity
            dom_freq = torch.stack(channel_dom_freqs).mean(dim=0)  # [topk_freq]
            dominant_freqs.append(dom_freq)
        
        spec_features = torch.cat(spec_features, dim=0)  # [batch_size, spec_dim]
        dominant_freqs = torch.stack(dominant_freqs)  # [batch_size, topk_freq]
        
        # 计算相似性特征
        similarity_features = self.similarity_proj(spec_features)  # [batch_size, 8]
        
        return {
            'spec_features': spec_features,  # 用于节点特征拼接
            'similarity_features': similarity_features,  # 用于注意力偏置
            'dominant_freqs': dominant_freqs  # 原始频率信息（可选）
        }


class SpectrumEnhancedNodeEncoder(nn.Module):
    """增强的节点特征编码器"""
    def __init__(self, seq_len=64, input_dim=12, temporal_dim=64, spec_dim=8, 
                 use_spec_features=True, fusion_method='concat', d_model=64, dropout=0.1):
        super().__init__()
        self.use_spec_features = use_spec_features
        self.fusion_method = fusion_method
        
        # 时序Transformer编码器
        # self.temporal_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=input_dim,
        #         nhead=4,
        #         dim_feedforward=128,
        #         dropout=0.1,
        #         batch_first=True
        #     ),
        #     num_layers=2
        # )
        self.temporal_encoder = SinCosPETimeTransEncoder(
            input_dim=input_dim, 
            d_model=d_model, 
            nhead=4, 
            nlayers=2,
            dropout=dropout
        )
        # self.temporal_encoder = RoPETransformerEncoder(
        #     input_dim=input_dim, 
        #     d_model=64, 
        #     nhead=4, 
        #     nlayers=2,
        #     dropout=dropout
        # )
        
        # 时序特征投影
        self.temporal_proj = nn.Sequential(
            nn.Linear(d_model, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 频谱编码器
        if use_spec_features:
            self.spec_encoder = EfficientSpectrumEncoder(spec_dim=spec_dim)
            
            # 特征融合
            if fusion_method == 'concat':
                self.output_dim = temporal_dim + spec_dim
            elif fusion_method == 'weighted':
                self.output_dim = temporal_dim
                self.fusion_weights = nn.Parameter(torch.ones(2))
                self.fusion_gate = nn.Sequential(
                    nn.Linear(temporal_dim + spec_dim, temporal_dim),
                    nn.Sigmoid()
                )
        
    def forward(self, x, return_components=False):
        """
        x: [batch_size, seq_len, input_dim]
        """
        # 时序特征
        temporal_features = self.temporal_encoder(x)  # [batch_size, seq_len, input_dim] -> [batch_size, d_model] 
        temporal_features = self.temporal_proj(temporal_features)  # [batch_size, temporal_dim]
        # print(f"temporal_features: {temporal_features.shape}")
        if not self.use_spec_features:
            return temporal_features
        
        # 频谱特征
        spec_outputs = self.spec_encoder(x)
        spec_features = spec_outputs['spec_features']  # [batch_size, spec_dim]
        freq_features = spec_outputs['dominant_freqs']  # [batch_size, spec_dim]
        similarity_features = spec_outputs['similarity_features']  # [batch_size, 8]
        # print(f"spec_features: {spec_features.shape}")
        # print(f"freq_features: {freq_features.shape}")
        
        
        # 特征融合
        if self.fusion_method == 'concat':
            node_features = torch.cat([temporal_features, spec_features, freq_features], dim=-1)
        elif self.fusion_method == 'weighted':
            # 门控融合
            combined = torch.cat([temporal_features, spec_features, freq_features], dim=-1)
            gate = self.fusion_gate(combined)
            node_features = gate * temporal_features + (1 - gate) * self.temporal_proj(spec_features)
        
        if return_components:
            return {
                'node_features': node_features,
                'temporal_features': temporal_features,
                'spec_features': spec_features,
                'dominant_freqs': freq_features,
                'similarity_features': similarity_features
            }
        
        return node_features


from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

class StandardGATMultiTask(nn.Module):
    """标准的GAT多任务模型 - 只修改节点特征，不修改注意力机制"""
    def __init__(self, node_dim=80, hidden_dim=128, n_cat=3, n_rcs=4, n_orbit=3, 
                 heads=8, num_layers=3, dropout=0.1, use_residual=True):  # node_dim = node_features 
        super().__init__()
        self.use_residual = use_residual
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        current_dim = node_dim
        
        for i in range(num_layers):
            # 确定当前层的头数和输出维度
            layer_heads = heads if i < num_layers - 1 else 1  # 最后一层单头
            layer_out_dim = hidden_dim
            
            gat_layer = GATConv(
                current_dim, 
                layer_out_dim // layer_heads,
                heads=layer_heads,
                dropout=dropout,
                concat=True
            )
            self.gat_layers.append(gat_layer)
            
            # 残差连接
            if use_residual and current_dim != layer_out_dim:
                self.add_module(f'residual_{i}', nn.Linear(current_dim, layer_out_dim))
            
            current_dim = layer_out_dim
        
        # 图池化 - 结合均值和最大值池化
        self.pooling = nn.Sequential(
            nn.Linear(current_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 多任务分类头
        self.classifiers = nn.ModuleDict({
            'cat': self._build_classifier(hidden_dim, n_cat, dropout),
            'rcs': self._build_classifier(hidden_dim, n_rcs, dropout),
            'orbit': self._build_classifier(hidden_dim, n_orbit, dropout)
        })
        
    def _build_classifier(self, hidden_dim, num_classes, dropout):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 通过GAT层
        for i, gat_layer in enumerate(self.gat_layers):
            # 残差连接
            if self.use_residual and i > 0:
                residual = x
                if hasattr(self, f'residual_{i}'):
                    residual = getattr(self, f'residual_{i}')(residual)
            
            # GAT前向传播
            x_new = gat_layer(x, edge_index)
            
            # 应用残差连接和激活
            if self.use_residual and i > 0:
                x = F.gelu(x_new + residual)
            else:
                x = F.gelu(x_new)
        
        # 图级池化
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_features = torch.cat([mean_pool, max_pool], dim=1)
        graph_features = self.pooling(graph_features)
        
        # 多任务输出
        outputs = {}
        for task_name, classifier in self.classifiers.items():
            outputs[task_name] = classifier(graph_features)
        
        return outputs

class SpectrumEnhancedGAT(nn.Module):
    """完整的频谱增强GAT模型"""
    def __init__(self, seq_len=64, input_dim=12, temporal_dim=64, spec_dim=8,
                 hidden_dim=128, n_cat=3, n_rcs=4, n_orbit=3, 
                 fusion_method='concat', gat_heads=8, gat_layers=3, dropout=0.1):
        super().__init__()
        
        # 节点特征编码器（时序 + 频谱）
        self.node_encoder = SpectrumEnhancedNodeEncoder(
            seq_len=seq_len,
            input_dim=input_dim,
            temporal_dim=temporal_dim,
            spec_dim=spec_dim,
            fusion_method=fusion_method,
            dropout=dropout
        )
        
        # 确定GAT输入维度
        if fusion_method == 'concat':
            gat_input_dim = temporal_dim + spec_dim*2
        else:
            gat_input_dim = temporal_dim
        
        # GAT网络
        self.gat_network = StandardGATMultiTask(
            node_dim=gat_input_dim,
            hidden_dim=hidden_dim,
            n_cat=n_cat,
            n_rcs=n_rcs,
            n_orbit=n_orbit,
            heads=gat_heads,
            num_layers=gat_layers,
            dropout=dropout
        )
        
        print(f"模型配置:")
        print(f"  - 节点特征维度: {gat_input_dim} (时序: {temporal_dim} + 频谱: {spec_dim})")
        print(f"  - GAT隐藏层: {hidden_dim}, 头数: {gat_heads}, 层数: {gat_layers}")
        print(f"  - 融合方法: {fusion_method}")
        
    def forward(self, data):
        # 编码节点特征
        node_features = self.node_encoder(data.x)  # [num_nodes, node_dim]
        
        # 创建新的数据对象（保持其他属性不变）
        from torch_geometric.data import Data
        new_data = Data(
            x=node_features,
            edge_index=data.edge_index,
            batch=data.batch,
            y_cat=data.y_cat,
            y_rcs=data.y_rcs,
            y_orbit=data.y_orbit
        )
        
        # 通过GAT网络
        return self.gat_network(new_data)
