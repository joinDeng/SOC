import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import math

import traceback


class PositionalEncoding(nn.Module):
    """Transformer位置编码"""
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SinCosPETimeTransEncoder(nn.Module):
    """改进的时序Transformer编码器"""
    def __init__(self, input_dim=16, d_model=128, nhead=8, nlayers=3, 
                 dropout=0.1, max_seq_len=1440):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        # 层归一化和dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 多头注意力池化
        self.attention_pool = nn.MultiheadAttention(d_model, 1, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim] 或 [num_nodes, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        
        # 输入投影
        scale = min(1.0, math.sqrt(1.0 / input_dim)) 
        x = self.input_proj(x) * scale  # [batch_size, seq_len, d_model]
        
        # 位置编码
        x = x.permute(1, 0, 2) 
        x = self.pos_encoding(x)
        x = x.permute(1, 0, 2) 
        
        # Transformer编码
        x = self.layer_norm(x)
        x = self.transformer(x)  # [batch_size, seq_len, d_model]
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # 注意力池化
        query = torch.mean(x, dim=1, keepdim=True)  # 全局查询 [batch_size, 1, d_model]
        attended, attn_weights = self.attention_pool(query, x, x)  # [batch_size, 1, d_model]
        
        return attended.squeeze(1)  # [batch_size, d_model]

class SinCosPEResidualGATBlock(nn.Module):
    """残差GAT块"""
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.gat_conv = GATConv(
            in_channels, out_channels, heads=heads, 
            dropout=dropout, concat=True
        )
        self.layer_norm = nn.LayerNorm(out_channels * heads)
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        if in_channels != out_channels * heads:
            self.residual = nn.Linear(in_channels, out_channels * heads)
        else:
            self.residual = nn.Identity()

    def forward(self, x, edge_index):
        residual = self.residual(x)
        x = self.gat_conv(x, edge_index)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual

class ImprovedGATMultiTaskSinCosPE(nn.Module):
    """改进的多任务GAT模型"""
    def __init__(self, input_dim=16, node_dim=128, hidden_dim=128, 
                 n_cat=3, n_rcs=4, n_orbit=3, n_heads=8, 
                 num_gat_layers=2, dropout=0.1, use_attention_pool=True):
        super().__init__()
        
        self.use_attention_pool = use_attention_pool
        
        # 节点特征编码器
        self.node_encoder = SinCosPETimeTransEncoder(
            input_dim=input_dim, 
            d_model=node_dim, 
            nhead=n_heads, 
            nlayers=2,
            dropout=dropout
        )
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        current_dim = node_dim
        
        for i in range(num_gat_layers):
            heads = n_heads if i < num_gat_layers - 1 else 1  # 最后一层单头
            gat_layer = SinCosPEResidualGATBlock(
                current_dim, hidden_dim, heads=heads, dropout=dropout
            )
            self.gat_layers.append(gat_layer)
            current_dim = hidden_dim * heads
        
        # 注意力池化层
        if use_attention_pool:
            self.graph_attention = nn.MultiheadAttention(
                current_dim, num_heads=4, dropout=dropout, batch_first=True
            )
        
        # 分类头
        self.classifier_heads = nn.ModuleDict({
            'cat': self._create_classifier(current_dim//2, n_cat, dropout),
            'rcs': self._create_classifier(current_dim//2, n_rcs, dropout),
            'orbit': self._create_classifier(current_dim//2, n_orbit, dropout)
        })
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 初始化权重
        self.apply(self._init_weights)

    def _create_classifier(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, out_dim)
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            with torch.no_grad():
                module.weight.clamp_(-0.1, 0.1)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 重塑节点特征: [batch_size * num_nodes, seq_len, input_dim] -> [num_nodes, seq_len, input_dim]
        original_shape = x.shape
        if len(original_shape) == 4:  # [batch_size, num_nodes, seq_len, input_dim]
            batch_size, num_nodes, seq_len, input_dim = original_shape
            x = x.view(batch_size * num_nodes, seq_len, input_dim)
        else:  # [num_nodes, seq_len, input_dim]
            num_nodes, seq_len, input_dim = original_shape
        
        # 1. 节点特征编码
        node_features = self.node_encoder(x)  # [num_nodes, node_dim]
        
        # 2. 图注意力传播
        for gat_layer in self.gat_layers:
            node_features = gat_layer(node_features, edge_index)

        # 3. 图级表示
        if self.use_attention_pool:
            # 使用注意力池化
            graph_rep = self._attention_pooling(node_features, batch)
        else:
            # 使用均值+最大值池化
            mean_pool = global_mean_pool(node_features, batch)
            max_pool = global_max_pool(node_features, batch)
            graph_rep = torch.cat([mean_pool, max_pool], dim=1)
            graph_rep = self.output_proj(graph_rep)
        
        # 4. 多任务输出
        outputs = {}
        for task_name, classifier in self.classifier_heads.items():
            outputs[task_name] = classifier(graph_rep)
        
        return outputs

    def _attention_pooling(self, node_features, batch):
        """注意力池化获取图级表示"""
        batch_size = batch.max().item() + 1
        
        # 为每个图创建查询
        graph_representations = []
        for i in range(batch_size):
            mask = (batch == i)
            graph_nodes = node_features[mask]  # [num_nodes_in_graph, features]
            
            if len(graph_nodes) > 0:
                # 添加图级查询
                graph_query = graph_nodes.mean(dim=0, keepdim=True)  # [1, features]
                
                # 注意力池化
                attended, _ = self.graph_attention(
                    graph_query.unsqueeze(0),  # [1, 1, features]
                    graph_nodes.unsqueeze(0),  # [1, num_nodes, features]
                    graph_nodes.unsqueeze(0)   # [1, num_nodes, features]
                )
                graph_rep = attended.squeeze(0)  # [1, features]
            else:
                graph_rep = torch.zeros(1, node_features.size(1), device=node_features.device)
            
            graph_representations.append(graph_rep)
        
        graph_rep = torch.cat(graph_representations, dim=0)  # [batch_size, features]
        graph_rep = self.output_proj(graph_rep)
        
        return graph_rep


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import traceback

class RotaryPositionalEmbedding(nn.Module):
    """改进的RoPE实现"""
    def __init__(self, dim, max_seq_len=1440, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算θ
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)   # [dim/2]

    def forward(self, x, seq_len=None):
        """
        x: [batch, num_heads, seq_len, head_dim]
        return: 旋转后的Q或K
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # 位置矩阵 [seq_len, dim/2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)  # [seq_len]
        freqs = torch.outer(t, self.inv_freq)                             # [seq_len, dim/2]
        cos, sin = freqs.cos(), freqs.sin()                               # [seq_len, dim/2]
        
        # 扩展维度以匹配输入
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
        
        # 拆分成实部和虚部
        assert head_dim%2 == 0
        x1, x2 = x.chunk(2, dim=-1)  # 各为 [batch, heads, seq_len, dim/2]
        
        # 应用旋转
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos
        
        # 重新组合
        x_rot = torch.cat([x_rot1, x_rot2], dim=-1)
        return x_rot

class RoPEMultiHeadAttention(nn.Module):
    """使用RoPE的多头注意力"""
    def __init__(self, embed_dim=64, num_heads=8, max_len=1440, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=max_len)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 投影到Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用RoPE到Q和K
        q = self.rope(q)
        k = self.rope(k)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重和输出
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)

class RoPETransformerEncoderLayer(nn.Module):
    """使用RoPE的Transformer编码层"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, max_len=1440):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, nhead, max_len=max_len, dropout=dropout)
        
        # Feed-forward网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = F.gelu

    def forward(self, src, src_mask=None):
        # 自注意力 + 残差
        src2 = self.self_attn(self.norm1(src), src_mask)
        src = src + self.dropout1(src2)
        
        # Feed-forward + 残差
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        
        return src

class RoPETransformerEncoder(nn.Module):
    """完整的RoPE Transformer编码器"""
    def __init__(self, input_dim=16, d_model=64, nhead=8, num_layers=2, dim_feedforward=2048, 
                 dropout=0.1, max_len=1440, pool_method='mean'):
        super().__init__()
        self.d_model = d_model
        self.pool_method = pool_method
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码（RoPE已经在注意力中实现，这里不需要额外的位置编码）
        
        # Transformer层
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, max_len)
            for _ in range(num_layers)
        ])
        
        # 输出归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 可选的池化方法
        if pool_method == 'attention':
            self.pooling_attention = nn.MultiheadAttention(d_model, 1, dropout=dropout, batch_first=True)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)  # [batch_size, seq_len, d_model]
        
        # 序列池化
        if self.pool_method == 'mean':
            # 全局平均池化
            x = x.mean(dim=1)  # [batch_size, d_model]
        elif self.pool_method == 'max':
            # 全局最大池化
            x = x.max(dim=1)[0]  # [batch_size, d_model]
        elif self.pool_method == 'attention':
            # 注意力池化
            query = x.mean(dim=1, keepdim=True)  # [batch_size, 1, d_model]
            attended, _ = self.pooling_attention(query, x, x)
            x = attended.squeeze(1)  # [batch_size, d_model]
        elif self.pool_method == 'first':
            # 取第一个token
            x = x[:, 0]  # [batch_size, d_model]
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")
        
        return x


class RoPEResidualGATBlock(nn.Module):
    """残差GAT块 - 修复版本"""
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.gat_conv = GATConv(
            in_channels, out_channels, heads=heads, 
            dropout=dropout, concat=True
        )
        self.layer_norm = nn.LayerNorm(out_channels * heads)
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        if in_channels != out_channels * heads:
            self.residual = nn.Linear(in_channels, out_channels * heads)
        else:
            self.residual = nn.Identity()

    def forward(self, x, edge_index):
        residual = self.residual(x)
        x = self.gat_conv(x, edge_index)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual

class ImprovedGATMultiTaskRoPE(nn.Module):
    """修复的多任务GAT模型"""
    def __init__(self, input_dim=16, node_dim=128, hidden_dim=128, 
                 n_cat=3, n_rcs=3, n_orbit=3, n_heads=8, 
                 num_gat_layers=2, dropout=0.1, use_attention_pool=True,
                 transformer_layers=2, max_seq_len=1440):
        super().__init__()
        
        self.use_attention_pool = use_attention_pool
        
        # 节点特征编码器 - 使用RoPE Transformer
        self.node_encoder = RoPETransformerEncoder(
            input_dim=input_dim,
            d_model=node_dim,
            nhead=n_heads,
            num_layers=transformer_layers,
            dim_feedforward=node_dim * 4,
            dropout=dropout,
            max_len=max_seq_len,
            pool_method='mean'  # 使用平均池化得到每个节点的特征
        )
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        current_dim = node_dim
        
        for i in range(num_gat_layers):
            heads = n_heads if i < num_gat_layers - 1 else 1  # 最后一层单头
            gat_layer = RoPEResidualGATBlock(
                current_dim, hidden_dim, heads=heads, dropout=dropout
            )
            self.gat_layers.append(gat_layer)
            current_dim = hidden_dim * heads
        
        # 注意力池化层
        if use_attention_pool:
            self.graph_attention = nn.MultiheadAttention(
                current_dim, num_heads=4, dropout=dropout, batch_first=True
            )
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 分类头 - 修复维度问题
        final_dim = hidden_dim // 2
        self.classifier_heads = nn.ModuleDict({
            'cat': self._create_classifier(final_dim, n_cat, dropout),
            'rcs': self._create_classifier(final_dim, n_rcs, dropout),
            'orbit': self._create_classifier(final_dim, n_orbit, dropout)
        })
        
        # 初始化权重
        self.apply(self._init_weights)

    def _create_classifier(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, out_dim)
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 检查输入维度并重塑
        original_shape = x.shape
        
        if len(original_shape) == 3:
            # [num_nodes, seq_len, input_dim] - 这是正确格式
            num_nodes, seq_len, input_dim = original_shape
        elif len(original_shape) == 4:
            # [batch_size, num_nodes, seq_len, input_dim] - 需要重塑
            batch_size, num_nodes, seq_len, input_dim = original_shape
            x = x.view(batch_size * num_nodes, seq_len, input_dim)
        else:
            raise ValueError(f"Unexpected input shape: {original_shape}")
        
        # 1. 节点特征编码 - 输出应该是 [num_nodes, node_dim]
        node_features = self.node_encoder(x)
        
        # 确保节点特征是2D的 [num_nodes, features]
        if len(node_features.shape) != 2:
            raise ValueError(f"Node features should be 2D, got {node_features.shape}")
        
        # 2. 图注意力传播
        for gat_layer in self.gat_layers:
            node_features = gat_layer(node_features, edge_index)
        
        # 3. 图级表示
        if self.use_attention_pool:
            graph_rep = self._attention_pooling(node_features, batch)
        else:
            # 使用均值+最大值池化
            mean_pool = global_mean_pool(node_features, batch)
            max_pool = global_max_pool(node_features, batch)
            graph_rep = torch.cat([mean_pool, max_pool], dim=1)
            graph_rep = self.output_proj(graph_rep)
        
        # 4. 多任务输出
        outputs = {}
        for task_name, classifier in self.classifier_heads.items():
            outputs[task_name] = classifier(graph_rep)
        
        return outputs

    def _attention_pooling(self, node_features, batch):
        """注意力池化获取图级表示"""
        if batch is None:
            # 如果没有批处理信息，假设所有节点属于同一个图
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
        
        batch_size = batch.max().item() + 1
        
        # 为每个图创建查询
        graph_representations = []
        for i in range(batch_size):
            mask = (batch == i)
            graph_nodes = node_features[mask]  # [num_nodes_in_graph, features]
            
            if len(graph_nodes) > 0:
                # 添加图级查询
                graph_query = graph_nodes.mean(dim=0, keepdim=True)  # [1, features]
                
                # 注意力池化
                attended, _ = self.graph_attention(
                    graph_query.unsqueeze(0),  # [1, 1, features]
                    graph_nodes.unsqueeze(0),  # [1, num_nodes, features]
                    graph_nodes.unsqueeze(0)   # [1, num_nodes, features]
                )
                graph_rep = attended.squeeze(0)  # [1, features]
            else:
                graph_rep = torch.zeros(1, node_features.size(1), device=node_features.device)
            
            graph_representations.append(graph_rep)
        
        graph_rep = torch.cat(graph_representations, dim=0)  # [batch_size, features]
        graph_rep = self.output_proj(graph_rep)
        
        return graph_rep


def register_nan_hook(model):
    def nan_hook(self, inp, output):
        if not isinstance(output, tuple):
            output = (output,)
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor) and torch.isnan(o).any():
                print(f"❌ NaN in {self.__class__.__name__} output[{i}]")
                print(f"output:", output)
                traceback.print_stack()
                raise RuntimeError("NaN detected")
    for module in model.modules():
        module.register_forward_hook(nan_hook)