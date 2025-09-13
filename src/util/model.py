import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self, input_size, embed_size, num_layers, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, embed_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.classifier1 = nn.Sequential(nn.Linear(hidden_size, output_size),
                                         nn.Softmax(dim=1))
        self.classifier2 = nn.Sequential(nn.Linear(hidden_size, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, 32),
                                         nn.ReLU(),
                                         nn.Linear(32, output_size),
                                         nn.Softmax(dim=1))
        self.init_weight()

    def init_weight(self):
        print("init weight...")

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1)

        nn.init.constant_(self.ln.weight, 1)
        nn.init.constant_(self.ln.bias, 0)

        for idx, module in enumerate(self.classifier1):
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(0, 1)
                module.bias.data.fill_(0)
        for idx, module in enumerate(self.classifier2):
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(0, 1)
                module.bias.data.fill_(0)

    def forward(self, x):
        # x: [batch_size, sequence_length, input_size]
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # (num_layers, batch, hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        # x = self.embedding(x)
        out, _ = self.lstm(x, (h0, c0))  # out 的形状为 (batch, seq, hidden_size)
        out = self.ln(out)  # 归一化

        # 使用最后一个时间步的输出进行分类
        out = out[:, -1, :]  # 取最后一个时间步的输出，形状为 (batch, hidden_size)

        # # 池化
        # out = torch.mean(out, dim=1)
        # out = torch.squeeze(out)

        out = self.classifier1(out)  # 全连接层，输出形状为 (batch, output_size)
        return out


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        dk = k.size(2)

        att_score = torch.bmm(q, k.transpose(1, 2))
        att_score = F.softmax(att_score / torch.sqrt_(dk), dim=-1)

        out = torch.bmm(att_score, v)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1440):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term_odd = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term_even = torch.exp(torch.arange(1, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term_odd)
        pe[:, 1::2] = torch.cos(position * div_term_even)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状为 (sequence_length, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_size, seq_len, d_model, nhead, num_layers, output_size, dim_feedforward=2048, dropout=0.1, batch_first=True, mode=1):
        super(TransformerModel, self).__init__()
        self.embedding1 = nn.Linear(input_size, d_model)
        self.embedding2 = nn.Linear(seq_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first),
            num_layers)
        self.fc1 = nn.Linear(d_model, output_size)
        self.fc2 = nn.Linear(input_size*d_model, output_size)
        self.d_model = d_model

        self.mode = mode  # 1: 采用方式2；其他：采用方式1

    def forward(self, x):
        # print("x.shape: ", x.shape)
        if self.mode != 1:
            # 1
            # x 的形状为 (batch_size, sequence_length, input_size)
            x = x.permute(1, 0, 2)  # 转换为 (sequence_length, batch_size, input_size)
            x = self.embedding1(x)  # 转换为 (sequence_length, batch_size, d_model)
            x = self.positional_encoding(x)  # 添加位置编码
            x = self.transformer_encoder(x)  # Transformer 编码器
            x = x.permute(1, 0, 2)  # 转换为 (batch_size, sequence_length, d_model)
            # x = x[:, -1, :]  # 取最后一个时间步的输出
            # 池化
            x = torch.mean(x, dim=1)
            x = torch.squeeze(x)

            x = self.fc1(x)  # 全连接层
        else:
            # 2
            # x 的形状为 (batch_size, sequence_length, input_size)
            x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_size, sequence_length)
            x = self.embedding2(x)  # 转换为 ((batch_size, input_size, d_model)
            # x = self.positional_encoding(x)  # 添加位置编码
            x = self.transformer_encoder(x)  # Transformer 编码器
            # x = x.permute(0, 2, 1)  # 转换为 (batch_size, d_model, input_size)
            # x = x[:, -1, :]  # 取最后一个时间步的输出
            # 特征拼接
            x = torch.reshape(x, (x.size(0), -1, 1))
            x = torch.squeeze(x)

            x = self.fc2(x)  # 全连接层

        out = F.softmax(x, dim=-1)

        return out





