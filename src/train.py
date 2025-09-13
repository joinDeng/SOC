import random
import swanlab
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from util.data import NCFDataset
from util.dtw_knn import KNN_DTW
from util.model import LSTM, TransformerModel


def show_sample(dataset_ts):
    color_tab = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig, axs = plt.subplots(4, 4)
    ncf_str = ['ncf_x', 'ncf_y', 'ncf_z']
    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            index = random.randint(0, len(dataset_ts) - 1)
            time = dataset_ts.sos_time[index]
            data = dataset_ts.sos_ncf[index]
            label = dataset_ts.sos_label[index]
            ax.set_title("sample {} Category:{}".format(4 * i + j + 1, label))
            for k in range(3):
                ax.plot(data[:, k], color=color_tab[label], label=ncf_str[k])

    for ax in axs.flat:
        ax.set_xlabel('time')
        ax.set_ylabel('ncf')
        ax.legend()

    fig.suptitle('NCF samples')
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_model_on_epoch(model, train_loader, criterion, optimizer, epoch, epochs, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, target.to(torch.long))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, batch_idx + 1,
                                                                          len(train_loader), loss.item()))
        if batch_idx % 10 == 9:
            swanlab.log({"train/loss": loss.item()})


def validate_model(model, test_loader, criterion, epoch, epochs, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target.to(torch.long))
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # 更新学习率并监测验证集上的性能
        scheduler.step(running_loss / len(test_loader))
        swanlab.log({"val/loss": running_loss / len(test_loader)})
        swanlab.log({"val/accuracy": correct / total})

    print(f"Epoch [{epoch + 1}/{epochs}], 测试准确率: {100 * correct / total:.2f}%")


if __name__ == '__main__':
    print("reading data...")
    folder_path = r"D:\研究生\项目\非保守力空间目标识别\数据集\NCF"  # # r"D:\app\program\Python\Pycharm\project\SOC\data"
    dataset = NCFDataset(folder_path, normalize=False, statistic=False, frequency=False)

    # NCF 样例
    # show_sample(dataset)

    # 参数设置
    seed = random.seed(0)
    run = swanlab.init(experiment_name="soc-lstm",
                       config={"epochs": 20,
                               "batch_size": 32,
                               "lr": 1e-2,
                               "input_size": 3,  # 输入时间序列长度/手动提取特征总数
                               "embedding_size": 64,
                               "num_layers": 1,
                               "hidden_size": 256})
    # run = swanlab.init(experiment_name="soc-attention",
    #                    config={"epochs": 10,
    #                            "batch_size": 64,
    #                            "lr": 1e-3,
    #                            "input_size": 3,
    #                            "embedding_size": 64,
    #                            "num_heads": 8,
    #                            "num_layers": 2,
    #                            "dim_feedforward": 128,
    #                            "dropout": 0.1})

    print("splitting data...")
    train_set, test_set = train_test_split(dataset, test_size=0.5, shuffle=True)

    # # baseline  knn+dtw
    # knn_dtw = KNN_DTW(k=1)
    # def split_data_labels(_dataset):
    #     data, labels = [], []
    #     for _, (_data, _label) in enumerate(_dataset):
    #         data.append(_data)
    #         labels.append(_label)
    #     return data, labels
    # train_data, train_labels = split_data_labels(train_set)
    # test_data, test_labels = split_data_labels(test_set)
    # labels_name = np.arange(9)
    # accuracy = knn_dtw.predict(train_data, train_labels, test_data, test_labels, labels_name)
    # print("DTW-KNN accuracy: ", accuracy)

    # exit(0)

    # 分批处理
    train_loader = DataLoader(train_set, batch_size=run.config["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

    # 检查是否有可用的 GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using device:", device)

    print("creating model...")
    # 定义模型、损失函数和优化器
    model = LSTM(input_size=run.config["input_size"],
                 embed_size=run.config["embedding_size"],
                 num_layers=run.config["num_layers"],
                 hidden_size=run.config["hidden_size"],
                 output_size=9)
    # model = TransformerModel(input_size=run.config["input_size"],
    #                          seq_len=1440,
    #                          d_model=run.config["embedding_size"],
    #                          nhead=run.config["num_heads"],
    #                          num_layers=run.config["num_layers"],
    #                          dim_feedforward=run.config["dim_feedforward"],
    #                          dropout=run.config["dropout"],
    #                          output_size=9,
    #                          mode=0)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run.config["lr"])
    # 定义学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 训练
    print("training model...")
    for epoch in range(run.config.epochs):
        train_model_on_epoch(model, train_loader, criterion, optimizer, epoch, run.config.epochs, device)
        validate_model(model, test_loader, criterion, epoch, run.config.epochs, device)
