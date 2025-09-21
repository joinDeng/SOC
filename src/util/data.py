import os
import re
from collections import Counter
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as st
from scipy.fftpack import fft

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def read_ncf_file(filename) -> list:
    """
    Reads data from txt file and returns it as a list of lists.
    :param filename:
    :return:
    """
    pattern = re.compile('[0-9]{5}[_]{1}[0-8]{1}')
    start, end = pattern.search(filename).span()
    id_label = filename[start:end]
    NORAD_ID, category = id_label.split('_')
    with open(filename, 'r') as f:
        f.readline()  # title
        lines = f.read().splitlines()

        data = []
        for line in lines:
            t, ncfx, ncfy, ncfz = line.split(',')
            t = datetime.strptime(str(t).strip(), '%Y-%m-%dT%H:%M:%S.%f')
            ncfx, ncfy, ncfz = float(ncfx.strip()), float(ncfy.strip()), float(ncfz.strip())
            data.append([t, ncfx, ncfy, ncfz, int(category)])
    # print(data[:5])
    return data


def split_data(data, length_days: int = 1, simple_rate: int = 60, missing_value: float = 0.0):
    """
    Splits data into smaller time series.
    :param missing_value:
    :param data:
    :param length_days: the length of smaller time series.
    :param simple_rate: unit: second
    :return:
    """
    day2second = 86400
    length_seconds = length_days * day2second
    lclip = length_seconds // simple_rate
    clip = [[missing_value] * 3 for _ in range(lclip)]

    times, ncfs, labels = [], [], []
    t0 = datetime(data[0][0].year, data[0][0].month, data[0][0].day)
    for i in range(len(data)):
        t, ncfx, ncfy, ncfz, label = data[i]
        delta = (t - t0).total_seconds()

        index = int(delta % length_seconds // simple_rate)
        clip[index] = [ncfx, ncfy, ncfz]

        if index == lclip-1:
            times.append(datetime(t.year, t.month, t.day))
            ncfs.append(clip.copy())
            labels.append(label)

    # print(times[:5], ncfs[:5], labels[:5])
    return times, ncfs, labels


def normalize_data_for_multi_batch(data: np.ndarray):
    # data: batch x seq_len x input_size
    max_, min_ = [], []
    for i in range(len(data)):
        max_.append(np.max(data[i], axis=0))
        min_.append(np.min(data[i], axis=0))
    max = np.max(max_, axis=0)
    min = np.min(min_, axis=0)
    # print(max_ncf, min_ncf)

    return (data - min) / (max - min)


class NCFDataset(Dataset):
    def __init__(self, data_folder, shuffle=False, normalize=True, statistic=False, frequency=False):
        self.step = 60*4  # 60min为一组（1天中），进行统计处理
        self.data_folder = data_folder
        self.sos_time = []
        self.sos_ncf = []
        self.sos_label = []
        self.statistic = statistic
        self.frequency = frequency

        self.statistic_data = {'sos_time': {'date': [], 'lag': []},
                               'sos_ncf': {'mean': [], 'std': [], 'min': [], 'max': [], 'kurt': [], 'skew': []}}
        self.frequency_data = {'sos_ncf': {'freq_topk': [], 'amp_topk': []}}
        self.ncf_data = np.array([])

        self.load_data()
        assert len(self.sos_time) == len(self.sos_ncf) == len(self.sos_label)
        if shuffle:
            self.shuffle_data()

        self.sos_ncf = np.array(self.sos_ncf)
        # if normalize:
        #     self.sos_ncf = normalize_data_for_multi_batch(np.array(self.sos_ncf))
        if statistic:
            self.get_statistic_data_in_array()
            if normalize:
                self.ncf_data = normalize_data_for_multi_batch(self.ncf_data)
            # print(self.ncf_data[:5])
        if frequency:
            self.get_frequency_domain_data()
            if normalize:
                self.frequency_data = normalize_data_for_multi_batch(self.ncf_data)

        msg = Counter(self.sos_label)
        print(msg)

    def __getitem__(self, index):
        # 将列表转换为张量
        # seq_len x feature
        if not self.statistic and not self.frequency:
            x = torch.tensor(self.sos_ncf[index], dtype=torch.float32)
        elif self.statistic and not self.frequency:
            x = torch.tensor(self.ncf_data[index], dtype=torch.float32)
        elif not self.statistic and self.frequency:
            x = torch.tensor(self.frequency_data[index], dtype=torch.float32)
        else:
            x = torch.tensor(np.concatenate([self.ncf_data, self.frequency_data], axis=-1), dtype=torch.float32)

        y = torch.tensor(self.sos_label[index], dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.sos_label)

    def load_data(self):
        # folder_path = r"D:\app\program\Python\Pycharm\project\SOC\data"
        pattern = re.compile('-NCF')
        ncf_files = []
        for root, dirs, files in os.walk(self.data_folder):
            for file_name in files:
                if pattern.search(file_name):
                    file_path = os.path.join(root, file_name)
                    ncf_files.append(file_path)
        for file_path in ncf_files:
            data = read_ncf_file(file_path)
            so_times, so_ncfs, so_labels = split_data(data, length_days=1, simple_rate=60, missing_value=0.0)
            for st, sn, sl in zip(so_times, so_ncfs, so_labels):
                self.sos_time.append(st)
                self.sos_ncf.append(sn)
                self.sos_label.append(sl)

        # print(self.sos_time[::6])
        # print(self.sos_ncf[-1])
        # print(self.sos_label[::6])

    def shuffle_data(self):
        shuffle_index = np.arange(len(self.sos_label))
        np.random.shuffle(shuffle_index)

        sos_time_ = self.sos_time.copy()
        sos_ncf_ = self.sos_ncf.copy()
        sos_label_ = self.sos_label.copy()
        for i in range(len(self.sos_label)):
            self.sos_time[i] = sos_time_[shuffle_index[i]]
            self.sos_ncf[i] = sos_ncf_[shuffle_index[i]]
            self.sos_label[i] = sos_label_[shuffle_index[i]]

        print(self.sos_time[::6])
        print(self.sos_ncf[-1])
        print(self.sos_label[::6])

    def get_statistic_data_in_array(self):
        step = self.step
        for i in range(len(self.sos_label)):
            array = np.array(self.sos_ncf[i])
            for j in range(0, len(array), step):
                self.statistic_data['sos_ncf']['mean'].append(np.mean(array[j:j + step], axis=0))
                self.statistic_data['sos_ncf']['std'].append(np.std(array[j:j + step], axis=0))
                self.statistic_data['sos_ncf']['min'].append(np.min(array[j:j + step], axis=0))
                self.statistic_data['sos_ncf']['max'].append(np.max(array[j:j + step], axis=0))
                self.statistic_data['sos_ncf']['kurt'].append(st.kurtosis(array[j:j + step], axis=0))
                self.statistic_data['sos_ncf']['skew'].append(st.skew(array[j:j + step], axis=0))

        def reshape(data: list, shape=(-1, 1440//self.step, 3)):
            return np.array(data).reshape(shape[0], shape[1], shape[2])

        mean = reshape(self.statistic_data['sos_ncf']['mean'])
        std = reshape(self.statistic_data['sos_ncf']['std'])
        min = reshape(self.statistic_data['sos_ncf']['min'])
        max = reshape(self.statistic_data['sos_ncf']['max'])
        kurt = reshape(self.statistic_data['sos_ncf']['kurt'])
        skew = reshape(self.statistic_data['sos_ncf']['skew'])
        # print(mean.shape)
        # print(std.shape)
        # print(min.shape)
        # print(max.shape)
        # print(kurt.shape)
        # print(skew.shape)

        self.ncf_data = np.concatenate((mean, std, min, max, kurt, skew), axis=2)
        # print(self.ncf_data.shape)

    def get_frequency_domain_data(self, topk=9):

        pass


def LargeNCFDataset(data_folder, shuffle=False, normalize=True):
    def compute_global_statistics(data_path, batch_size=1000):
        """计算全局均值和标准差"""
        mean = None
        std = None
        total_samples = 0

        for batch in load_data_in_batches(data_path, batch_size):
            if mean is None:
                mean = np.zeros(batch.shape[1])
                std = np.zeros(batch.shape[1])
            mean += np.sum(batch, axis=0)
            std += np.sum(batch ** 2, axis=0)
            total_samples += batch.shape[0]

        mean /= total_samples
        std = np.sqrt(std / total_samples - mean ** 2)
        return mean, std

    def normalize_data(data_path, output_path, mean, std, batch_size=1000):
        """逐块归一化数据"""
        for batch in load_data_in_batches(data_path, batch_size):
            normalized_batch = (batch - mean) / std

    def load_data_in_batches(data_path, batch_size):
        """生成器函数，逐块加载数据"""
        # 实现你的数据加载逻辑
        # 这里假设数据存储为多个文件或一个大文件
        # 返回一个批次的数据
        pass


def test():
    folder_path = r"/data"
    dataset = NCFDataset(folder_path)
    print(dataset.statistic_data['sos_ncf']['mean'][:5])
    print(dataset.ncf_data[:5][:, :3])
    dataloader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=4)

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# test()



