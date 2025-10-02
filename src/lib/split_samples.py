#!/usr/bin/env python3
"""
步骤3: 时间划分样本
输入: sample_index.json, space_objects.h5
输出: train_samples.json, val_samples.json, test_samples.json
"""
import json
import argparse
import h5py
import numpy as np
import pandas as pd


def get_center_time(grp, start_idx, end_idx):
    """计算窗口中心时间"""
    t_start = grp['t'][start_idx]
    t_end = grp['t'][end_idx - 1]
    return (t_start + t_end) / 2


def utc_to_time_stamp(t):
    """时间转换为 pandas 的 datetime 格式"""
    return pd.to_datetime(t, format="%Y-%m-%d %H:%M:%S").timestamp()


def main():
    parser = argparse.ArgumentParser(description='时间划分样本')
    parser.add_argument('--h5_file', required=True, help='原始HDF5文件路径')
    parser.add_argument('--sample_index', required=True, help='样本索引文件路径')
    parser.add_argument('--output_prefix', required=True, help='输出文件前缀')
    parser.add_argument('--config', default='../config/pipeline_config.json', help='配置文件路径')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    time_boundaries = config.get('time_boundaries', {})
    train_end = time_boundaries.get('train_end', "2022-10-01 00:00:00")    # 2022-10-01 00:00 UTC
    train_end = utc_to_time_stamp(train_end)
    val_end = time_boundaries.get('val_end', "2022-11-16 00:00:00")        # 2022-11-16 00:00 UTC
    val_end = utc_to_time_stamp(val_end)
    test_end = time_boundaries.get('test_end', "2023-01-01 00:00:00")      # 2023-01-01 00:00 UTC
    test_end = utc_to_time_stamp(test_end)

    # 加载样本索引
    with open(args.sample_index, 'r') as f:
        samples = json.load(f)

    # 打开HDF5文件
    h5 = h5py.File(args.h5_file, 'r')

    train_samples = []
    val_samples = []
    test_samples = []
    skipped_samples = []  # 时间范围外的样本

    for rec in samples:
        nid = rec['norad_id']
        s, e = rec['start_idx'], rec['end_idx']

        # 获取窗口中心时间
        center_t = get_center_time(h5[nid], s, e)

        # 根据时间划分
        if center_t < train_end:
            train_samples.append(rec)
        elif center_t < val_end:
            val_samples.append(rec)
        elif center_t <= test_end:
            test_samples.append(rec)
        else:
            skipped_samples.append(rec)

    h5.close()

    # 保存划分结果
    with open(f"{args.output_prefix}_train.json", 'w') as f:
        json.dump(train_samples, f, indent=2)

    with open(f"{args.output_prefix}_val.json", 'w') as f:
        json.dump(val_samples, f, indent=2)

    with open(f"{args.output_prefix}_test.json", 'w') as f:
        json.dump(test_samples, f, indent=2)

    print(
        f"[INFO] Split results: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}, skipped={len(skipped_samples)}")


if __name__ == '__main__':
    main()
