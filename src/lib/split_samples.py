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
    parser.add_argument('--split_mode', default='yearly', help='时间划分模式:yearly:1整年,monthly:逐月划分')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 加载样本索引
    with open(args.sample_index, 'r') as f:
        samples = json.load(f)

    # 打开HDF5文件
    h5 = h5py.File(args.h5_file, 'r')

    train_samples = []
    val_samples = []
    test_samples = []
    skipped_samples = []  # 时间范围外的样本

    if args.split_mode=="yearly":
        time_boundaries = config.get('time_split_yearly', {})
        train_end = time_boundaries.get('train_end', "2022-10-01 00:00:00")    # 2022-10-01 00:00 UTC
        train_end = utc_to_time_stamp(train_end)
        val_end = time_boundaries.get('val_end', "2022-11-16 00:00:00")        # 2022-11-16 00:00 UTC
        val_end = utc_to_time_stamp(val_end)
        test_end = time_boundaries.get('test_end', "2023-01-01 00:00:00")      # 2023-01-01 00:00 UTC
        test_end = utc_to_time_stamp(test_end)

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
        with open(f"{args.output_prefix}_{args.split_mode}_train.json", 'w') as f:
            json.dump(train_samples, f, indent=2)

        with open(f"{args.output_prefix}_{args.split_mode}_val.json", 'w') as f:
            json.dump(val_samples, f, indent=2)

        with open(f"{args.output_prefix}_{args.split_mode}_test.json", 'w') as f:
            json.dump(test_samples, f, indent=2)

        print(f"[INFO] Yearly split results: "
            f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}, skipped={len(skipped_samples)}")
    else:
        split_cfg = config.get('time_split_monthly', {})

        train_days = split_cfg['train_days']
        val_days = split_cfg['val_days']
        test_days = split_cfg['test_days']
        start_month = pd.to_datetime(split_cfg['start_month'])
        end_month = pd.to_datetime(split_cfg['end_month'])

        # 生成每月边界
        months = pd.date_range(start_month, end_month, freq='MS')

        for month_start in months:
            month_end = month_start + pd.DateOffset(months=1)
            # 三段边界（时间戳）
            train_end = (month_start + pd.DateOffset(days=train_days)).timestamp()
            val_end = (month_start + pd.DateOffset(days=train_days + val_days)).timestamp()
            test_end = (month_start + pd.DateOffset(days=train_days + val_days + test_days)).timestamp()

            for rec in samples:
                nid = rec['norad_id']
                s, e = rec['start_idx'], rec['end_idx']
                center_t = get_center_time(h5[nid], s, e)

                if month_start.timestamp()< center_t < train_end:
                    train_samples.append(rec)
                elif train_end < center_t < val_end:
                    val_samples.append(rec)
                elif val_end < center_t < test_end:
                    test_samples.append(rec)
                else:
                    skipped_samples.append(rec)

        h5.close()

        # 按月标识输出
        month_str = f"{start_month.strftime('%Y%m')}-{end_month.strftime('%Y%m')}"
        with open(f"{args.output_prefix}_{args.split_mode}_train.json", 'w') as f:
            json.dump(train_samples, f, indent=2)
        with open(f"{args.output_prefix}_{args.split_mode}_val.json", 'w') as f:
            json.dump(val_samples, f, indent=2)
        with open(f"{args.output_prefix}_{args.split_mode}_test.json", 'w') as f:
            json.dump(test_samples, f, indent=2)

        print(f"[INFO] Monthly split {month_str}: "
            f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}, skipped={len(skipped_samples)}")



if __name__ == '__main__':
    main()
