#!/usr/bin/env python3
"""
步骤2: 构建样本级索引
输入: selected_ids.json, rare_ids.json, space_objects.h5
输出: sample_index.json
"""
import json
import argparse
import h5py
import numpy as np
from tqdm import tqdm

from stratify_ids import categorize_orbit


def slide_window(T, window_length, slide_step):
    """生成滑动窗口"""
    if T < window_length:
        return [(0, T)]
    return [(i, i + window_length) for i in range(0, T - window_length + 1, slide_step)]


def main():
    # parser = argparse.ArgumentParser(description='构建样本级索引')
    # parser.add_argument('--h5_file', required=True, help='原始HDF5文件路径')
    # parser.add_argument('--selected_ids', required=True, help='选中的ID列表文件路径')
    # parser.add_argument('--rare_ids', required=True, help='稀缺ID列表文件路径')
    # parser.add_argument('--metrics', required=True, help='metrics文件路径')
    # parser.add_argument('--output', required=True, help='输出样本索引文件路径')
    # parser.add_argument('--config', default='../config/pipeline_config.json', help='配置文件路径')

    parser = argparse.ArgumentParser(description='构建样本级索引')
    file_required = '../../data/space_object_metrics.json'
    parser.add_argument('--h5_file', default='../../data/space.h5', help='原始HDF5文件路径')
    parser.add_argument('--selected_ids', default='../../data/selected_object_ids.json', help='选中的ID列表文件路径')
    parser.add_argument('--rare_ids', default='../../data/rare_object_ids.json', help='稀缺ID列表文件路径')
    parser.add_argument('--metrics', default='../../data/space_object_metrics.json', help='metrics文件路径')
    parser.add_argument('--output', default='../../data/sample_index.json', help='输出样本索引文件路径')
    parser.add_argument('--config', default='../config/pipeline_config.json', help='配置文件路径')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    window_length = config.get('window_length', 1440)
    slide_step = config.get('slide_step', 360)
    grid_min = config.get('grid_min', 5)
    augmentation_multiplier = config.get('augmentation_multiplier', 3)

    # 加载数据
    with open(args.selected_ids, 'r') as f:
        selected_ids = json.load(f)

    with open(args.rare_ids, 'r') as f:
        rare_ids = set(json.load(f))

    with open(args.metrics, 'r') as f:
        metrics_data = json.load(f)

    # 创建NORAD ID到网格的映射
    grid_of = {}
    for m in metrics_data:
        nid = m['norad_id']
        grid_of[nid] = {
            'orbit_class': categorize_orbit(m['mean_height']),
            'final_cat': m['final_cat'],
            'final_rcs': m['final_rcs']
        }

    # 打开HDF5文件
    h5 = h5py.File(args.h5_file, 'r')

    samples = []

    # 对每个ID进行滑动窗口分割
    for nid in tqdm(selected_ids, desc='Processing NORAD IDs'):
        if nid not in h5:
            print(f"Warning: {nid} not found in HDF5 file. Skipping.")
            continue

        grp = h5[nid]
        T = grp['t'].shape[0]

        # 生成滑动窗口
        windows = slide_window(T, window_length, slide_step)

        # 确定是否需要增强
        needs_augmentation = nid in rare_ids

        # 添加原始窗口
        for s, e in windows:
            samples.append({
                "norad_id": nid,
                "start_idx": s,
                "end_idx": e,
                "aug_noise": False,
                "grid": grid_of.get(nid, {})
            })

        # 对稀缺网格进行过采样
        if needs_augmentation:
            for _ in range(augmentation_multiplier):
                for s, e in windows:
                    samples.append({
                        "norad_id": nid,
                        "start_idx": s,
                        "end_idx": e,
                        "aug_noise": True,  # 标记为需要增强
                        "grid": grid_of.get(nid, {})
                    })

    h5.close()

    # 保存样本索引
    with open(args.output, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"[INFO] Generated {len(samples)} samples.")


if __name__ == '__main__':
    main()
