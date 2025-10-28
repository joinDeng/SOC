#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
一键绘制 selected 空间目标分布
> python plot_dataset_dist.py --ids selected_ids.json --metrics space_object_metrics.json --out dist.pdf
"""
import h5py
import json, argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

from lib.stratify_ids import categorize_orbit

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",   # 与 TNR 一致的数学字体
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

def main_plot_dataset_dist():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', default='../../../db/intermediate/selected_ids.json', help='selected_ids.json')
    parser.add_argument('--metrics', default='../../../db/space_object_metrics.json', help='space_object_metrics.json')
    parser.add_argument('--out', default='../output/other/dataset_dist.pdf', help='输出高清 PDF')
    args = parser.parse_args()

    # 1. 读数据
    ids = json.load(open(args.ids))
    with open(args.metrics, 'r') as f:    
        metrics = json.load(f)
    for m in metrics:
        m['orbit_class'] = categorize_orbit(m['mean_height'])
    # metrics = pd.read_json(args.metrics)          # List[Dict] → DataFrame
    metrics = pd.DataFrame(metrics)
    df = metrics[metrics['norad_id'].isin(ids)].copy()

    # 2. 打印分布
    print('==========  Selected 数据集分布 ==========')
    for col in ['final_cat', 'final_rcs', 'orbit_class']:
        cnt = df[col].value_counts()
        print(f'\n{col.upper()}  (N={len(df)})')
        for k, v in cnt.items():
            print(f'  {k:>10}: {v:>5} ({v/len(df)*100:5.1f}%)')

    # 3. 堆叠柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    title_dict = {'final_cat': 'Category', 'final_rcs': 'RCS size', 'orbit_class': 'Orbit type'}
    for ax, col in zip(axes, ['final_cat', 'final_rcs', 'orbit_class']):
        cnt = df[col].value_counts()
        cnt.plot(kind='bar', ax=ax, color=sns.color_palette('Set2'))

        ax.set_title(f'{title_dict[col]}')
        ax.set_xlabel('')
        ax.set_ylabel('Sample size')
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x()+p.get_width()/2, p.get_height()),
                        ha='center', va='bottom', fontsize=16)
    plt.tight_layout()
    fig.savefig(args.out.replace('.pdf', '_bar.pdf'), dpi=300)
    print(f'柱状图已保存：{args.out.replace(".pdf", "_bar.pdf")}')

    # 4. 交叉热力图（轨道 vs 类别）
    cross = pd.crosstab(df['orbit_class'], df['final_cat'])
    plt.figure(figsize=(9, 9))
    sns.heatmap(cross, annot=True, fmt='d', cmap='Blues')
    plt.title('Category × Orbit type')
    plt.xlabel('Category', fontsize=18)
    plt.ylabel('Orbit type', fontsize=18)
    plt.savefig(args.out.replace('.pdf', '_heatmap.pdf'), dpi=300)
    print(f'热力图已保存：{args.out.replace(".pdf", "_heatmap.pdf")}')

    print('==========  完成！ ==========')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画选中目标的时间序列长度分布
> python plot_seq_length.py --ids selected_ids.json --h5 ncf_20220101-20230101.h5 --out len_dist.pdf
"""

def main_plot_seq_length():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', default='../../../db/intermediate/selected_ids.json', help='selected_ids.json')
    parser.add_argument('--h5', default='../../../db/ncf_20220101-20230101.h5', help='ncf_20220101-20230101.h5')
    parser.add_argument('--out', default='../output/other/len_dist.pdf', help='输出高清 PDF')
    args = parser.parse_args()

    # 1. 读 ID 列表
    ids = json.load(open(args.ids))
    print(f'共 {len(ids)} 个选中目标')

    # 2. 遍历 H5 统计长度
    lengths = []
    with h5py.File(args.h5, 'r') as f:
        for nid in tqdm(ids, desc='统计长度'):
            if nid not in f:
                continue
            L = f[nid]['t'].shape[0]   # 时间戳维度
            lengths.append(L)
    lengths = np.array(lengths)
    print('========== 序列长度统计 ==========')
    print(f'均值: {lengths.mean():.1f}')
    print(f'中位数: {np.median(lengths):.0f}')
    print(f'Q1: {np.percentile(lengths, 25):.0f}')
    print(f'Q3: {np.percentile(lengths, 75):.0f}')
    print(f'最小: {lengths.min()}')
    print(f'最大: {lengths.max()}')

    # 3. 画图（三图一页）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 直方图
    ax = axes[0]
    ax.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    ax.axvline(lengths.mean(), color='red', linestyle='--', label=f'Mean={lengths.mean():.0f}')
    ax.axvline(np.median(lengths), color='orange', linestyle='--', label=f'Median={np.median(lengths):.0f}')
    ax.set_xlabel('Sequence Length (steps)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram')
    ax.legend()

    # 箱线图
    ax = axes[1]
    ax.boxplot(lengths, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='red'))
    ax.set_ylabel('Length (steps)')
    ax.set_title('Box Plot')

    # 累计分布
    ax = axes[2]
    sorted_len = np.sort(lengths)
    cum = np.arange(1, len(sorted_len)+1) / len(sorted_len)
    ax.plot(sorted_len, cum, linewidth=2, color='darkgreen')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(np.median(lengths), color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Length (steps)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f'分布图已保存 → {args.out}')

if __name__ == '__main__':
    main_plot_seq_length()
    