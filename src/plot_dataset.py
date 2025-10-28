#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
一键绘制 selected 空间目标分布
> python plot_dataset_dist.py --ids selected_ids.json --metrics space_object_metrics.json --out dist.pdf
"""
import json, argparse, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
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

def main():
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

if __name__ == '__main__':
    main()