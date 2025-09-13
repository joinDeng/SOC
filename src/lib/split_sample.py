#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在上一步 tle_metrics.json 基础上做可靠性过滤 + 分层均衡抽样
> python sample_split.py --metrics tle_metrics.json --out split.json
"""
import json, argparse, pandas as pd, numpy as np
from datetime import datetime
from tqdm import tqdm

# ---------- 默认超参 ----------
CAT_REL_TH   = 0.7     # 类别可靠性阈值
RCS_REL_TH   = 0.7     # RCS  可靠性阈值
MAX_PER_CELL = 500     # 每格（类别×轨道×RCS）上限
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train/val/test


# ---------- 函数 ----------
def load_metrics(path):
    with open(path, encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))


def add_orbit_class(df):
    """按平均高度分 LEO/MEO/GEO"""
    conditions = [
        df['mean_height'] < 2000,
        df['mean_height'] < 20000
    ]
    choices = ['LEO', 'MEO']
    df['orbit_class'] = np.select(conditions, choices, default='GEO')
    return df


def time_split(df, ratios=SPLIT_RATIOS):
    """按 first_date 时间排序后切分"""
    df = df.sort_values('first_date')
    n = len(df)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train+n_val]
    test  = df.iloc[n_train+n_val:]
    return train, val, test


def sample_balance(df, max_cell=MAX_PER_CELL):
    """分层均衡抽样"""
    key = ['final_cat', 'orbit_class', 'final_rcs']
    grouped = df.groupby(key, dropna=False)
    sampled = []
    for grp_key, sub in grouped:
        if len(sub) > max_cell:
            sub = sub.sample(n=max_cell, random_state=42)
        sampled.append(sub)
    return pd.concat(sampled).reset_index(drop=True)


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True, help='tle_metrics.json')
    parser.add_argument('--out', default='split.json', help='输出划分文件')
    parser.add_argument('--report', default='balance_report.csv', help='抽样报告')
    parser.add_argument('--cat_th', type=float, default=CAT_REL_TH)
    parser.add_argument('--rcs_th', type=float, default=RCS_REL_TH)
    parser.add_argument('--max_cell', type=int, default=MAX_PER_CELL)
    args = parser.parse_args()

    # 1. 读入
    df = load_metrics(args.metrics)
    print(f'[INFO] 原始目标 {len(df)}')

    # 2. 过滤可靠性 + 必须有 NCF
    df = add_orbit_class(df)
    mask = (df['cat_reliability'] >= args.cat_th) & \
           (df['rcs_reliability'] >= args.rcs_th) & \
           (df['has_ncf'] == True)
    df = df[mask].copy()
    print(f'[INFO] 可靠性过滤后 {len(df)}')

    # 3. 分层均衡抽样
    df_bal = sample_balance(df, max_cell=args.max_cell)
    print(f'[INFO] 均衡抽样后 {len(df_bal)}')

    # 4. 时间顺序切分
    train, val, test = time_split(df_bal)
    split = {
        'train': train['norad_id'].tolist(),
        'val':   val['norad_id'].tolist(),
        'test':  test['norad_id'].tolist()
    }
    with open(args.out, 'w') as f:
        json.dump(split, f, indent=2)
    print(f'[INFO] 划分已写入 {args.out}')
    print(f'       train {len(train)} | val {len(val)} | test {len(test)}')

    # 5. 生成抽样报告
    report = df_bal.groupby(['final_cat', 'orbit_class', 'final_rcs']).size().reset_index(name='count')
    report.to_csv(args.report, index=False)
    print(f'[INFO] 报告已写入 {args.report}')


if __name__ == '__main__':
    main()
