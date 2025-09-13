#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历 space_object_tle/NoradCatID_xxxxx.txt
计算类别/RCS 可靠性、轨道根数等指标
> python tle_metrics.py --tle_dir space_object_tle --out tle_metrics.json
"""
import os, json, argparse, numpy as np, pandas as pd
from tqdm import tqdm
from collections import defaultdict

ALPHA = 0.15      # 类别可靠性衰减系数
BETA  = 0.15      # RCS  可靠性衰减系数
RE    = 6378.137  # 地球赤道半径 km


# ---------- 工具函数 ----------
def calc_height(a, e):
    """给定半长轴 km、偏心率，返回 (hp, ha, h_avg)"""
    hp = a * (1 - e) - RE
    ha = a * (1 + e) - RE
    return hp, ha, (hp + ha) / 2


def reliability_score(switch_cnt, unknown_ratio, gamma):
    return np.exp(-gamma * switch_cnt) * (1 - unknown_ratio)


# ---------- 指标计算器 ----------
class TLEMetrics:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, skipinitialspace=True)
        self.df['EPOCH'] = pd.to_datetime(self.df['EPOCH'])
        self.df = self.df.sort_values('EPOCH')  # 时间序
        self.norad_id = os.path.basename(file_path).split('_')[1].split('.')[0]

    # ---- 类别通道 ----
    def cat_metrics(self):
        s = self.df['OBJECT_TYPE'].str.lower()
        valid = s[s.isin(['payload', 'rocket body', 'debris'])]
        switch = (valid != valid.shift()).sum() - 1
        switch = int(max(0, switch))
        unk_ratio = s.isin(['unknown', 'tba', 'tbd', 'null', '']).mean()
        final = valid.iloc[-1] if len(valid) else 'unknown'
        R = reliability_score(switch, unk_ratio, ALPHA)
        return {
            'final_cat': final,
            'cat_switches': switch,
            'cat_unknown_ratio': round(unk_ratio, 3),
            'cat_reliability': round(R, 3)
        }

    # ---- RCS 通道 ----
    def rcs_metrics(self):
        s = self.df['RCS_SIZE'].str.lower()
        # valid = s[~s.isin(['unknown', ''])]
        valid = s[s.isin(['small', 'medium', 'large'])]
        switch = (valid != valid.shift()).sum() - 1
        switch = int(max(0, switch))
        unk_ratio = s.isin(['unknown', 'tba', 'tbd', 'null', '']).mean()
        final = valid.iloc[-1] if len(valid) else 'unknown'
        R = reliability_score(switch, unk_ratio, BETA)
        return {
            'final_rcs': final,
            'rcs_switches': switch,
            'rcs_unknown_ratio': round(unk_ratio, 3),
            'rcs_reliability': round(R, 3)
        }

    # ---- 轨道根数 ----
    def orbit_metrics(self):
        a   = self.df['SEMIMAJOR_AXIS'].iloc[-1]  # 用最新一条
        ecc = self.df['ECCENTRICITY'].iloc[-1]
        hp, ha, h_avg = calc_height(a, ecc)
        return {
            'perigee_height': round(hp, 1),
            'apogee_height': round(ha, 1),
            'mean_height': round(h_avg, 1),
            'eccentricity': round(ecc, 6)
        }

    # ---- 汇总 ----
    def to_dict(self):
        return {
            'norad_id': self.norad_id,
            **self.cat_metrics(),
            **self.rcs_metrics(),
            **self.orbit_metrics(),
            'has_ncf': None  # 占位，后续 merge
        }


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tle_dir', required=True, help='space_object_tle 文件夹')
    parser.add_argument('--out', default='space_object_metrics.json', help='输出 json')
    args = parser.parse_args()

    files = [os.path.join(args.tle_dir, f) for f in os.listdir(args.tle_dir)
             if f.startswith('NoradCatID_') and f.endswith('.txt')]
    print(f'[INFO] 共 {len(files)} 个 TLE 文件')

    pool = []
    for fp in tqdm(files, desc='Processing TLE'):
        try:
            pool.append(TLEMetrics(fp).to_dict())
        except Exception as e:
            print(f'[WARN] skip {fp}: {e}')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    print(f'[INFO] 已写入 {args.out}')


if __name__ == '__main__':
    main()
