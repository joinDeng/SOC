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
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

ALPHA = 0.15      # 类别可靠性衰减系数
BETA  = 0.15      # RCS  可靠性衰减系数
RE    = 6378.137  # 地球赤道半径 km
DATE_FMT = "%Y-%m-%dT%H:%M:%S.%f"

# ---------- 工具函数 ----------
def calc_height(a, e):
    """给定半长轴 km、偏心率，返回 (hp, ha, h_avg)"""
    hp = a * (1 - e) - RE
    ha = a * (1 + e) - RE
    return hp, ha, (hp + ha) / 2


def reliability_score(switch_cnt, unknown_ratio, gamma):
    return np.exp(-gamma * switch_cnt) * (1 - unknown_ratio)


# 参数范围设置
switch_cnt_range = np.arange(0, 10, 1)
unknown_ratios = [0.1, 0.3, 0.5]
gamma_values = [0.1, 0.5, 1.0, 2.0]

# 创建交互式图表
plt.figure(figsize=(12, 8))
ax = plt.subplot(111)
plt.subplots_adjust(bottom=0.3)

# 初始绘制
lines = []
for ur in unknown_ratios:
    line, = ax.plot(switch_cnt_range,
                   reliability_score(switch_cnt_range, ur, gamma_values[0]),
                   label=f'Unknown ratio={ur}')
    lines.append(line)

# 图表装饰
ax.set_xlabel('Switch Count')
ax.set_ylabel('Reliability Score')
ax.set_title('Reliability Score vs Switch Count\n(Gamma={})'.format(gamma_values[0]))
ax.grid(True)
ax.legend()

# 添加gamma滑动条
axgamma = plt.axes([0.2, 0.1, 0.6, 0.03])
gamma_slider = Slider(axgamma, 'Gamma', 0.1, 3.0, valinit=gamma_values[0])

# 更新函数
def update(val):
    gamma = gamma_slider.val
    for i, ur in enumerate(unknown_ratios):
        lines[i].set_ydata(reliability_score(switch_cnt_range, ur, gamma))
    ax.set_title('Reliability Score vs Switch Count\n(Gamma={:.2f})'.format(gamma))
    plt.draw()

gamma_slider.on_changed(update)
plt.show()



