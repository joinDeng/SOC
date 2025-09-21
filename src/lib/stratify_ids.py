#!/usr/bin/env python3
"""
步骤1: 分层抽样NORAD ID
输入: tle_metrics.json
输出: selected_ids.json, rare_ids.json
"""
import json
import argparse
import random
from collections import defaultdict


def categorize_orbit(mean_height):
    """根据平均高度分类轨道类型"""
    if mean_height < 1200:
        return 'LEO'
    elif 1200 <= mean_height < 35786:
        return 'MEO'
    else:
        return 'HEO'


def categorize_orbit_v2(mean_height, eccentricity):
    """改进版轨道分类（符合国际标准）"""
    if mean_height < 2000:  # 标准LEO范围
        return 'LEO'
    elif 2000 <= mean_height < 35786:  # MEO范围
        return 'MEO'
    elif abs(mean_height - 35786) <= 100 and eccentricity <= 0.001:  # 严格GEO条件
        return 'GEO'
    elif mean_height >= 35786:  # HEO或其他
        return 'HEO' if eccentricity > 0.1 else 'GEO'  # 'NEAR_GEO'
    else:
        return 'OTHER'


def main():
    parser = argparse.ArgumentParser(description='分层抽样NORAD ID')
    parser.add_argument('--metrics', required=True, help='输入metrics文件路径')
    parser.add_argument('--output', required=True, help='输出选中的ID列表文件路径')
    parser.add_argument('--rare_output', required=True, help='输出稀缺ID列表文件路径')
    parser.add_argument('--config', default='../config/pipeline_config.json', help='配置文件路径')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    grid_min = config.get('grid_min', 50)
    grid_max = config.get('grid_max', 500)

    # 加载metrics数据
    with open(args.metrics, 'r') as f:
        metrics = json.load(f)

    # 过滤掉没有ncf数据、cat和rcs可靠性较低的目标
    metrics = [m for m in metrics if m.get('has_ncf') and m.get('cat_reliability') > 0.6 and m.get('rcs_reliability') > 0.6]

    # 为每个目标计算轨道类型
    for m in metrics:
        m['orbit_class'] = categorize_orbit(m['mean_height'])

    # 按(轨道类型, 类别, RCS)分组
    pool = defaultdict(list)
    for rec in metrics:
        g = (rec['orbit_class'], rec['final_cat'], rec['final_rcs'])
        pool[g].append(rec['norad_id'])

    selected_ids = []
    rare_ids = []  # 记录稀缺网格中的ID

    for g, ids in pool.items():
        if len(ids) < grid_min:
            # 稀缺网格，全部选入，并标记为稀缺
            selected_ids.extend(ids)
            rare_ids.extend(ids)
        elif len(ids) > grid_max:
            # 过多，随机采样grid_max个
            selected_ids.extend(random.sample(ids, grid_max))
        else:
            selected_ids.extend(ids)

    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(selected_ids, f, indent=2)

    with open(args.rare_output, 'w') as f:
        json.dump(rare_ids, f, indent=2)

    print(f"[INFO] Selected {len(selected_ids)} NORAD IDs. Rare IDs: {rare_ids}")


if __name__ == '__main__':
    main()
