#!/usr/bin/env python3
"""
步骤4: 生成HDF5数据集
输入: {train,val,test}_samples.json, space_objects.h5
输出: {train,val,test}.h5
"""
import json
import argparse
import h5py
import numpy as np
from tqdm import tqdm


def main():
    # parser = argparse.ArgumentParser(description='生成HDF5数据集')
    # parser.add_argument('--h5_file', required=True, help='原始HDF5文件路径')
    # parser.add_argument('--sample_index', required=True, help='样本索引文件路径')
    # parser.add_argument('--split_name', required=True, choices=['train', 'val', 'test'], help='数据集名称')
    # parser.add_argument('--output', required=True, help='输出HDF5文件路径')
    # parser.add_argument('--config', default='../config/pipeline_config.json', help='配置文件路径')

    parser = argparse.ArgumentParser(description='生成HDF5数据集')
    parser.add_argument('--h5_file', default='../../data/space.h5', help='原始HDF5文件路径')
    parser.add_argument('--sample_index', default='../../data/sample_index.json', help='样本索引文件路径')
    parser.add_argument('--split_name', default='train', help='数据集名称')
    parser.add_argument('--output', default='../../data/train.h5', help='输出HDF5文件路径')
    parser.add_argument('--config', default='../config/pipeline_config.json', help='配置文件路径')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    noise_level = config.get('noise_level', 0.01)
    compression = config.get('compression', {})
    comp_alg = compression.get('algorithm', 'gzip')
    comp_level = compression.get('level', 9)

    # 加载样本索引
    with open(args.sample_index, 'r') as f:
        samples = json.load(f)

    # 创建输出文件
    with h5py.File(args.output, 'w') as dst, \
            h5py.File(args.h5_file, 'r') as src:

        for rec in tqdm(samples, desc=f'Writing {args.split_name}'):
            nid = rec['norad_id']
            s, e = rec['start_idx'], rec['end_idx']
            aug = rec['aug_noise']

            # 验证集和测试集中不进行数据增强
            if args.split_name != 'train' and aug:
                continue

            # 创建唯一组名
            cnt = 0
            grp_name = f"{nid}_{s}_{e}_{cnt}"
            while grp_name in dst:
                cnt += 1
                grp_name = f"{nid}_{s}_{e}_{cnt}"
            grp = dst.create_group(grp_name)

            # 复制数据并应用增强
            for k in ['t', 'pos', 'vel', 'ncf']:
                data = src[nid][k][s:e]

                if args.split_name == 'train' and aug and k != 't':  # 仅在训练集中，对非时间数据添加噪声
                    noise = np.random.normal(0, noise_level, data.shape)
                    data = data + noise

                grp.create_dataset(
                    k,
                    data=data,
                    compression=comp_alg,
                    compression_opts=comp_level
                )

            # 保存网格信息作为属性
            grp.attrs['grid'] = json.dumps(rec['grid'])

    print(f"[INFO] Finished writing {len(samples)} samples to {args.output}")


if __name__ == '__main__':
    main()
