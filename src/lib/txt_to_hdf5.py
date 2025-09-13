#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 space_object/<ID>/ncf/*.txt 与 orbit/*.txt 合并成单个 HDF5 文件
> python txt2hdf5.py --root /data/space_object --out space.h5
"""
import os
import json
import argparse
import h5py
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathos.pools import ProcessPool as Pool

DATE_FMT = "%Y-%m-%dT%H:%M:%S.%f"


def parse_one_object(dir_name: str, root: str):
    """
    dir_name: 例如 '12345_payload'
    返回 dict or None
    """
    try:
        # ----   1. 拆分 ID 与标签 ----
        norad_id, label = dir_name.split('_', 1)  # 只切一次，防止标签里还有下划线
        obj_types = ['payload', 'rocket body', 'debris']
        rcs_sizes = ['small', 'medium', 'large']
        try:
            cat_label = obj_types[int(label) // 3]
            rcs_label = rcs_sizes[int(label) % 3]
        except TypeError as e:
            print(e, 'in label')

        ncf_dir   = os.path.join(root, dir_name, "ncf")
        orbit_dir = os.path.join(root, dir_name, "orbit")
        ncf_txt   = [f for f in os.listdir(ncf_dir)   if f.endswith(".txt")][0]
        orbit_txt = [f for f in os.listdir(orbit_dir) if f.endswith(".txt")][0]
        ncf_path  = os.path.join(ncf_dir, ncf_txt)
        orbit_path= os.path.join(orbit_dir, orbit_txt)

        # ---- 2. 读取、对齐（同旧脚本） ----
        ncf_df  = pd.read_csv(ncf_path,  header=None, names=["datetime","ncf_x","ncf_y","ncf_z"], skiprows=1, sep='\s*,\s*', engine='python')
        orbit_df= pd.read_csv(orbit_path,header=None, names=["datetime","pos_x","pos_y","pos_z","vel_x","vel_y","vel_z"], skiprows=1, sep='\s*,\s*', engine='python')
        ncf_df["datetime"]  = pd.to_datetime(ncf_df["datetime"], format=DATE_FMT)
        orbit_df["datetime"]= pd.to_datetime(orbit_df["datetime"], format=DATE_FMT)
        df = pd.merge(ncf_df, orbit_df, on="datetime", how="inner")
        if len(df) < 30:
            return None

        # ---- 3. 打包 ----
        t   = df["datetime"].astype(np.int64)//10**9
        pos = df[["pos_x","pos_y","pos_z"]].values
        vel = df[["vel_x","vel_y","vel_z"]].values
        ncf = df[["ncf_x","ncf_y","ncf_z"]].values
        return {
            "norad_id": norad_id,
            "t": t,
            "pos": pos,
            "vel": vel,
            "ncf": ncf
        }
    except Exception as e:
        print(f"[ERROR] {dir_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="space_object根目录")
    parser.add_argument("--out", type=str, default="space.h5", help="输出HDF5文件")
    args = parser.parse_args()

    # ---- 1. 扫描所有 NORAD ID ----
    root = args.root
    all_ids = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    print(f"[INFO] 找到 {len(all_ids)} 个 NORAD ID")

    # ---- 2. 多进程解析 ----
    with Pool() as pool:
        results = list(tqdm(pool.map(lambda x: parse_one_object(x, root), all_ids), total=len(all_ids)))
    pool.close()
    pool.join()

    # ---- 3. 过滤失败的 ----
    results = [r for r in results if r is not None]
    print(f"[INFO] 成功解析 {len(results)} 个目标")

    # ---- 4. 写入 HDF5 ----
    with h5py.File(args.out, "w") as h5f:
        for res in tqdm(results):
            if res is None:
                continue
            norad_id = res["norad_id"]
            grp = h5f.create_group(norad_id)
            grp.attrs["label"] = res["label"]  # 🔧 把类别存成属性
            grp.create_dataset("t", data=res["t"], compression="gzip", compression_opts=9)
            grp.create_dataset("pos", data=res["pos"], compression="gzip", compression_opts=9)
            grp.create_dataset("vel", data=res["vel"], compression="gzip", compression_opts=9)
            grp.create_dataset("ncf", data=res["ncf"], compression="gzip", compression_opts=9)

# def main():
#     # ---- 1. 扫描所有 NORAD ID ----
#     root = "D:\\app\\program\\Python\\Pycharm\\project\\SOC\\data"
#     all_ids = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
#     print(f"[INFO] 找到 {len(all_ids)} 个 NORAD ID")
#
#     # ---- 2. 多进程解析 ----
#     with Pool() as pool:
#         results = list(tqdm(pool.map(lambda x: parse_one_object(x, root), all_ids), total=len(all_ids)))
#     pool.close()
#     pool.join()
#
#     # ---- 3. 过滤失败的 ----
#     results = [r for r in results if r is not None]
#     print(f"[INFO] 成功解析 {len(results)} 个目标")
#
#     # ---- 4. 写入 HDF5 ----
#     out = "D:\\app\\program\\Python\\Pycharm\\project\\SOC\\dataset\\space.h5"
#     with h5py.File(out, "w") as h5f:
#         for res in tqdm(results):
#             if res is None:
#                 continue
#             norad_id = res["norad_id"]
#             grp = h5f.create_group(norad_id)
#             grp.attrs["label"] = res["label"]  # 🔧 把类别存成属性
#             grp.create_dataset("t", data=res["t"], compression="gzip", compression_opts=9)
#             grp.create_dataset("pos", data=res["pos"], compression="gzip", compression_opts=9)
#             grp.create_dataset("vel", data=res["vel"], compression="gzip", compression_opts=9)
#             grp.create_dataset("ncf", data=res["ncf"], compression="gzip", compression_opts=9)


if __name__ == "__main__":
    main()
