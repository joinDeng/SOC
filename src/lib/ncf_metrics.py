import json
import os
import argparse
import h5py


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--ncf_hdf5', type=str, required=True, help='ncf data file')
    parse.add_argument('--metrics_json', type=str, default='space_object_metrics.json', help='metrics json file')
    args = parse.parse_args()
    ncf_hdf5 = args.ncf_hdf5
    metrics_json = args.metrics_json

    ncf_ids = set()
    with h5py.File(ncf_hdf5, 'r') as h5f:
        for key in h5f.keys():
            if key not in ncf_ids:
                ncf_ids.add(key)
    metrics = json.load(open(metrics_json))

    # 先清空has_ncf记录，再更新
    for m in metrics:
        m['has_ncf'] = False
    for m in metrics:
        m['has_ncf'] = m['norad_id'] in ncf_ids
    json.dump(metrics, open(metrics_json, 'w'), indent=2)


if __name__ == '__main__':
    main()
