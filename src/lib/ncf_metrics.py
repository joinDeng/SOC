import json, os
import argparse


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--ncf_dir', type=str, required=True, help='ncf dataset director')
    parse.add_argument('--json_file', type=str, default='space_object_metrics.json', help='metrics json file')
    args = parse.parse_args()
    metrics_json = args.json_file
    ncf_dir = args.ncf_dir

    metrics = json.load(open(metrics_json))
    ncf_ids = {f.split('_')[1].split('.')[0] for f in os.listdir(ncf_dir) if f.startswith('NoradCatID_')}
    for m in metrics:
        m['has_ncf'] = m['norad_id'] in ncf_ids
    json.dump(metrics, open('tle_metrics.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
