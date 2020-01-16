import argparse

import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_files', nargs='+', help='.csv files to be merged')
    parser.add_argument('-o', '--output', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    df = pd.concat([pd.read_csv(file) for file in args.csv_files], copy=False)
    df.to_csv(args.output, index=False)
