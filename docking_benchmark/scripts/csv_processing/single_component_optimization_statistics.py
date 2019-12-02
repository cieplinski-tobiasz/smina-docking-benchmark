import argparse
import os

import pandas as pd

from docking_benchmark.scripts.metrics.column_metrics import make_metrics


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_files', nargs='+')
    parser.add_argument('-c', '--columns', nargs='+', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    for csv_file in args.csv_files:
        path = os.path.abspath(csv_file)
        csv = pd.read_csv(path)

        print(f'{path}:')

        for column in args.columns:
            if column not in csv.columns:
                print(f' No column {column} in {path}')
                print()
                continue

            print(f' {column}')

            for metric in make_metrics(column):
                print(f'  {metric.name}: {metric.function(csv)}')

            print()

        print()
