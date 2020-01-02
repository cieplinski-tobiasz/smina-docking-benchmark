import argparse

import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('-c', '--column', default='affinity')
    parser.add_argument('-n', '--n-top', type=int, default=3)

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    dataset = pd.read_csv(args.dataset).sort_values(args.column, ascending=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):
        print(dataset[:args.n_top])
