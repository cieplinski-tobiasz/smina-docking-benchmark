import argparse
import collections
import math

import pandas as pd

SMILES_COLUMN = 'SMILES'
DOCKING_COLUMN = 'PREDICTED_DOCKING_SCORE'

Metric = collections.namedtuple('Metric', ['name', 'function'])


def make_mean_top_n(n, column=DOCKING_COLUMN, ascending=True):
    def mean_top_n(df):
        # Sort ascending, because the lesser docking score value
        # the better
        df = df.sort_values(column, ascending=ascending)[:n]
        return df[column].mean()

    return mean_top_n


def make_mean_top_n_percent(n, column=DOCKING_COLUMN, ascending=True):
    def mean_top_n_percent(df):
        size, _ = df.shape
        # Sort ascending, because the lesser docking score value
        # the better
        df = df.sort_values(column, ascending=ascending)[:math.ceil(n / 100 * size)]
        return df[column].mean()

    return mean_top_n_percent


def make_mean(column=DOCKING_COLUMN):
    def docking_mean(df):
        return df[column].mean()

    return docking_mean


def make_metrics(column=DOCKING_COLUMN):
    return (
        Metric('Mean Top 10 (ascending)', make_mean_top_n(10, column=column, ascending=True)),
        Metric('Mean Top 10 (descending)', make_mean_top_n(10, column=column, ascending=False)),
        Metric('Mean Top 1% (ascending)', make_mean_top_n_percent(1, column=column, ascending=True)),
        Metric('Mean Top 1% (descending)', make_mean_top_n_percent(1, column=column, ascending=False)),
        Metric('Mean', make_mean(column=column)),
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str)
    parser.add_argument('-c', '--column', type=str, default=DOCKING_COLUMN)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    scores = pd.read_csv(args.csv)

    for m in make_metrics(args.column):
        print(f'{m.name}: {m.function(scores)}')
