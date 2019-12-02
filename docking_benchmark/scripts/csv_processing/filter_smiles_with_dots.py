import argparse
import os

import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-o', '--output')
    parser.add_argument('-sc', '--smiles-column', default='SMILES')
    parser.add_argument('-t', '--text-file', action='store_true')
    arguments = parser.parse_args()
    arguments.file = os.path.abspath(arguments.file)

    if arguments.output is None:
        arguments.output = arguments.file + '.nodots'

    return arguments


if __name__ == '__main__':
    args = _parse_args()

    if args.text_file:
        with open(args.file) as in_file, open(args.output, 'w') as out_file:
            for line in in_file:
                if '.' not in line and 'i' not in line and 'I' not in line:
                    out_file.write(line)
    else:
        df = pd.read_csv(args.file)
        df = df.loc[~df[args.smiles_column].str.contains(r'\.')]
        df.to_csv(args.output, index=False)
