import argparse

import numpy as np
import pandas as pd

from docking_benchmark.utils.chemistry import calculate_pairwise_similarities


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generated', required=True)
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-o', '--output')
    parser.add_argument('-sc', '--smiles-column', default='SMILES')

    arguments = parser.parse_args()

    if not arguments.output:
        arguments.output = arguments.generated

    return arguments


if __name__ == '__main__':
    args = _parse_args()
    generated = pd.read_csv(args.generated)
    dataset = pd.read_csv(args.dataset)

    tanimoto_similarities = calculate_pairwise_similarities(
        generated[args.smiles_column].tolist(),
        dataset[args.smiles_column].tolist()
    )
    most_similars = np.argmax(tanimoto_similarities, axis=1)
    ix_to_ix = np.array([[i[0], val] for i, val in np.ndenumerate(most_similars)])

    max_similarities = np.array([tanimoto_similarities[i, most_similars[i]] for i in range(generated.shape[0])])

    generated['MAX_TANIMOTO_SIMILARITY'] = max_similarities
    generated['MOST_SIMILAR_SMILES'] = dataset[args.smiles_column][most_similars].tolist()

    generated.to_csv(args.output, index=False, header=True)
