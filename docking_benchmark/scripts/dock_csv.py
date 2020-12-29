#!/usr/bin/env python3

"""
Script that docks ligands passed in a .csv file,
stores the docked ligand in .mol2 format
and also stores the interaction statistics
in a text file with .ita extension.
"""

import argparse
import logging
import os
import os.path

import pandas as pd

from docking_benchmark.data import proteins

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ligands')
    parser.add_argument('-s', '--smiles-column', default='SMILES')
    parser.add_argument('-c', '--csv-name', default='molecules.csv')
    parser.add_argument('-p', '--protein', default='5ht1b')
    parser.add_argument('--n-cpu', type=int, default=8)
    parser.add_argument('-o', '--output-dir')
    parser.add_argument('-i', '--start-index', default=1, type=int)
    arguments = parser.parse_args()

    if not arguments.output_dir:
        arguments.output_dir = os.path.dirname(os.path.abspath(arguments.ligands))

    return arguments


if __name__ == '__main__':
    args = _parse_args()
    logger.info(f'Passed arguments: {args}')

    protein = proteins.get_proteins()[args.protein]
    ligands_csv = pd.read_csv(args.ligands)
    csv_size, _ = ligands_csv.shape
    directory = args.output_dir
    output_csv = os.path.join(directory, args.csv_name)

    for i, row in ligands_csv.iterrows():
        if i + 1 < args.start_index:
            logger.info(f'Ignoring row {i}')
            continue

        smiles = row[args.smiles_column]
        logger.info(f'Docking: {smiles} ({i + 1}/{csv_size})')

        try:
            scores = protein.dock_smiles_to_protein(
                smiles,
                output_path=os.path.join(directory, f'{i}.mol2'),
                atom_terms_path=os.path.join(directory, f'{i}.ita'),
            )
            scores['mol_number'] = i
            scores['SMILES'] = smiles
            df = pd.DataFrame.from_dict(scores, orient='index').T

            if not os.path.isfile(output_csv):
                df.to_csv(output_csv, header=True, index=False)
            else:
                df.to_csv(output_csv, mode='a', header=False, index=False)
        except Exception as e:
            logger.error(f'Failed docking: {smiles} ({i + 1}/{csv_size})', exc_info=e)
