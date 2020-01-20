"""
Script that docks ligands passed in a .csv file,
stores the docked ligand in .mol2 format
and also stores the interaction statistics
in a text file with .ita extension.
"""

import argparse
import logging
import os.path

import pandas as pd

from docking_benchmark.data import proteins
from docking_benchmark.docking.smina import docking
from docking_benchmark.utils.logging import setup_and_get_logger

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ligands')
    parser.add_argument('-p', '--protein', default='5ht1b')
    parser.add_argument('--n-cpu', type=int, default=8)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-od', '--output-dir')
    parser.add_argument('-c', '--custom-scoring')
    parser.add_argument('-s', '--start-index', default=1, type=int)
    arguments = parser.parse_args()

    setup_and_get_logger(arguments.debug)

    if not arguments.output_dir:
        arguments.output_dir = os.path.dirname(os.path.abspath(arguments.ligands))

    return arguments


def _get_scoring_function(path):
    if path is not None:
        with open(args.custom_scoring) as function_file:
            return function_file.read()

    return None


if __name__ == '__main__':
    args = _parse_args()
    logger.info(f'Passed arguments: {args}')
    protein = proteins.get_proteins()[args.protein]
    csv = pd.read_csv(args.ligands)
    csv_size, _ = csv.shape
    directory = args.output_dir
    scoring_function = _get_scoring_function(args.custom_scoring)

    for i, row in csv.iterrows():
        if i + 1 < args.start_index:
            logger.info(f'Ignoring row {i}')
            continue

        smiles = row['SMILES']
        logger.info(f'Docking: {smiles} ({i + 1}/{csv_size})')

        try:
            docking.dock_to_mol2(smiles, protein.path, pocket_center=protein.pocket_center,
                                 output_path=os.path.join(directory, f'{i}.mol2'),
                                 atom_terms_path=os.path.join(directory, f'{i}.ita'))
        except Exception as e:
            logger.error(f'Failed docking: {smiles} ({i + 1}/{csv_size})')
            logger.exception(e)
