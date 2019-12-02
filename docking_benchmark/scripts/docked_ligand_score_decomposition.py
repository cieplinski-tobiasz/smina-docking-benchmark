"""
Script that creates .csv file with docking score components' values.
It requires docked ligands in .mol2 format to be in one directory
with file names resembling row numbers of a .csv file containing SMILES representations
of ligands that were docked.
"""

import argparse
import os

import pandas as pd

import docking_benchmark.data.proteins as proteins
import docking_benchmark.docking.smina.docking as smina


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--docked-ligands-directory', required=True)
    parser.add_argument('-c', '--csv', required=True)
    parser.add_argument('-p', '--protein', default='5ht1b')
    parser.add_argument('--smiles-column', default='SMILES')
    parser.add_argument('-f', '--filter-results', default='minimum_affinity')
    parser.add_argument('-o', '--output', required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    ligands_csv = pd.read_csv(args.csv)
    protein = proteins.get_proteins()[args.protein]
    decomposed_ligands = []

    for entry in os.scandir(args.docked_ligands_directory):
        if not entry.name.endswith('.mol2'):
            continue

        scores = smina.score_only(entry.path, protein.path, filter_results=args.filter_results)
        scores.update(scores['pre_weighting_terms'])
        del scores['pre_weighting_terms']
        smiles_number, _ = os.path.splitext(os.path.basename(entry.path))
        smiles = ligands_csv.iloc[int(smiles_number)][args.smiles_column]
        scores['smiles'] = smiles
        decomposed_ligands.append(scores)

    df = pd.DataFrame(decomposed_ligands)
    df.to_csv(args.output, index=False)
