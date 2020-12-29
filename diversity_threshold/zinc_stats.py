import argparse
import os
from typing import Dict, Iterable, Tuple

import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm


def load_zinc(zinc_dir: str = 'zinc') -> pd.DataFrame:
    """Load a downloaded ZINC sample and return a dataframe with loaded compounds.

    Args:
        zinc_dir: path to the directory containing SMI files with compounds sampled from ZINC.

    Raises:
        FileNotFoundError: the path provided in zinc_dir does not exist.

    Returns:
        A dataframe containing ZINC compounds joined from all the input files.
    """
    if not os.path.exists(zinc_dir):
        raise FileNotFoundError('A sample of ZINC dataset should be downloaded to `zinc_dir`, '
                                'but this directory does not exist.')
    master_df = pd.DataFrame()
    for file_name in filter(lambda x: x.endswith('.smi'), os.listdir(zinc_dir)):
        with open(os.path.join(zinc_dir, file_name), 'r') as file:
            df = pd.read_csv(file, sep=' ')
        if len(df) > 1000:
            df = df.sample(n=1000)
        master_df = master_df.append(df, ignore_index=True)
    master_df = master_df.drop_duplicates('smiles')
    return master_df.drop_duplicates('zinc_id')


def get_compound_fingerprints(
        data_dir: str,
        target_names: Iterable[str] = ('5ht1b', '5ht2b', 'acm2', 'cyp2d6')
) -> Tuple[dict, dict]:
    """Gets ECFP fingerprints of compounds for each biological target.

    Args:
        data_dir: directory containing activity data towards specified biological targets; the compounds relevant
                  for the given target should be placed in `{data_dir}/{target_name}/datasets/{target_name}.csv`.
        target_names: names of biological targets that should be loaded.

    Returns:
        A pair (compound_dfs, fingerprints) where `compound_dfs` is a dictionary of dataframes containing compound SMILES
        representations and `fingerprints` is a dictionary of ECFP fingerprints.
    """
    compound_dfs = {}
    for prot_name in target_names:
        data_path = f'{data_dir}/{prot_name}/datasets/{prot_name}.csv'
        compound_dfs[prot_name] = pd.read_csv(data_path)

    fps = {}
    for prot_name in target_names:
        fps[prot_name] = []
        for i, row in tqdm(compound_dfs[prot_name].iterrows(), total=len(compound_dfs[prot_name])):
            smiles = row.SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps[prot_name].append(fp)
            else:
                fps[prot_name].append(None)
    return compound_dfs, fps


def get_zinc_fingerprints(zinc_df: pd.DataFrame) -> list:
    """Gets ECFP fingerprints of the input compounds.

    Args:
        zinc_df: a dataframe with ZINC sample compounds.

    Returns:
        A list of ECFP fingerprints.
    """
    fps = []
    for i, row in tqdm(zinc_df.iterrows(), total=len(zinc_df)):
        smiles = row.smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps.append(fp)
    return fps


def calculate_similarity(
        fps: dict,
        master_df: pd.DataFrame,
        data_df: Dict[str, pd.DataFrame],
        target_name: str
) -> list:
    """Calculates Tanimoto similarity between ZINC sample and activity data.

    Args:
        fps: a dictionary of ECFP fingerprints.
        master_df: a dataframe of ZINC compounds.
        data_df: a dictionary of dataframes of activity data.
        target_name: the name of biological target.

    Returns:
        A list of triples (distance, target SMILES, ZINC SMILES) sorted by the Tanimoto distance.
    """
    results = []
    for i, fp_i in enumerate(fps['master']):
        min_dist = 1.
        min_idx = -1
        for j, fp_j in enumerate(fps[target_name]):
            if not fp_j:
                continue
            dist = 1. - DataStructs.FingerprintSimilarity(fp_i, fp_j)
            if dist < min_dist:
                min_dist = dist
                min_idx = j

        results.append((min_dist, master_df.iloc[i].smiles, data_df[target_name].iloc[min_idx].SMILES))

        if i > 10000:
            break

    return sorted(results, key=lambda x: x[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zinc-dir', type=str,
                        help='path to the directory of sampled ZINC SMILES')
    parser.add_argument('--data-dir', type=str,
                        help='path to the activity data')
    parser.add_argument('--targets', nargs='+', default=('5ht1b', '5ht2b', 'acm2', 'cyp2d6'),
                        help='names of the activity targets')
    parser.add_argument('--percentile', type=int, default=95,
                        help='percentile of compound distances at which the threshold should be set')
    args = parser.parse_args()

    master_df = load_zinc(zinc_dir=args.zinc_dir)
    data_df, fps = get_compound_fingerprints(data_dir=args.data_dir, target_names=args.targets)
    fps['master'] = get_zinc_fingerprints(master_df)
    for target_name in args.targets:
        results = calculate_similarity(fps, master_df, data_df, target_name)
        threshold = results[int(len(results) * args.percentile / 100)]
        print(f'{args.percentile}th percentile threshold for `{target_name}` is {threshold[0]}')
