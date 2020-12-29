"""
Code wrapping smina docking

Note: it is recommended to use a larger docking pocket. Otherwise many ligands fail to dock.
"""

import logging
import os
import subprocess
import tempfile
from typing import Union, List, Tuple

import numpy as np

import docking_benchmark.docking.smina.parsing
import docking_benchmark.utils.babel.smiles as convert_smiles

logger = logging.getLogger(__name__)


def min_aggregator(scores):
    return min(scores, key=lambda mode_score: mode_score['docking_score'])


def top_n_aggregator(n: int, score_component: str = 'docking_score', reverse: bool = False):
    if n <= 0:
        raise ValueError('n should be greater than zero')

    def aggregator(scores):
        if not scores:
            return scores

        top_scores = sorted(scores, key=lambda mode_score: mode_score[score_component], reverse=reverse)[:n]

        if not all(score.keys() == top_scores[0].keys() for score in top_scores):
            raise ValueError('one of the modes does not contain common component')

        aggregated = {}
        for key in top_scores[0].keys():
            for mode in top_scores:
                if key in aggregated:
                    aggregated[key] += mode[key]
                else:
                    aggregated[key] = mode[key]

            aggregated[key] /= len(top_scores)
        return aggregated

    return aggregator


def _exec_subprocess(command: List[str], timeout: int = None) -> List[str]:
    cmd = ' '.join([str(entry) for entry in command])

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=timeout)
        out, err, return_code = str(result.stdout, 'utf-8').split('\n'), str(result.stderr, 'utf-8'), result.returncode

        if return_code != 0:
            logger.error('Docking failed with command "' + cmd + '", stderr: ' + err)
            raise ValueError('Docking failed')

        return out
    except subprocess.TimeoutExpired:
        logger.error('Docking failed with command ' + cmd)
        raise ValueError('Docking timeout')


def dock_to_mol2(smiles: str, receptor_path: str, *, output_path, pocket_center: Union[List, np.array, Tuple],
                 pocket_range: Union[int, List[int], np.array, Tuple[int]] = 25, exhaustiveness: int = 16,
                 seed: int = 0, timeout: int = 600, n_cpu: int = 8, atom_terms_path=None,
                 scoring='vinardo') -> float:
    """
    Docks the passed molecule and saves the docked ligand in .mol2 file.

    Args:
        smiles: Ligand in SMILES format.
        receptor_path: Path to receptor in pdbqt format.
        pocket_center: x, y, z coordinates of docking center.
        pocket_range: Size of the docking pocket.
        exhaustiveness: Best practice is to bump to 16 (as compared to Vina 8s)
        seed: Random seed used by SMINA.
        timeout: Maximum docking time in seconds.
        n_cpu: Number of threads used by SMINA.
        output_path: Path the docked ligand will be saved to.
        atom_terms_path: Path the atom interactions will be saved to.
        scoring: Name of builtin SMINA scoring function.

    Returns:
        output_path
    """
    if type(pocket_range) is int:
        pocket_range = [pocket_range] * 3

    with tempfile.NamedTemporaryFile(suffix='.mol2') as ligand:
        convert_smiles.to_mol2_file(smiles, ligand.name)

        cmd = [
            'smina',
            '--receptor', receptor_path,
            '--ligand', ligand.name,
            '--seed', seed,
            '--cpu', n_cpu,
            '--center_x', pocket_center[0],
            '--center_y', pocket_center[1],
            '--center_z', pocket_center[2],
            '--size_x', pocket_range[0],
            '--size_y', pocket_range[1],
            '--size_z', pocket_range[2],
            '--exhaustiveness', exhaustiveness,
            '--out', os.path.abspath(output_path),
            '--scoring', scoring,
        ]

        if atom_terms_path is not None:
            cmd += ['--atom_terms', atom_terms_path]

        _exec_subprocess(cmd, timeout=timeout)

        return output_path


def dock_smiles(smiles: str, receptor_path, *, output_path=None, pocket_center: Union[List, np.array, Tuple],
                pocket_range: Union[int, List[int], np.array, Tuple[int]] = 25, exhaustiveness: int = 16,
                seed: int = 0, timeout: int = 600, n_cpu: int = 8, atom_terms_path=None,
                aggregator=top_n_aggregator(5)) -> dict:
    with tempfile.NamedTemporaryFile(suffix='.mol2') as temp_output_path:
        output_path = output_path if output_path is not None else temp_output_path.name
        dock_to_mol2(
            smiles, receptor_path,
            output_path=output_path, pocket_center=pocket_center, pocket_range=pocket_range,
            exhaustiveness=exhaustiveness, seed=seed, timeout=timeout, n_cpu=n_cpu,
            atom_terms_path=atom_terms_path)
        scores = score_only(output_path, receptor_path, timeout=timeout)
        return aggregator(scores)


def score_only(mol2_path, receptor, timeout: int = 600, scoring='vinardo') -> dict:
    """Evaluates the docked molecule.

    Args:
        mol2_path: Path to docked molecule in .mol2 format.
        receptor: Path to receptor in .pdbqt format.
        timeout: Maximum scoring time in seconds.
        scoring: Name of builtin SMINA scoring function.

    Returns:
        dict: Dictionary of dictionaries containing score values for given mode.
              Mode is a particular docking position of ligand.
    """
    cmd = [
        'smina',
        '--scoring', scoring,
        '-l', os.path.abspath(mol2_path),
        '--score_only',
        '-r', os.path.abspath(receptor),
    ]

    stdout = _exec_subprocess(cmd, timeout=timeout)

    scores = docking_benchmark.docking.smina.parsing.parse_score_only(stdout)

    for mode_score in scores:
        mode_score['docking_score'] = mode_score.pop('affinity')
        mode_score.update(mode_score.pop('pre_weighting_terms'))

    return scores
