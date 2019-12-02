"""
Code wrapping smina docking

Note: recommended way to use is to pass mol2 ligands, not pdbqt. Parsing files to pdbqt can be tricky and buggy.
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


def _exec_subprocess(command: str, timeout: int = None) -> Tuple[List[str], List[str], int]:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=timeout)
        out, err, return_code = str(result.stdout, 'utf-8').split('\n'), str(result.stderr, 'utf-8').split(
            '\n'), result.returncode
        return out, err, return_code
    except subprocess.TimeoutExpired:
        raise ValueError('Docking timeout')


def dock_smiles(smiles: str, receptor: str, pocket_center: Union[List, np.array, Tuple] = None,
                pocket_range: Union[int, List[int], np.array, Tuple[int]] = 25, exhaustiveness: int = 16,
                seed: int = 0, timeout: int = 600, n_cpu: int = 8, output_path=None, atom_terms_path=None,
                custom_scoring: str = None, additional_args: list = None) -> float:
    """
    Docks the passed molecule

    Args:
        smiles: Ligand in SMILES format
        receptor: Path to receptor in pdbqt or mol2 format
        pocket_center: x, y, z coordinates of docking center
        pocket_range: How far from the center are we trying to dock. If list is passed it must be of size 3.
        exhaustiveness: Best practice is to bump to 16 (as compared to Vina 8s)
        seed: Random seed passed to the smina simulator
        timeout: Maximum waiting time in seconds
        n_cpu: How many cpus to use
        output_path: Docked ligand will be saved to this path
        atom_terms_path: Atom interactions will be saved to this path
        custom_scoring: Custom scoring function
        additional_args: Arguments passed to smina
                         in ['--option', 'value', ...] format

    Returns:
        Minimal docking score of the ligand
    """
    if type(pocket_range) is int:
        pocket_range = [pocket_range] * 3

    with tempfile.NamedTemporaryFile(suffix='.mol2') as ligand, tempfile.NamedTemporaryFile(mode='w+') as function_file:
        convert_smiles.to_mol2_file(smiles, ligand.name)

        cmd = [
            'smina',
            '--receptor', receptor,
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
        ]

        if custom_scoring is not None:
            function_file.write(custom_scoring)
            function_file.flush()
            cmd += ['--custom_scoring', function_file.name]

        if output_path is not None:
            cmd += ['--out', output_path]

        if atom_terms_path is not None:
            cmd += ['--atom_terms', atom_terms_path]

        if additional_args is not None:
            cmd += additional_args

        cmd = ' '.join([str(entry) for entry in cmd])
        stdout, stderr, return_code = _exec_subprocess(cmd, timeout=timeout)

        if return_code != 0:
            logger.error(stderr)
            raise ValueError(f'Failed to dock {smiles} to {os.path.basename(receptor)}')

        scores = docking_benchmark.docking.smina.parsing.parse_docking_score(stdout)

        return min(scores)


def score_only(mol2_path, receptor, timeout: int = 600, filter_results=None):
    cmd = ' '.join([
        'smina',
        '-l',
        os.path.abspath(mol2_path),
        '--score_only',
        '-r',
        os.path.abspath(receptor)
    ])

    stdout, stderr, return_code = _exec_subprocess(cmd, timeout=timeout)

    if return_code != 0:
        logger.error(stderr)
        raise ValueError(f'Failed to score {mol2_path} to {os.path.basename(receptor)}')

    scores = docking_benchmark.docking.smina.parsing.parse_score_only(stdout)

    if filter_results == 'minimum_affinity':
        return min(scores, key=lambda d: d['affinity'])
    if filter_results == 'maximum_affinity':
        return max(scores, key=lambda d: d['affinity'])
    else:
        return scores
