"""
Some functions adapted from guacamol library
https://github.com/benevolentAI/guacamol
"""

from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from docking_benchmark.utils.logging import setup_and_get_logger

logger = setup_and_get_logger(name=__name__)


def embed_rdkit_molecule(molecule: Chem.Mol, seed: int, silent: bool = True) -> Optional[Chem.Mol]:
    """
    Embeds RDKit molecule in place.

    The function tries to embed the molecule using random coordinates firstly.
    If this fails the molecule is embedded with not random coordinates.
    The molecule passed as an argument is left intact if both trials fail.

    Args:
        molecule: RDKit molecule
        seed: Seed used for embedding
        silent: If False, exception is thrown when embedding fails. Otherwise, None is returned.

    Returns:
        Embedded RDKit molecule if embedding succeeds.
        Otherwise, if silent is True, returns None.
        The returned molecule is the same molecule that was passed as an argument.

    Raises:
        RuntimeError: If embedding fails and silent is False.
    """
    try:
        molecule = Chem.AddHs(molecule)
        conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=True, ignoreSmoothingFailures=True, randomSeed=seed)

        if conf_id == -1:
            conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=False, ignoreSmoothingFailures=True,
                                            randomSeed=seed)

        if conf_id == -1:
            raise ValueError('Embedding failure')
    except Exception as e:
        if silent:
            return None
        else:
            raise ValueError(e)

    return molecule


def optimize_rdkit_molecule(molecule: Chem.Mol, silent: bool = False) -> Optional[Chem.Mol]:
    """
    Optimizes the structure of RDKit molecule in place.

    The function tries a number of maxIters parameters.
    The molecule passed as an argument is left intact
    if none of the optimization trials succeed.

    Args:
        molecule: RDKit molecule
        silent: If False, exception is thrown when optimization fails. Otherwise, None is returned.

    Returns:
        Embedded RDKit molecule if optimization succeeds.
        Otherwise, if silent is True, returns None.
        The returned molecule is the same molecule that was passed as an argument.

    Raises:
        RuntimeError: If optimization fails and silent is False.
    """
    try:
        for max_iterations in [200, 2000, 20000, 200000]:
            if AllChem.UFFOptimizeMolecule(molecule, maxIters=max_iterations) == 0:
                break
        else:
            raise ValueError('Structure optimization failure')
    except Exception as e:
        if silent:
            return None
        else:
            raise ValueError(e)

    return molecule


def is_valid(smiles: str):
    """
    Verifies whether a SMILES string corresponds to a valid molecule.

    Args:
        smiles: SMILES string

    Returns:
        True if the SMILES strings corresponds to a valid, non-empty molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    return smiles != '' and mol is not None and mol.GetNumAtoms() > 0


def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters) if mol is not None else None
    except Exception:
        return None


def calculate_similarity(smiles1, smiles2):
    fp1 = get_fingerprints(get_mols([smiles1]))
    fp2 = get_fingerprints(get_mols([smiles2]))
    return DataStructs.TanimotoSimilarity(fp1[0], fp2[0])


def calculate_internal_pairwise_diversities(smiles_list) -> np.array:
    """
    Computes the pairwise similarities of the provided list of smiles against itself.

    Returns:
        Symmetric matrix of pairwise similarities. Diagonal is set to zero.
    """
    if len(smiles_list) > 10000:
        logger.warning(f'Calculating internal similarity on large set of '
                       f'SMILES strings ({len(smiles_list)})')

    mols = get_mols(smiles_list)
    fps = get_fingerprints(mols)
    nfps = len(fps)

    similarities = np.zeros((nfps, nfps))

    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarities[i, :i] = [1 - sim for sim in sims]
        similarities[:i, i] = [1 - sim for sim in sims]

    return similarities


def calculate_pairwise_similarities(smiles_list1, smiles_list2) -> np.array:
    """
    Computes the pairwise ECFP4 tanimoto similarity of the two smiles containers.

    Returns:
        Pairwise similarity matrix as np.array
    """
    if len(smiles_list1) > 10000 or len(smiles_list2) > 10000:
        logger.warning(f'Calculating similarity between large sets of '
                       f'SMILES strings ({len(smiles_list1)} x {len(smiles_list2)})')

    mols1 = get_mols(smiles_list1)
    fps1 = get_fingerprints(mols1)

    mols2 = get_mols(smiles_list2)
    fps2 = get_fingerprints(mols2)

    similarities = []

    for fp1 in fps1:
        sims = DataStructs.BulkTanimotoSimilarity(fp1, fps2)

        similarities.append(sims)

    similarities = np.array(similarities)

    return similarities


def get_fingerprints_from_smileslist(smiles_list, length: int = 4096):
    """
    Converts the provided smiles into ECFP4 bitvectors.

    Args:
        smiles_list: list of SMILES strings
        length: length of the bitvector

    Returns: ECFP4 bitvectors.

    """
    return get_fingerprints(get_mols(smiles_list), length=length)


def get_fingerprints(mols, radius=2, length=4096):
    """
    Converts molecules to ECFP bitvectors.

    Args:
        mols: RDKit molecules
        radius: ECFP fingerprint radius
        length: number of bits

    Returns: a list of fingerprints
    """
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, length) for m in mols]


def get_mols(smiles_list):
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
            if mol is not None:
                yield mol
        except Exception as e:
            logger.warning(e)
