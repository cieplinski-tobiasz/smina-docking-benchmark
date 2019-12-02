from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem


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
    molecule = Chem.AddHs(molecule)
    conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=True, ignoreSmoothingFailures=True, randomSeed=seed)

    if conf_id == -1:
        conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=False, ignoreSmoothingFailures=True, randomSeed=seed)

    if conf_id == -1:
        if silent:
            return None
        else:
            raise ValueError('Embedding failure')

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
    for max_iterations in [200, 2000, 20000, 200000]:
        if AllChem.UFFOptimizeMolecule(molecule, maxIters=max_iterations) == 0:
            break
    else:
        if silent:
            return None
        else:
            raise ValueError('Structure optimization failure')

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

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None
