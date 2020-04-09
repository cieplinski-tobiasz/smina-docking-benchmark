import subprocess
from typing import Optional

from rdkit import Chem

import docking_benchmark.utils.chemistry as chemistry


def to_rdkit_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Converts a SMILES string to a RDKit molecule.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        RDKit Mol, None if the SMILES string is invalid
    """
    mol = Chem.MolFromSmiles(smiles)

    #  Sanitization check (detects invalid valence)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None

    return mol


def to_mol2_file(smiles: str, output_filename: str, seed: int = 0, silent: bool = False) -> Optional[str]:
    """
    Converts a SMILES string to mol2 file.

    Args:
        smiles: SMILES string of the molecule
        output_filename: Path of the file the converted molecule will be saved to
        seed: Seed used during optimizing and embedding molecule
        silent: If False, exception is thrown when conversion fails. Otherwise, None is returned.

    Returns:
        Output filename if the conversion succeeds, else None if silent is True.
        The returned filename is the same filename that was passed as an argument.

    Raises:
        RuntimeError: If conversion fails and silent is False.
    """
    try:
        molecule = to_rdkit_mol(smiles)

        if molecule is None:
            raise ValueError(f'Failed to convert {smiles} to RDKit mol')

        molecule = chemistry.embed_rdkit_molecule(molecule, seed)
        chemistry.optimize_rdkit_molecule(molecule)
        Chem.MolToMolFile(molecule, output_filename)

        command = f'obabel -imol {output_filename} -omol2 -O {output_filename}'
        openbabel_return_code = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL).returncode

        if openbabel_return_code != 0:
            raise ValueError(f'Failed to convert {smiles} to .mol2')
    except Exception as e:
        if silent:
            return None
        else:
            raise ValueError(e)

    return output_filename
