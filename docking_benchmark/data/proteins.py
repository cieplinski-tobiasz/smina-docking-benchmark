import json
import os
from typing import List

import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection

import docking_benchmark.data.directories
from docking_benchmark.docking.smina.docking import dock_smiles

PROTEIN_FORMATS = (
    '.pdb',
    '.pdbqt',
)


class Datasets:
    """Container for datasets for a protein.

    Provides methods for accessing datasets for a protein.
    The datasets are loaded lazily.

    Attributes:
        protein (Protein): Protein that the datasets are for.
    """
    _DEFAULT_SMILES_COLUMN = 'SMILES'
    _DEFAULT_SCORE_COLUMN = 'DOCKING_SCORE'

    def __init__(self, protein):
        """Creates dataset container for given protein.

        Args:
            protein (Protein): Protein that datasets are loaded for.
        """
        self.protein = protein
        self._datasets = protein.metadata.get('datasets', dict())

    def __getitem__(self, dataset_name: str):
        """Loads the dataset with given name to memory.

        Args:
            dataset_name (str): Name of the dataset to load.

        Returns:
            tuple[list[str], np.array]: SMILES and score tuple.

        Raises:
            KeyError: If dataset with given name does not exist.
        """
        if dataset_name not in self._datasets:
            raise KeyError(f'No dataset named {dataset_name} for protein {self.protein.name}')

        dataset = self._datasets[dataset_name]
        csv = pd.read_csv(os.path.join(self.protein.directory, dataset['path']))
        smiles_column = dataset.get('smiles_column', self._DEFAULT_SMILES_COLUMN)
        score_column = dataset.get('score_column', self._DEFAULT_SCORE_COLUMN)
        return csv[smiles_column].tolist(), csv[score_column].to_numpy()

    def with_train_test_split(self, dataset_name: str, *,
                              test_size: float,
                              random_state: int = 0,
                              stratify_via_column: str = None):
        """Loads the dataset with given name to memory
        and splits the dataset into train and test datasets.

        Args:
            dataset_name (str): Name of the dataset to load.
            test_size (float): Fraction of test size.
            random_state (int): Seed used in splitting.
            stratify_via_column (str): Which column use for stratify.

        Returns:
            tuple[list[str], list[str], np.array, np.array]: Train SMILES, test SMILES, train score, test score

        Raises:
            KeyError: If dataset with given name does not exist.
        """
        if dataset_name not in self._datasets:
            raise KeyError(f'No dataset named {dataset_name} for protein {self.protein.name}')

        dataset = self._datasets[dataset_name]
        csv = pd.read_csv(os.path.join(self.protein.directory, dataset['path']))
        smiles_column = dataset.get('smiles_column', self._DEFAULT_SMILES_COLUMN)
        score_column = dataset.get('score_column', self._DEFAULT_SCORE_COLUMN)
        stratify = csv[stratify_via_column] if stratify_via_column is not None else None
        x_train, x_test = sklearn.model_selection.train_test_split(
            csv,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        return x_train[smiles_column].tolist(), x_test[smiles_column].tolist(), \
               x_train[score_column].to_numpy(), x_test[score_column].to_numpy()

    def with_linear_combination_score(self, dataset_name, **component_weights):
        """Loads the dataset with score as a linear combination of given components.

        The method loads the dataset and creates score
        according to the components and the corresponding weights passed.

        For example:
        `linear_combination('test', A=1.0, B=2.0)`
        will create a dataset with SMILES from `test` dataset
        and with the score equal to 1.0 * A + 2.0 * B
        for each SMILES.

        Args:
            dataset_name (str): Name of the dataset to load.
            **component_weights (float): Component weights used in linear combination.

        Returns:
            tuple[list[str], np.array]: SMILES and score tuple.

        Raises:
            ValueError: If dataset with dataset_name does not exist,
                        no component_weights are passed
                        or component with given name is not present
                        in the dataset.
        """
        dataset = self._datasets[dataset_name]
        csv = pd.read_csv(os.path.join(self.protein.directory, dataset['path']))

        if len(component_weights) == 0:
            raise ValueError('No components\' weights passed')

        for component, _ in component_weights.items():
            if component not in csv.columns:
                raise ValueError('No component named ' + component + ' in ' + dataset_name + ' dataset')

        combination = np.zeros(csv.shape[0])

        for component, weight in component_weights.items():
            combination += csv[component].to_numpy() * weight

        smiles_column = dataset.get('smiles_column', self._DEFAULT_SMILES_COLUMN)

        return csv[smiles_column], combination


class Protein:
    """Container for protein-related data.

    The class provides required protein information
    for SMINA docking software.

    Single protein's data must be put into one directory.
    This directory *must* contain a `metadata.json` file
    and a *single* file with .pdb/.pdbqt extension.

    `metadata.json` stores basic information about protein,
    such as its pocket center coordinates and available datasets
    associated with it.

    See the example below for `metadata.json` file structure.

    Example `metadata.json` file:
    ```json
    {
        "pocket_center": [0.0, 0.0, 0.0],
        "datasets": {
            "first_dataset": {
                "path": "datasets/first.csv",
                "smiles_column": "smi",
                "score_column": "activity"
            },
            "second_dataset": {
                "path": "datasets/second.csv"
            }
        }
    }
    ```

    `pocket_center` field *is required*.
    It must be a list of three numbers - coordinates used for docking.

    `datasets` key is optional.
    Each entry in the `datasets` dictionary *must* have `path` field present.
    `smiles_column` and `score_column` are optional.

    Attributes:
        name (str): Name of the protein, e.g. '5ht1b'.
        directory: Path to the directory containing protein related files.
        path: Path to the protein file.
        metadata (dict): Dictionary with parsed metadata.json file (see above for description).
        datasets (Datasets): Available datasets for the protein.
    """
    _METADATA_FILENAME = 'metadata.json'
    _POCKET_CENTER_LENGTH = 3

    def __init__(self, name: str, directory):
        """Creates a Protein from given directory.

        See the class-level docs for required directory structure.

        Args:
            name (str): Name of the protein.
            directory: Path to the directory containing protein related files.

        Raises:
            ValueError: If directory does not exist,
                        does not contain metadata.json file,
                        contains invalid metadata.json file
                        or does not contain a single .pdb/.pdbqt file.
        """
        if not os.path.exists(directory):
            raise ValueError(f'Directory {directory} does not exist')

        self.name = name
        self.directory = directory
        self.path = self._load_protein_file_path()
        self.metadata = self._load_metadata()
        self.datasets = Datasets(self)

    def _load_protein_file_path(self):
        """Returns protein file path from provided directory.

        The method looks for .pdb/.pdbqt file in provided directory.
        No validity check of the file is performed.

        Raises:
            ValueError: If no, or more than one .pdb/.pdbqt file
                        is present in the directory.

        Returns:
            Path to .pdb/.pdbqt file.
        """
        protein_files = [entry for entry in os.listdir(self.directory)
                         if any(entry.endswith(fmt) for fmt in PROTEIN_FORMATS)]

        if len(protein_files) != 1:
            msg_start = 'No protein files' if len(protein_files) == 0 else 'More than one protein file'
            raise ValueError(msg_start + ' inside ' + self.name + ' protein data directory')

        return os.path.join(self.directory, protein_files[0])

    def _load_metadata(self) -> dict:
        """Loads and parses `metadata.json` file from provided directory.

        Raises:
            ValueError: If `metadata.json file` does not exist or is invalid.

        Returns:
            dict: Parsed `metadata.json` file.
        """
        metadata_path = os.path.join(self.directory, self.__class__._METADATA_FILENAME)

        if not os.path.exists(metadata_path):
            raise ValueError('No metadata.json file for ' + self.name)

        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)

            if 'pocket_center' not in metadata:
                raise ValueError('Protein ' + self.name +
                                 ' metadata must contain pocket_center key')

            if len(metadata['pocket_center']) != self._POCKET_CENTER_LENGTH:
                raise ValueError('Pocket center for ' + self.name +
                                 'must be a list of three floats')

            for coordinate in metadata['pocket_center']:
                if type(coordinate) is not float:
                    raise ValueError('Pocket center for ' + self.name +
                                     'must be a list of three floats')

            return metadata

    @property
    def pocket_center(self) -> List[float]:
        """list[float]: Pocket center coordinates used for docking"""
        return self.metadata['pocket_center']

    def dock_smiles_to_protein(self, smiles, **docking_kwargs):
        if 'pocket_center' in docking_kwargs:
            raise ValueError('Do not pass pocket_center as it is provided by the protein.')

        if 'receptor_path' in docking_kwargs:
            raise ValueError('Do not pass receptor_path as it is provided by the protein.')

        return dock_smiles(smiles, self.path, pocket_center=self.pocket_center, **docking_kwargs)


def get_proteins():
    return {
        protein_dir.lower(): Protein(
            protein_dir.lower(),
            os.path.join(docking_benchmark.data.directories.PROTEINS, protein_dir.lower())
        ) for protein_dir in os.listdir(docking_benchmark.data.directories.PROTEINS)}
