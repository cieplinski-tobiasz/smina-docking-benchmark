import json
import os

import pandas as pd

import docking_benchmark.data.directories

PROTEIN_FORMATS = (
    '.pdb',
    '.pdbqt',
)


class Datasets:
    _DEFAULT_SMILES_COLUMN = 'SMILES'
    _DEFAULT_SCORE_COLUMN = 'DOCKING_SCORE'

    def __init__(self, protein):
        self.protein = protein
        self._datasets = protein.metadata.get('datasets', list())

    def __getitem__(self, dataset_name):
        if dataset_name not in self._datasets:
            raise KeyError(f'No dataset named {dataset_name} for protein {self.protein.name}')

        dataset = self._datasets[dataset_name]
        csv = pd.read_csv(os.path.join(self.protein.directory, dataset['path']))
        smiles_column = dataset.get('smiles_column', self._DEFAULT_SMILES_COLUMN)
        score_column = dataset.get('score_column', self._DEFAULT_SCORE_COLUMN)
        return csv[smiles_column].tolist(), csv[score_column].to_numpy()

    @property
    def default(self):
        return self['default']


class Protein:
    _METADATA_FILENAME = 'metadata.json'

    def __init__(self, name, directory):
        self.name = name
        self.directory = directory
        self.path = self._load_protein_file_path()
        self.metadata = self._load_metadata()
        self.datasets = Datasets(self)

    def _load_protein_file_path(self):
        protein_files = [entry for entry in os.listdir(self.directory)
                         if any(entry.endswith(fmt) for fmt in PROTEIN_FORMATS)]

        if len(protein_files) != 1:
            msg_start = 'No protein files' if len(protein_files) == 0 else 'More than one protein file'
            raise ValueError(msg_start + ' inside ' + self.name + ' protein data directory')

        return os.path.join(self.directory, protein_files[0])

    def _load_metadata(self):
        with open(os.path.join(self.directory, self.__class__._METADATA_FILENAME)) as metadata_file:
            metadata = json.load(metadata_file)

            if 'pocket_center' not in metadata:
                raise ValueError('Protein ' + self.name + ' metadata must contain pocket_center key')

            return metadata

    @property
    def pocket_center(self):
        return self.metadata['pocket_center']


def get_proteins():
    return {
        protein_dir.lower(): Protein(
            protein_dir.lower(),
            os.path.join(docking_benchmark.data.directories.PROTEINS, protein_dir.lower())
        ) for protein_dir in os.listdir(docking_benchmark.data.directories.PROTEINS)}
