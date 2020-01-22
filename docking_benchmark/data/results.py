import math
import pickle
from typing import List

import numpy as np
import pandas as pd

from docking_benchmark.utils.chemistry import calculate_pairwise_similarities


class OptimizedMolecules:
    def __init__(self, molecules: pd.DataFrame, **precalculated_metrics):
        self.molecules = molecules
        self.metrics = precalculated_metrics

    def rmse(self, col1: str, col2: str):
        if col1 not in self.molecules.columns:
            raise ValueError('No column "' + col1 + '" in OptimizedMolecules')

        if col2 not in self.molecules.columns:
            raise ValueError('No column "' + col2 + '" in OptimizedMolecules')

        series1 = self.molecules[col1]
        series2 = self.molecules[col2]

        if series1.isnull().values.any() or series2.isnull().values.any():
            raise ValueError('Selected columns contain NaNs')

        squared_error = (series1 - series2) ** 2
        return squared_error.mean() ** (1 / 2)

    def get_first_n(self, n: int, *, by_column: str, sort_ascending: bool = True):
        if by_column not in self.molecules.columns:
            raise ValueError('No column "' + by_column + '" in OptimizedMolecules')

        if n <= 0:
            raise ValueError('n must be positive')

        return self.molecules.sort_values(by_column, ascending=sort_ascending)[:n]

    def get_first_fraction(self, fraction: float, *, by_column: str, sort_ascending: bool = True):
        if by_column not in self.molecules.columns:
            raise ValueError('No column "' + by_column + '" in OptimizedMolecules')

        if fraction <= 0 or fraction > 1:
            raise ValueError('fraction must be in range (0, 1]')

        n = math.ceil(fraction * self.molecules.shape[0])

        return self.molecules.sort_values(by_column, ascending=sort_ascending)[:n]

    def most_similar_tanimoto(self, to_smiles: List[str]):
        tanimoto_similarities = calculate_pairwise_similarities(
            self.molecules.index.tolist(),
            to_smiles
        )

        most_similar_indices = np.argmax(tanimoto_similarities, axis=1)
        max_tanimoto_similarities = np.array(
            [tanimoto_similarities[i, most_similar_indices[i]] for i in range(self.molecules.shape[0])])

        most_similars = pd.DataFrame(index=self.molecules.index.copy())
        most_similars['tanimoto_similarity'] = max_tanimoto_similarities
        most_similars['most_similar_smiles'] = [to_smiles[most_similar_indices[i]] for i in
                                                range(self.molecules.shape[0])]

        return most_similars

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def to_csv(self, path, index_label='SMILES', without_columns=None, **pd_kwargs):
        if without_columns is not None and 'columns' in pd_kwargs:
            raise ValueError('without_columns and columns cannot be used together')

        if without_columns is not None:
            pd_kwargs['columns'] = set(self.molecules.columns.to_list()) - set(without_columns)

        self.molecules.to_csv(path, index_label=index_label, **pd_kwargs)

    class Builder:
        def __init__(self):
            self.total_samples = 0
            self._molecule_attributes = {}

        def __contains__(self, smiles):
            return smiles in self._molecule_attributes

        def append(self, smiles: str, **attributes):
            if smiles in self._molecule_attributes:
                return False

            if type(smiles) is not str:
                raise TypeError('Expecting SMILES to be a str')

            smiles = smiles.strip()

            if not smiles:
                raise ValueError('SMILES is empty')

            self._molecule_attributes[smiles] = attributes
            return True

        @property
        def size(self):
            return len(self._molecule_attributes)

        def _precalculate_metrics(self):
            metrics = {}

            if self.total_samples > 0:
                metrics['validity'] = len(self._molecule_attributes) / self.total_samples

            return metrics

        def build(self):
            return OptimizedMolecules(
                pd.DataFrame.from_dict(self._molecule_attributes, orient='index'),
                **self._precalculate_metrics()
            )
