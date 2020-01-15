import pandas as pd


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

    class Builder:
        def __init__(self):
            self.total_samples = 0
            self._molecule_attributes = {}

        def append(self, smiles: str, **attributes):
            if type(smiles) is not str:
                raise TypeError('Expecting SMILES to be a str')

            smiles = smiles.strip()

            if not smiles:
                raise ValueError('SMILES is empty')

            self._molecule_attributes[smiles] = attributes

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
