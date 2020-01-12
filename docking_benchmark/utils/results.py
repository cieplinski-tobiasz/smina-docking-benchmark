import pickle

import numpy as np


def assert_or_raise(condition, *, error_msg, error_cls=ValueError):
    if not condition:
        raise error_cls(error_msg)


class OptimizedMolecules:
    def __init__(self, smiles, scores, *, predicted_scores=None, latent_vectors=None, total_samples=None):
        self.smiles = smiles
        self.scores = scores
        self.predicted_scores = predicted_scores
        self.latent_vectors = latent_vectors
        self._total_samples = total_samples
        self.metrics = {}
        self._calculate_metrics()

    def _calculate_metrics(self):
        if self.predicted_scores is not None:
            self.metrics['rmse'] = ((self.scores - self.predicted_scores) ** 2).mean(axis=None) ** (1 / 2)

        if self._total_samples is not None:
            self.metrics['validity'] = len(self.smiles) / self._total_samples

    def metric(self, name):
        if name.lower() in self.metrics:
            return self.metrics[name.lower()]

        raise ValueError('No such metric')

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)


class OptimizedMoleculesBuilder:

    def __init__(self) -> None:
        self._molecules = []
        self._scores = []
        self._latent_vectors = []
        self._predicted_scores = []
        self._total_samples = None

    def append_molecule(self, smiles: str, score: float, latent_vector=None, predicted_score: float = None):
        self._molecules.append(smiles)
        self._scores.append(score)

        if latent_vector is not None:
            self._latent_vectors.append(latent_vector)

        if predicted_score is not None:
            self._predicted_scores.append(predicted_score)

    def increment_sample_count(self, by):
        if self._total_samples is None:
            self._total_samples = 0

        self._total_samples += by

    @property
    def size(self):
        return len(self._molecules)

    def _validate(self):
        assert_or_raise(all(smi is not None for smi in self._molecules),
                        error_msg='None in generated SMILES')
        assert_or_raise(all(score is not None for score in self._scores),
                        error_msg='None in scores')
        assert_or_raise(len(self._molecules) == len(self._scores),
                        error_msg='Molecule and Score length mismatch')

        if self._predicted_scores:
            assert_or_raise(len(self._predicted_scores) == len(self._scores),
                            error_msg='Each molecule should have a predicted score associated')

        if self._latent_vectors:
            assert_or_raise(len(self._latent_vectors) == len(self._scores),
                            error_msg='Each molecule should have a latent vector associated')

    def build(self) -> OptimizedMolecules:
        self._validate()

        latents = np.concatenate(self._latent_vectors) if self._latent_vectors else None

        if latents is not None and len(self._molecules) == 1:
            latents = latents.reshape(1, -1)

        return OptimizedMolecules(
            self._molecules,
            np.array(self._scores),
            latent_vectors=latents,
            predicted_scores=np.array(self._predicted_scores) if self._predicted_scores else None,
            total_samples=self._total_samples
        )
