import logging
import tempfile

import numpy as np

import docking_benchmark.models.gvae.zinc_grammar as zinc_grammar
from docking_benchmark.docking.predicted_docking_functions import MLPPredictedDockingScore
from docking_benchmark.models.gvae.model_zinc import MoleculeVAE
from docking_benchmark.models.gvae.molecule_vae import ZincGrammarModel
from docking_benchmark.utils.chemistry import is_valid, canonicalize

logger = logging.getLogger(__name__)


class GVAEGradientGenerator:
    def __init__(self, pretrained_path, dataset,
                 max_len=277, latent=56, batch_size=256,
                 fine_tune_epochs=5, fine_tune_test_size=0.1,
                 fit_epochs=30, descent_iterations=50, descent_lr=0.01,
                 mode='minimize'):
        gvae = MoleculeVAE()
        gvae.load(zinc_grammar.gram.split('\n'), pretrained_path, latent_rep_size=latent, max_length=max_len)

        self.latent = latent
        self.gvae = ZincGrammarModel(gvae)
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_test_size = fine_tune_test_size
        self.batch_size = batch_size
        self.fit_epochs = fit_epochs
        self.descent_iterations = descent_iterations
        self.descent_lr = descent_lr
        self.max_len = max_len
        self.dataset = self.get_filtered_dataset(dataset)

        if mode == 'minimize':
            self.descent_lr *= -1
        elif mode != 'maximize':
            raise ValueError(f'Unknown mode {mode}')

        self.fine_tuned = False
        self.mlp = None

    def _smiles_to_latent_fn(self):
        def smiles_to_latent(smiles):
            nonlocal self

            if type(smiles) is str:
                smiles = [smiles]

            return self.gvae.encode(smiles)

        return smiles_to_latent

    def _fine_tune(self):
        if self.fine_tuned:
            return

        one_hots = self.gvae.to_one_hots(self.dataset[0])
        self.gvae.vae.autoencoder.fit(one_hots, one_hots, epochs=self.fine_tune_epochs,
                                      validation_split=self.fine_tune_test_size)

        with tempfile.NamedTemporaryFile() as tmp:
            self.gvae.vae.autoencoder.save(tmp.name)
            tuned = MoleculeVAE()
            tuned.load(zinc_grammar.gram.split('\n'), tmp.name, latent_rep_size=self.latent, max_length=self.max_len)
            self.gvae = ZincGrammarModel(tuned)

    def _train_mlp(self):
        if self.mlp is None:
            self.mlp = MLPPredictedDockingScore(
                self.dataset, input_dim=self.latent, to_latent_fn=self._smiles_to_latent_fn()
            )

    def random_gauss_rmse(self, smiles_docking_score_fn, test_size=1):
        rmse_samples = []

        while len(rmse_samples) < test_size:
            latents = np.random.normal(size=(self.batch_size, self.latent))
            smiles = [canonicalize(smi) for smi in self.gvae.decode(latents)]

            for i, smi in enumerate(smiles):
                if smi is not None and is_valid(smi):
                    rmse_samples.append((smi, self.mlp.latent_score(latents[i].reshape(1, -1))))

        rmse_samples = [(smi, ls, smiles_docking_score_fn(smi)) for smi, ls in rmse_samples]
        rmse_samples = rmse_samples[:test_size]

        return (sum((ls - ds) ** 2 for _, ls, ds in rmse_samples) / test_size) ** (1 / 2)

    def generate_optimized_molecules(self, number_molecules: int, smiles_docking_score_fn=None, rmse_test: int = 100):
        self._fine_tune()
        self._train_mlp()

        if smiles_docking_score_fn is not None and rmse_test is not None:
            gauss_rmse = self.random_gauss_rmse(smiles_docking_score_fn, rmse_test)
            logger.info(f'Gauss RMSE: {gauss_rmse:.2f}')

        valid_samples = []
        total_sampled = 0

        while len(valid_samples) < number_molecules:
            latents = np.random.normal(size=(self.batch_size, self.latent))

            for _ in range(self.descent_iterations):
                latents += self.mlp.gradient(latents) * self.descent_lr

            smiles = [canonicalize(smi) for smi in self.gvae.decode(latents)]

            for i, smi in enumerate(smiles):
                if smi is not None and is_valid(smi):
                    latent_score = self.mlp.latent_score(latents[i].reshape(1, -1))
                    valid_samples.append((smi, latent_score))

            total_sampled += self.batch_size

            logger.info(f'Sampled {len(valid_samples) / number_molecules}')

        if len(valid_samples) > number_molecules:
            valid_samples = valid_samples[:number_molecules]

        return valid_samples

    def get_filtered_dataset(self, dataset):
        smiles, scores = dataset

        _, valid_indices = self.gvae.to_one_hots(smiles, with_valid_indices=True)

        return [smiles[i] for i in valid_indices], scores[valid_indices]
