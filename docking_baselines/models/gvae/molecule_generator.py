import logging
import os.path
import tempfile

import numpy as np

import docking_baselines.models.gvae.zinc_grammar as zinc_grammar
from docking_baselines.models.gvae.model_zinc import MoleculeVAE
from docking_baselines.models.gvae.molecule_vae import ZincGrammarModel
from docking_baselines.models.predicted_docking_functions import MLPPredictedDockingScore
from docking_benchmark.data.results import OptimizedMolecules
from docking_benchmark.utils.chemistry import is_valid, canonicalize

logger = logging.getLogger(__name__)


class GVAEGradientGenerator:
    def __init__(self, pretrained_path, dataset,
                 max_len=277, latent=56, batch_size=256,
                 fine_tune_epochs=5, fine_tune_test_size=0.1,
                 mlp_fit_epochs=50, mlp_test_size=0.1,
                 fit_epochs=30, descent_iterations=50, descent_lr=0.01,
                 mode='minimize',
                 output_dir=None, docking_n_cpu=4):
        gvae = MoleculeVAE()
        gvae.load(zinc_grammar.gram.split('\n'), pretrained_path, latent_rep_size=latent, max_length=max_len)

        self.latent = latent
        self.gvae = ZincGrammarModel(gvae)
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_test_size = fine_tune_test_size
        self.mlp_fit_epochs = mlp_fit_epochs
        self.mlp_test_size = mlp_test_size
        self.output_dir = output_dir
        self.docking_n_cpu = docking_n_cpu
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
        if self.fine_tune_epochs <= 0 or self.fine_tuned:
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
                self.dataset, input_dim=self.latent,
                epochs=self.mlp_fit_epochs, test_fraction=self.mlp_test_size,
                to_latent_fn=self._smiles_to_latent_fn()
            )

    def random_gauss(self, smiles_docking_score_fn, size):
        results_builder = OptimizedMolecules.Builder()

        while results_builder.size < size:
            logger.info(f'Random sampled {results_builder.size} / {size}')
            latents = np.random.normal(size=(self.batch_size, self.latent))
            smiles = [canonicalize(smi) for smi in self.gvae.decode(latents)]

            for i, smi in enumerate(smiles):
                if smi is not None and is_valid(smi):
                    try:
                        if smi not in results_builder:
                            output_path = os.path.join(
                                self.output_dir,
                                f'{results_builder.size}.gauss.mol2') if self.output_dir is not None else None
                            docking_score = smiles_docking_score_fn(smi, output_path=output_path,
                                                                    n_cpu=self.docking_n_cpu)
                            results_builder.append(
                                smi,
                                latent_vector=latents[i],
                                predicted_score=self.mlp.latent_score(latents[i].reshape(1, -1)),
                                **docking_score
                            )
                        else:
                            logger.info('Generated SMILES %s already present in OptimizedMoleculesBuilder', smi)
                    except (ValueError, RuntimeError):
                        logger.error('Docking failed')

                    if results_builder.size >= size:
                        logger.info('Random sampling finished')
                        break

        return results_builder.build()

    def generate_optimized_molecules(self,
                                     number_molecules: int,
                                     with_random_samples: bool = False,
                                     smiles_docking_score_fn=None,
                                     random_samples: int = 100):
        self._fine_tune()
        self._train_mlp()

        gauss_samples = None

        if with_random_samples:
            assert smiles_docking_score_fn is not None
            assert random_samples is not None
            logger.info('Random sampling')
            gauss_samples = self.random_gauss(smiles_docking_score_fn, random_samples)

        results_builder = OptimizedMolecules.Builder()

        while results_builder.size < number_molecules:
            logger.info(f'Generated {results_builder.size} / {number_molecules}')
            latents = np.random.normal(size=(self.batch_size, self.latent))

            before = [self.mlp.latent_score(latents[i].reshape(1, -1))
                      for i in range(self.batch_size)]

            for _ in range(self.descent_iterations):
                latents += self.mlp.gradient(latents) * self.descent_lr

            try:
                smiles = [canonicalize(smi) for smi in self.gvae.decode(latents)]
            except (RuntimeError, ValueError):
                logger.error('Decoding failed')
                continue

            for i, smi in enumerate(smiles):
                if smi is not None and is_valid(smi):
                    try:
                        if smi not in results_builder:
                            latent_score = self.mlp.latent_score(latents[i].reshape(1, -1))
                            logger.info(f'Optimized from {before[i]} to {latent_score}')
                            output_path = os.path.join(
                                self.output_dir,
                                f'{results_builder.size}.mol2') if self.output_dir is not None else None
                            docking_score = smiles_docking_score_fn(smi, output_path=output_path,
                                                                    n_cpu=self.docking_n_cpu)
                            results_builder.append(
                                smi,
                                latent_vector=latents[i],
                                predicted_score=latent_score,
                                **docking_score
                            )
                        else:
                            logger.info('Generated SMILES %s already present in OptimizedMoleculesBuilder', smi)
                    except (RuntimeError, ValueError):
                        logger.error('Docking failed for ' + smi)

                    if results_builder.size >= number_molecules:
                        logger.info(f'Generating finished')
                    break

            results_builder.total_samples += self.batch_size

        return results_builder.build(), gauss_samples

    def get_filtered_dataset(self, dataset):
        logger.info('Filtering dataset')
        smiles, scores = dataset
        _, valid_indices = self.gvae.to_one_hots(smiles, with_valid_indices=True)
        data, scores = [smiles[i] for i in valid_indices], scores[valid_indices]
        logger.info('Filtering finished')
        return data, scores
