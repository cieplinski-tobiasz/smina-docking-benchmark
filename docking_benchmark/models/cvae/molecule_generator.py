import logging

import numpy as np
from sklearn.model_selection import train_test_split

import docking_benchmark.models.cvae.mol_utils as mol_utils
import docking_benchmark.models.cvae.vae_utils as vae_utils
from docking_benchmark.docking.predicted_docking_functions import MLPPredictedDockingScore
from docking_benchmark.utils.chemistry import is_valid, canonicalize

logger = logging.getLogger(__name__)


class CVAEGradientGenerator:
    def __init__(self, pretrained_path, dataset,
                 latent=196, batch_size=256,
                 fine_tune_epochs=5, fine_tune_test_size=0.1,
                 mlp_fit_epochs=50, mlp_test_size=0.1,
                 descent_iterations=50, descent_lr=0.05,
                 mode='minimize'):
        self.latent = latent
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_test_size = fine_tune_test_size
        self.cvae = vae_utils.VAEUtils(directory=pretrained_path)
        self.batch_size = batch_size
        self.mlp_fit_epochs = mlp_fit_epochs
        self.mlp_test_size = mlp_test_size
        self.descent_iterations = descent_iterations
        self.descent_lr = descent_lr
        smiles, scores = dataset
        filtered_smiles, valid_indices = mol_utils.smiles_to_hot_filter(
            smiles,
            self.cvae.char_to_index,
            self.cvae.params['MAX_LEN'],
            with_valid_indices=True
        )
        self.dataset = (filtered_smiles, scores[valid_indices])

        if mode == 'minimize':
            self.descent_lr *= -1
        elif mode != 'maximize':
            raise ValueError(f'Unknown mode {mode}')

    def _smiles_to_latent_fn(self):
        def smiles_to_latent(smiles):
            nonlocal self

            if type(smiles) is str:
                smiles = [smiles]

            return self.cvae.encode(self.cvae.smiles_to_hot(smiles))

        return smiles_to_latent

    def _fine_tune(self):
        x_dataset, _ = self.dataset

        one_hots = mol_utils.smiles_to_hot(
            x_dataset,
            self.cvae.params['MAX_LEN'],
            self.cvae.params['PADDING'],
            self.cvae.char_to_index,
            self.cvae.params['NCHARS']
        )

        train, test = train_test_split(one_hots, test_size=self.fine_tune_test_size)

        # CVAE implementation requires the dataset size
        # to be a multiple of 32
        train, test = train[:-(train.shape[0] % 32)], test[:-(test.shape[0] % 32)]

        logger.debug(f'Train dataset shape: {train.shape}')
        logger.debug(f'Test dataset shape: {test.shape}')

        model_train_targets = {
            'x_pred': train,
            'z_mean_log_var': np.ones((np.shape(train)[0], self.cvae.params['hidden_dim'] * 2))
        }
        model_test_targets = {
            'x_pred': test,
            'z_mean_log_var': np.ones((np.shape(test)[0], self.cvae.params['hidden_dim'] * 2))
        }

        self.cvae.autoencoder.fit(
            train, model_train_targets,
            epochs=self.fine_tune_epochs,
            validation_data=[test, model_test_targets]
        )

    def generate_optimized_molecules(self, number_molecules: int):
        self._fine_tune()

        predicted_docking_function = MLPPredictedDockingScore(
            self.dataset, input_dim=self.latent,
            epochs=self.mlp_fit_epochs, test_fraction=self.mlp_test_size,
            to_latent_fn=self._smiles_to_latent_fn(),
        )

        valid_samples = []
        total_sampled = 0

        while len(valid_samples) < number_molecules:
            latents = np.random.normal(size=(self.batch_size, self.latent))

            before = [predicted_docking_function.latent_score(latents[i].reshape(1, -1))
                      for i in range(self.batch_size)]

            for _ in range(self.descent_iterations):
                latents += predicted_docking_function.gradient(latents) * self.descent_lr

            smiles = [canonicalize(smi) for smi in
                      self.cvae.hot_to_smiles(self.cvae.decode(latents), strip=True, numpy=True)]

            for i, smi in enumerate(smiles):
                if smi is not None and is_valid(smi):
                    latent_score = predicted_docking_function.latent_score(latents[i].reshape(1, -1))
                    logger.info(f'Optimized from {before[i]} to {latent_score}')
                    valid_samples.append((smi, latent_score))

            total_sampled += self.batch_size

            logger.info(f'Sampled {len(valid_samples) / number_molecules}')

        if len(valid_samples) > number_molecules:
            valid_samples = valid_samples[:number_molecules]

        return valid_samples
