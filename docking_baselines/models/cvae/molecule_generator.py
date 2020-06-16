import logging
import os.path

import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors

import docking_baselines.models.cvae.mol_utils as mol_utils
import docking_baselines.models.cvae.vae_utils as vae_utils
from docking_baselines.models.predicted_docking_functions import MLPPredictedDockingScore
from docking_benchmark.data.results import OptimizedMolecules
from docking_benchmark.utils.chemistry import is_valid, canonicalize


def lipinski_filter(smiles):
    mol = MolFromSmiles(smiles)
    return MolLogP(mol) <= 5 and NumHAcceptors(mol) <= 10 and NumHDonors(mol) <= 5 and 100 <= ExactMolWt(mol) <= 500


logger = logging.getLogger(__name__)


class CVAEGradientGenerator:
    def __init__(self, pretrained_path, dataset,
                 latent=196, batch_size=256,
                 fine_tune_epochs=5, fine_tune_test_size=0.1,
                 mlp_fit_epochs=50, mlp_test_size=0.1,
                 descent_iterations=50, descent_lr=0.05,
                 mode='minimize',
                 output_dir=None, docking_n_cpu=4):
        self.latent = latent
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_test_size = fine_tune_test_size
        self.cvae = vae_utils.VAEUtils(directory=pretrained_path)
        self.batch_size = batch_size
        self.mlp_fit_epochs = mlp_fit_epochs
        self.mlp_test_size = mlp_test_size
        self.descent_iterations = descent_iterations
        self.descent_lr = descent_lr
        self.output_dir = output_dir
        self.docking_n_cpu = docking_n_cpu
        train_smiles, test_smiles, train_scores, test_scores = dataset

        filtered_smiles, valid_indices = mol_utils.smiles_to_hot_filter(
            train_smiles,
            self.cvae.char_to_index,
            self.cvae.params['MAX_LEN'],
            with_valid_indices=True
        )
        self.train_dataset = filtered_smiles, train_scores[valid_indices]

        filtered_smiles, valid_indices = mol_utils.smiles_to_hot_filter(
            test_smiles,
            self.cvae.char_to_index,
            self.cvae.params['MAX_LEN'],
            with_valid_indices=True
        )
        self.test_dataset = filtered_smiles, test_scores[valid_indices]

        if mode == 'minimize':
            self.descent_lr *= -1
        elif mode != 'maximize':
            raise ValueError(f'Unknown mode {mode}')

        self.mlp = None
        self.fine_tuned = False

    def _smiles_to_latent_fn(self):
        def smiles_to_latent(smiles):
            nonlocal self

            if type(smiles) is str:
                smiles = [smiles]

            return self.cvae.encode(self.cvae.smiles_to_hot(smiles))

        return smiles_to_latent

    def fine_tune(self):
        if self.fine_tune_epochs <= 0 or self.fine_tuned:
            return

        x_train_dataset, _ = self.train_dataset
        x_test_dataset, _ = self.test_dataset

        train = mol_utils.smiles_to_hot(
            x_train_dataset,
            self.cvae.params['MAX_LEN'],
            self.cvae.params['PADDING'],
            self.cvae.char_to_index,
            self.cvae.params['NCHARS']
        )

        test = mol_utils.smiles_to_hot(
            x_test_dataset,
            self.cvae.params['MAX_LEN'],
            self.cvae.params['PADDING'],
            self.cvae.char_to_index,
            self.cvae.params['NCHARS']
        )

        # CVAE implementation requires the dataset size
        # to be a multiple of 32
        train, test = train[:-(train.shape[0] % 32)], test[:-(test.shape[0] % 32)]

        # Save train and test for descent_step
        self.train, self.test = train, test

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

    def train_mlp(self):
        if self.mlp is None:
            self.mlp = MLPPredictedDockingScore(
                self.train_dataset, input_dim=self.latent,
                epochs=self.mlp_fit_epochs, test_fraction=self.mlp_test_size,
                to_latent_fn=self._smiles_to_latent_fn(),
            )

    def random_gauss(self, smiles_docking_score_fn, size):
        assert smiles_docking_score_fn is not None
        assert size is not None

        logger.info('Random gauss sampling start')

        results_builder = OptimizedMolecules.Builder()

        while results_builder.size < size:
            logger.info(f'Random sampled {results_builder.size} / {size}')
            latents = np.random.normal(size=(self.batch_size, self.latent))

            smiles = [canonicalize(smi) for smi in
                      self.cvae.hot_to_smiles(self.cvae.decode(latents), strip=True, numpy=True)]

            for i, smi in enumerate(smiles):
                try:
                    if smi is not None and is_valid(smi) and lipinski_filter(smi):
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
                except Exception:
                    logger.error('Docking failed for ' + smi)

                if results_builder.size >= size:
                    logger.info('Random sampling finished')
                    break

            results_builder.total_samples += self.batch_size

        logger.info('Random gauss sampling finished')

        return results_builder.build()

    def generate_optimized_molecules(self, number_molecules, smiles_docking_score_fn):
        results_builder = OptimizedMolecules.Builder()

        while results_builder.size < number_molecules:
            logger.info(f'Generated {results_builder.size} / {number_molecules}')
            latents = np.random.normal(size=(self.batch_size, self.latent))

            before = [self.mlp.latent_score(latents[i].reshape(1, -1))
                      for i in range(self.batch_size)]

            for _ in range(self.descent_iterations):
                latents += self.mlp.gradient(latents) * self.descent_lr

            try:
                smiles = [canonicalize(smi) for smi in
                          self.cvae.hot_to_smiles(self.cvae.decode(latents), strip=True, numpy=True)]
            except (RuntimeError, ValueError):
                logger.error('Decoding failed')
                continue

            for i, smi in enumerate(smiles):
                try:
                    if smi is not None and is_valid(smi) and lipinski_filter(smi):
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
                except Exception:
                    logger.error('Docking failed for ' + smi)

                if results_builder.size >= number_molecules:
                    logger.info('Generating finished')
                    break

            results_builder.total_samples += self.batch_size

        return results_builder.build()

    def descent_steps(self, smiles_docking_score_fn, size):
        assert smiles_docking_score_fn is not None
        assert size is not None
        logger.info('Descent steps start')

        batch_size = 32
        min_valid_steps = 5

        results = []

        while len(results) < size:
            logger.info(f'Valid descent steps results: {len(results)} / {size}')
            one_hots = self.train[np.random.choice(self.train.shape[0], batch_size, replace=False), :]
            latents = self.cvae.encode(one_hots)
            latent_changes = [latents]

            for _ in range(self.descent_iterations):
                latents += self.mlp.gradient(latents) * self.descent_lr
                latent_changes.append(np.copy(latents))

            decoded_series = [self.cvae.hot_to_smiles(self.cvae.decode(delta), strip=True, numpy=True)
                              for delta in latent_changes]

            smiles_changes = [
                [canonicalize(smi) for smi in smi_list]
                for smi_list in decoded_series
            ]

            for i in range(batch_size):
                descent_results = OptimizedMolecules.Builder()

                for j in range(self.descent_iterations + 1):
                    smi = smiles_changes[j][i]

                    if smi is not None and is_valid(smi):
                        try:
                            docking_score = smiles_docking_score_fn(smi, n_cpu=self.docking_n_cpu)
                            descent_results.append(
                                smi,
                                latent_vector=latent_changes[j][i],
                                step=j,
                                **docking_score
                            )
                        except (ValueError, RuntimeError, TypeError):
                            logger.error('Docking failed for %s', smi)

                if descent_results.size > min_valid_steps:
                    results.append(descent_results.build())

        return results
