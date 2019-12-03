import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from keras import backend as K
from keras import optimizers
from keras.layers import Lambda
from keras.models import Model

from docking_benchmark.models.cvae import mol_utils as mu, hyperparameters
from docking_benchmark.models.cvae.models import load_encoder, load_decoder, load_property_predictor, \
    variational_layers
from docking_benchmark.utils.scripting import setup_and_get_logger

CHARS = [
    "7", "6", "o", "]", "3", "s", "(", "-", "S", "/", "B",
    "4", "[", ")", "#", "I", "l", "O", "H", "c", "1", "@",
    "=", "n", "P", "p", "8", "C", "2", "F", "5", "r", "N", "+",
    "e", "9", "%", "b", "0", "\\", " ",
]

logger = setup_and_get_logger(name=__name__)


def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    logger.info('x_mean shape in kl_loss: ' + x_mean.get_shape())
    return - 0.5 * K.mean(1 + x_log_var - K.square(x_mean) - K.exp(x_log_var), axis=-1)


class VAEUtils(object):
    def __init__(self,
                 exp_file='exp.json',
                 encoder_file=None,
                 decoder_file=None,
                 directory=None,
                 lr=2e-4,
                 estimate_estandard=True):
        # files
        if directory is not None:
            curdir = os.getcwd()
            os.chdir(os.path.join(curdir, directory))
            # exp_file = os.path.join(directory, exp_file)

        # load parameters
        self.params = hyperparameters.load_params(exp_file, False)
        if encoder_file is not None:
            self.params["encoder_weights_file"] = encoder_file
        if decoder_file is not None:
            self.params["decoder_weights_file"] = decoder_file

        # char stuff
        chars = yaml.safe_load(open(self.params['char_file'])) if 'char_file' in self.params else CHARS
        self.chars = chars
        self.params['NCHARS'] = len(chars)
        self.char_to_index = dict((char, i) for i, char in enumerate(chars))
        self.index_to_char = dict((i, char) for i, char in enumerate(chars))
        self.np_ix_char = np.array(chars)

        # encoder, decoder
        self.enc = load_encoder(self.params)
        self.dec = load_decoder(self.params)

        # make autoencoder for tuning
        x_in = self.enc.inputs[0]
        z_mean, enc_output = self.enc(x_in)
        kl_loss_var = K.variable(self.params['kl_loss_weight'])
        z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, self.params)

        x_out = self.dec([z_samp, x_in]) if self.params['do_tgru'] else self.dec(z_samp)
        x_out = Lambda(lambda x: K.identity(x), name='x_pred')(x_out)
        model_outputs = [x_out, z_mean_log_var_output]
        self.autoencoder = Model(x_in, model_outputs)

        def vae_anneal_metric(y_true, y_pred):
            return kl_loss_var

        self.kl_loss_var = kl_loss_var
        xent_loss_weight = K.variable(self.params['xent_loss_weight'])
        model_losses = {
            'x_pred': self.params['loss'],
            'z_mean_log_var': kl_loss
        }

        self.autoencoder.compile(
            loss=model_losses,
            loss_weights=[xent_loss_weight, kl_loss_var],
            optimizer=optimizers.Adam(lr=lr),
            metrics={'x_pred': ['categorical_accuracy', vae_anneal_metric]})

        self.encode, self.decode = self.enc_dec_functions()

        if self.params['do_prop_pred']:
            self.property_predictor = load_property_predictor(self.params)

        self.data = None

        if estimate_estandard:
            # Load data without normalization as dataframe
            df = pd.read_csv(self.params['data_file'])
            df.iloc[:, 0] = df.iloc[:, 0].str.strip()
            df = df[df.iloc[:, 0].str.len() <= self.params['MAX_LEN']]
            self.smiles = df.iloc[:, 0].tolist()
            if df.shape[1] > 1:
                self.data = df.iloc[:, 1:]

            self.estimate_estandarization()

        if directory is not None:
            os.chdir(curdir)

    def estimate_estandarization(self):
        logger.info('Standarization: estimating mu and std values...')

        # sample Z space
        smiles = self.random_molecules(size=50000)
        batch = 2500
        latents = np.zeros((len(smiles), self.params['hidden_dim']))

        for chunk in self.chunks(list(range(len(smiles))), batch):
            sub_smiles = [smiles[i] for i in chunk]
            one_hot = self.smiles_to_hot(sub_smiles)
            latents[chunk, :] = self.encode(one_hot, False)

        self.mu = np.mean(latents, axis=0)
        self.std = np.std(latents, axis=0)
        self.Z = self.standardize_z(latents)

        logger.info('Standardization finished')

    def standardize_z(self, z):
        return (z - self.mu) / self.std

    def unstandardize_z(self, z):
        return (z * self.std) + self.mu

    def perturb_z(self, z, noise_norm, constant_norm=False):
        if noise_norm > 0.0:
            noise_vec = np.random.normal(0, 1, size=z.shape)
            noise_vec = noise_vec / np.linalg.norm(noise_vec)
            if constant_norm:
                return z + (noise_norm * noise_vec)
            else:
                noise_amp = np.random.uniform(
                    0, noise_norm, size=(z.shape[0], 1))
                return z + (noise_amp * noise_vec)
        else:
            return z

    def enc_dec_functions(self, standardized=True):
        logger.info('Using standardized functions? {}'.format(standardized))

        if not self.params['do_tgru']:
            def decode(z, standardize=standardized):
                if standardize:
                    z = self.unstandardize_z(z)

                return self.dec.predict(z)
        else:
            def decode(z, standardize=standardized):
                fake_shape = (z.shape[0], self.params['MAX_LEN'], self.params['NCHARS'])
                fake_in = np.zeros(fake_shape)

                if standardize:
                    z = self.unstandardize_z(z)

                return self.dec.predict([z, fake_in])

        def encode(one_hot, standardize=standardized):
            latent = self.enc.predict(one_hot)[0]
            return self.standardize_z(latent) if standardize else latent

        return encode, decode

    def smiles_to_hot(self, smiles, canonize_smiles=True, check_smiles=False):
        if isinstance(smiles, str):
            smiles = [smiles]

        if canonize_smiles:
            smiles = [mu.canon_smiles(s) for s in smiles]

        if check_smiles:
            smiles = mu.smiles_to_hot_filter(smiles, self.char_to_index, self.params['MAX_LEN'])

        z = mu.smiles_to_hot(
            smiles,
            self.params['MAX_LEN'],
            self.params['PADDING'],
            self.char_to_index,
            self.params['NCHARS']
        )

        return z

    def hot_to_smiles(self, hot_x, strip=False, numpy=False):
        if not numpy:
            smiles = mu.hot_to_smiles(hot_x, self.index_to_char)
        else:
            smiles = mu.hot_to_smiles(hot_x, self.np_ix_char)

        if strip:
            smiles = [s.strip() for s in smiles]

        return smiles

    def random_molecules(self, size=None):
        if size is None:
            return self.smiles
        else:
            return random.sample(self.smiles, size)

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def dataset_generator_wrapper(self, gen, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = self.params['hidden_dim']

        while True:
            batch = next(gen)[0]
            wrapped = {
                'input_molecule_smi': batch,
                'x_pred': batch,
                'z_mean_log_var': np.ones((np.shape(batch)[0], hidden_dim * 2))
            }
            yield wrapped, wrapped

    def dataset_wrapper(self, dataset, hidden_dim=None, batch_size=None):
        if hidden_dim is None:
            hidden_dim = self.params['hidden_dim']

        if batch_size is None:
            batch_size = self.params['batch_size']

        if dataset.shape[0] % batch_size != 0:
            dataset = dataset[:-(dataset.shape[0] % batch_size)]

        wrapped = {
            'input_molecule_smi': dataset,
            'x_pred': dataset,
            'z_mean_log_var': np.ones((dataset.shape[0], hidden_dim * 2))
        }
        return wrapped, wrapped

    def test_dataset_wrapper(self, dataset, hidden_dim=None, batch_size=None):
        if hidden_dim is None:
            hidden_dim = self.params['hidden_dim']

        if batch_size is None:
            batch_size = self.params['batch_size']

        if dataset.shape[0] % batch_size != 0:
            dataset = dataset[:-(dataset.shape[0] % batch_size)]

        model_test_targets = {
            'x_pred': dataset,
            'z_mean_log_var': np.ones((np.shape(dataset)[0], hidden_dim * 2))
        }

        return dataset, model_test_targets
