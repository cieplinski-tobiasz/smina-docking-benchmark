import math

from keras.callbacks import ModelCheckpoint

import docking_benchmark.models.gvae.zinc_grammar as zinc_grammar
from docking_benchmark.models.gvae.model_zinc import MoleculeVAE, MAX_LEN
from docking_benchmark.models.gvae.molecule_vae import ZincGrammarModel
from docking_benchmark.utils.scripting import setup_and_get_logger

logger = setup_and_get_logger(name=__name__)


def get_gvae_model(latent=56, max_len=MAX_LEN, lr=2e-4):
    gvae = MoleculeVAE()
    gvae.create(zinc_grammar.gram.split('\n'), max_length=max_len, latent_rep_size=latent, lr=lr)
    return ZincGrammarModel(gvae, latent_rep_size=latent)


def train_gvae(gvae, dataset, *, epochs=15, mode='generator', batch_size=64, save_path):
    if mode == 'generator':
        gvae.vae.autoencoder.fit_generator(
            dataset.load_split_as_generator('train', smiles_to_input_fn=gvae.to_one_hots, batch_size=batch_size),
            math.ceil(dataset.size_for_split('train') / batch_size),
            epochs=epochs,
            validation_data=dataset.load_split_as_generator('valid', smiles_to_input_fn=gvae.to_one_hots,
                                                            batch_size=batch_size),
            validation_steps=math.ceil(dataset.size_for_split('valid') / batch_size),
            callbacks=[
                ModelCheckpoint(save_path + '-epoch{epoch:02d}-val_loss{val_loss:.2f}-val_acc{val_acc:.2f}.hdf5',
                                monitor='val_loss',
                                save_best_only=True)
            ]
        )
    else:
        train_one_hots = gvae.to_one_hots(dataset.load_split('train'))
        valid_one_hots = gvae.to_one_hots(dataset.load_split('valid'))
        gvae.vae.autoencoder.fit(
            train_one_hots,
            train_one_hots,
            epochs=epochs,
            validation_data=(valid_one_hots, valid_one_hots)
        )

    return gvae
