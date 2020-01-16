import math

from keras.callbacks import ModelCheckpoint

from docking_baselines.models.cvae.vae_utils import VAEUtils
from docking_benchmark.utils.logging import setup_and_get_logger

logger = setup_and_get_logger(name=__name__)


def get_cvae_model(lr=2e-4):
    return VAEUtils(exp_file=None, lr=lr)


def train_cvae(cvae, dataset, *, epochs=15, mode='generator', batch_size=64, save_path=None):
    valid_one_hots = dataset.load_split('valid', cvae.smiles_to_hot)
    callbacks = []

    if save_path is not None:
        callbacks.append(ModelCheckpoint(
            save_path + '-epoch{epoch:02d}-val_loss{val_loss:.2f}'
            + '-val_pred_acc{val_x_pred_categorical_accuracy:.2f}.hdf5',
            monitor='val_loss',
            save_best_only=True))

    if mode == 'generator':
        cvae.autoencoder.fit_generator(
            cvae.dataset_generator_wrapper(
                dataset.load_split_as_generator('train', smiles_to_input_fn=cvae.smiles_to_hot, batch_size=batch_size)),
            math.ceil(dataset.size_for_split('train') / batch_size),
            epochs=epochs,
            validation_data=cvae.dataset_wrapper(valid_one_hots),
            callbacks=callbacks
        )
    else:
        train_one_hots = cvae.smiles_to_hot(dataset.load_split('train'))
        cvae.autoencoder.fit(
            train_one_hots,
            train_one_hots,
            epochs=epochs,
            validation_data=(valid_one_hots, valid_one_hots)
        )

    return cvae
