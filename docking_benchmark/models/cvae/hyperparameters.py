import json
from collections import OrderedDict
from copy import deepcopy

from docking_benchmark.utils.scripting import setup_and_get_logger

DEFAULT_PARAMETERS = {
    # for starting model from a checkpoint
    'reload_model': False,
    'prev_epochs': 0,

    # general parameters
    'batch_size': 64,
    'epochs': 30,
    'val_split': 0.1,  # validation split
    'loss': 'categorical_crossentropy',  # set reconstruction loss

    # convolution parameters
    'batchnorm_conv': True,
    'conv_activation': 'tanh',
    'conv_depth': 4,
    'conv_dim_depth': 8,
    'conv_dim_width': 8,
    'conv_d_growth_factor': 1.15875438383,
    'conv_w_growth_factor': 1.1758149644,

    # decoder parameters
    'gru_depth': 4,
    'rnn_activation': 'tanh',
    'recurrent_dim': 488,
    'do_tgru': True,  # use custom terminal gru layer
    'terminal_GRU_implementation': 0,  # CPU intensive implementation (only one present)
    'tgru_dropout': 0.0,
    'temperature': 1.00,  # amount of noise for sampling the final output

    # middle layer parameters
    'hg_growth_factor': 1.2281884874932403,  # growth factor applied to determine size of next middle layer.
    'hidden_dim': 196,
    'middle_layer': 1,
    'dropout_rate_mid': 0.082832929704794792,
    'batchnorm_mid': True,  # apply batch normalization to middle layers
    'activation': 'tanh',

    # Optimization parameters
    'lr': 0.00039192162392520126,
    'momentum': 0.97170900638688007,
    'optim': 'adam',  # optimizer to be used

    # vae parameters
    'vae_annealer_start': 29,  # Center for variational weigh annealer
    'batchnorm_vae': False,  # apply batch normalization to output of the variational layer
    'vae_activation': 'tanh',
    'xent_loss_weight': 1.0,  # loss weight to assign to reconstruction error.
    'kl_loss_weight': 1.0,  # loss weight to assing to KL loss
    "anneal_sigmod_slope": 0.51066543057913916,  # slope of sigmoid variational weight annealer
    "freeze_logvar_layer": False,
    # Choice of freezing the variational layer until close to the anneal starting epoch
    "freeze_offset": 1,

    # property prediction parameters:
    'do_prop_pred': False,  # whether to do property prediction
    'prop_pred_depth': 3,
    'prop_hidden_dim': 36,
    'prop_growth_factor': 0.8,  # ratio between consecutive layer in property prediction
    'prop_pred_activation': 'tanh',
    'reg_prop_pred_loss': 'mse',  # loss function to use with property prediction error for regression tasks
    'logit_prop_pred_loss': 'binary_crossentropy',

    # loss function to use with property prediction for logistic tasks
    'prop_pred_loss_weight': 0.5,
    'prop_pred_dropout': 0.0,
    'prop_batchnorm': True,

    # print output parameters
    "verbose_print": 0,

    'MAX_LEN': 200,
    'RAND_SEED': 0,
    'PADDING': 'right'
}

logger = setup_and_get_logger(name=__name__)


def _log_loaded_params(params):
    logger.info('CVAE overwritten hyper-parameters:')

    for key, value in params.items():
        logger.info('{:25s} - {:12}'.format(key, str(value)))


def load_params(param_file=None, verbose=True):
    if param_file is None:
        return deepcopy(DEFAULT_PARAMETERS)

    loaded_parameters = json.loads(
        open(param_file).read(),
        object_pairs_hook=OrderedDict
    )

    if verbose:
        _log_loaded_params(loaded_parameters)

    parameters = deepcopy(DEFAULT_PARAMETERS)
    parameters.update(loaded_parameters)
    return parameters
