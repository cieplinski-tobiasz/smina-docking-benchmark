import os

from docking_baselines.models.cvae.molecule_generator import CVAEGradientGenerator
from docking_baselines.models.cvae.trainer import get_cvae_model, train_cvae
from docking_baselines.models.gvae.molecule_generator import GVAEGradientGenerator
from docking_baselines.models.gvae.trainer import get_gvae_model, train_gvae
from docking_benchmark.data.directories import PRETRAINED_MODELS

ALL_MODELS = {
    'cvae': {
        'cls': CVAEGradientGenerator,
        'pretrained': os.path.join(PRETRAINED_MODELS, 'cvae'),
        'training': {
            'create_fn': get_cvae_model,
            'train_fn': train_cvae,
        },
    },
    'gvae': {
        'cls': GVAEGradientGenerator,
        'pretrained': os.path.join(PRETRAINED_MODELS, 'gvae', 'zinc_vae_grammar_L56_E100_val.hdf5'),
        'training': {
            'create_fn': get_gvae_model,
            'train_fn': train_gvae,
        },
    },
}
