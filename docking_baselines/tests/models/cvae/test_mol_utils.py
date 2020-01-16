import pytest

import docking_baselines.models.cvae.mol_utils as uut


@pytest.mark.parametrize(
    'smiles,char_indices,max_len,expected',
    [
        (
                [
                    'CCCC',
                    'CC',
                    'C',
                ],
                {
                    'C': 0
                },
                3,
                [
                    'CC',
                    'C',
                ]
        )
    ]
)
def test_smiles_to_hot_filter_ignores_too_long_smiles(smiles, char_indices, max_len, expected):
    assert uut.smiles_to_hot_filter(smiles, char_indices, max_len) == expected


@pytest.mark.parametrize(
    'smiles,char_indices,max_len,expected',
    [
        (
                [
                    'CN',
                    'CC',
                    'C',
                ],
                {
                    'C': 0
                },
                3,
                [
                    'CC',
                    'C',
                ]
        )
    ]
)
def test_smiles_to_hot_filter_ignores_invalid_chars(smiles, char_indices, max_len, expected):
    assert uut.smiles_to_hot_filter(smiles, char_indices, max_len) == expected
