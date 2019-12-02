import pytest

import docking_benchmark.docking.smina.parsing as parsing


@pytest.mark.parametrize(
    'stdout,expected',
    [
        (
                [
                    '## Name a b c d e f',
                    'Affinity: -8.86427 (kcal/mol)',
                    'Intramolecular energy: -0.50516',
                    'Term values, before weighting:',
                    '## ***** 85.70935 1433.76200 1.34850 37.42693 1.41956 0.00000',
                ],
                [
                    {
                        'affinity': -8.86427,
                        'intramolecular_energy': -0.50516,
                        'pre_weighting_terms': {
                            'a': 85.70935,
                            'b': 1433.76200,
                            'c': 1.34850,
                            'd': 37.42693,
                            'e': 1.41956,
                            'f': 0.0,
                        },
                    }
                ]
        ),
        (
                [
                    '## Name a b',
                    'Affinity: -1 (kcal/mol)',
                    'Intramolecular energy: -1',
                    'Term values, before weighting:',
                    '## ***** 1 2',
                    'Refine time 0.01',
                    'Affinity: -2 (kcal/mol)',
                    'Intramolecular energy: -2',
                    'Term values, before weighting:',
                    '## ***** 3 4',
                ],
                [
                    {
                        'affinity': -1,
                        'intramolecular_energy': -1,
                        'pre_weighting_terms': {
                            'a': 1,
                            'b': 2,
                        },
                    },
                    {
                        'affinity': -2,
                        'intramolecular_energy': -2,
                        'pre_weighting_terms': {
                            'a': 3,
                            'b': 4,
                        },
                    }
                ]
        )
    ]
)
def test_parse_score_only(stdout, expected):
    assert parsing.parse_score_only(stdout) == expected


@pytest.mark.parametrize(
    'stdout,expected',
    [
        (
                [
                    '## Name name,with,comma name_without_comma commas,_and,underlines',
                    'Affinity: 0 (kcal/mol)',
                    'Intramolecular energy: 0',
                    'Term values, before weighting:',
                    '## ***** 1 2 3',
                ],
                [
                    {
                        'affinity': 0,
                        'intramolecular_energy': 0,
                        'pre_weighting_terms': {
                            'name_with_comma': 1,
                            'name_without_comma': 2,
                            'commas__and_underlines': 3,
                        },
                    }
                ]
        )
    ]
)
def test_parse_score_only_substitutes_commas(stdout, expected):
    assert parsing.parse_score_only(stdout) == expected
