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
        ),
        (
                [
                    '## Name gauss(o=0,_w=0.5,_c=8) gauss(o=3,_w=2,_c=8) repulsion(o=0,_c=8) hydrophobic(g=0.5,_b=1.5,_c=8) non_dir_h_bond(g=-0.7,_b=0,_c=8) num_tors_div',
                    'Affinity: -7.46660 (kcal/mol)',
                    'Intramolecular energy: 0.13012',
                    'Term values, before weighting:',
                    '## CHEMBL364005 53.01924 1000.82690 1.75064 40.33972 1.55378 0.00000',
                ],
                [
                    {
                        'affinity': -7.46660,
                        'intramolecular_energy': 0.13012,
                        'pre_weighting_terms': {
                            'gauss(o=0__w=0.5__c=8)': 53.01924,
                            'gauss(o=3__w=2__c=8)': 1000.82690,
                            'repulsion(o=0__c=8)': 1.75064,
                            'hydrophobic(g=0.5__b=1.5__c=8)': 40.33972,
                            'non_dir_h_bond(g=-0.7__b=0__c=8)': 1.55378,
                            'num_tors_div': 0.00000,
                        },
                    },
                ]
        ),
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
