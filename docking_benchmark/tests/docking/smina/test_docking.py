import pytest

import docking_benchmark.docking.smina.docking as docking


@pytest.mark.parametrize(
    'scores,expected',
    [
        (
                [{'docking_score': -1, 'a': 1}],
                {'docking_score': -1, 'a': 1},
        ),
        (
                [{'docking_score': -1, 'a': 1}, {'docking_score': -2, 'a': 2}],
                {'docking_score': -1.5, 'a': 1.5},
        ),
        (
                [{'docking_score': -3, 'a': 1}, {'docking_score': -2, 'a': 3}, {'docking_score': -1, 'a': 5}],
                {'docking_score': -2.5, 'a': 2},
        ),
    ]
)
def test_top_n_aggregator(scores, expected):
    assert docking.top_n_aggregator(2)(scores) == expected


def test_top_n_raises_when_scores_inconsistent():
    with pytest.raises(ValueError):
        inconsistent_scores = [
            {'docking_score': -1, 'a': 0},
            {'docking_score': -2},
        ]
        docking.top_n_aggregator(2)(inconsistent_scores)


def test_top_n_raises_for_negative_n():
    with pytest.raises(ValueError):
        docking.top_n_aggregator(-1)


def test_top_n_raises_for_zero_n():
    with pytest.raises(ValueError):
        docking.top_n_aggregator(0)


@pytest.mark.parametrize(
    'scores,expected',
    [
        (
                [{'docking_score': -1}],
                {'docking_score': -1},
        ),
        (
                [{'docking_score': -1}, {'docking_score': -2}],
                {'docking_score': -2},
        ),
        (
                [{'docking_score': -3}, {'docking_score': -2}, {'docking_score': -1}],
                {'docking_score': -3},
        ),
    ]
)
def test_min_aggregator(scores, expected):
    assert docking.min_aggregator(scores) == expected
