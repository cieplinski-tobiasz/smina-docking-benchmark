import textwrap

import pandas as pd
import pytest

from docking_benchmark.data.results import OptimizedMolecules


def test_builder_should_raise_on_empty_smiles():
    uut = OptimizedMolecules.Builder()

    with pytest.raises(ValueError):
        uut.append('')


def test_builder_should_raise_on_only_whitespace_smiles():
    uut = OptimizedMolecules.Builder()

    with pytest.raises(ValueError):
        uut.append('     ')


def test_builder_should_raise_when_smiles_is_not_str():
    uut = OptimizedMolecules.Builder()

    with pytest.raises(TypeError):
        uut.append(34)


def test_builder_happy_path():
    uut = OptimizedMolecules.Builder()
    uut.append('C', col1=1, col2=2)
    uut.append('H', col1=3, col2=4)
    expected = pd.DataFrame.from_dict({'C': {'col1': 1, 'col2': 2}, 'H': {'col1': 3, 'col2': 4}}, orient='index')

    result = uut.build()

    assert result.molecules.equals(expected)


def test_builder_happy_path_with_missing_data():
    uut = OptimizedMolecules.Builder()
    uut.append('C', col1=1)
    uut.append('H', col2=2)
    expected = pd.DataFrame.from_dict({'C': {'col1': 1}, 'H': {'col2': 2}}, orient='index')

    result = uut.build()

    assert result.molecules.equals(expected)


def test_builder_calculates_validity_when_total_samples_positive():
    uut = OptimizedMolecules.Builder()
    uut.append('C', col1=1)
    uut.append('H', col2=2)
    uut.total_samples += 10

    result = uut.build()

    assert result.metrics['validity'] == 0.2


def test_builder_does_not_calculate_validity_when_total_samples_is_zero():
    uut = OptimizedMolecules.Builder()
    uut.append('C', col1=1)
    uut.append('H', col2=2)

    result = uut.build()

    assert 'validity' not in result.metrics


def test_rmse_raises_when_col1_not_in_molecules():
    builder = OptimizedMolecules.Builder()
    builder.append('C', col=1)
    builder.append('H', col=2)
    uut = builder.build()

    with pytest.raises(ValueError):
        uut.rmse('no_such_column', 'col')


def test_rmse_raises_when_col2_not_in_molecules():
    builder = OptimizedMolecules.Builder()
    builder.append('C', col=1)
    builder.append('H', col=2)
    uut = builder.build()

    with pytest.raises(ValueError):
        uut.rmse('col', 'no_such_column')


def test_rmse_raises_when_col1_contains_nans():
    builder = OptimizedMolecules.Builder()
    builder.append('C', col1=1, col2=2)
    builder.append('H', col2=2)
    uut = builder.build()

    with pytest.raises(ValueError):
        uut.rmse('col1', 'col2')


def test_rmse_raises_when_col2_contains_nans():
    builder = OptimizedMolecules.Builder()
    builder.append('C', col1=1)
    builder.append('H', col1=1, col2=2)
    uut = builder.build()

    with pytest.raises(ValueError):
        uut.rmse('col1', 'col2')


def test_rmse_no_error_happy_path():
    builder = OptimizedMolecules.Builder()
    builder.append('C', col1=0, col2=0)
    builder.append('H', col1=1, col2=1)
    uut = builder.build()

    result = uut.rmse('col1', 'col2')

    assert result == 0


def test_rmse_with_error_happy_path():
    builder = OptimizedMolecules.Builder()
    builder.append('C', col1=0, col2=1)
    builder.append('H', col1=0, col2=1)
    builder.append('Cl', col1=0, col2=1)
    uut = builder.build()

    result = uut.rmse('col1', 'col2')

    assert result == 1


def test_to_csv_happy_path(tmp_path):
    builder = OptimizedMolecules.Builder()
    builder.append('C', col1=0, col2=1)
    builder.append('H', col1=2.5, col2=1)
    uut = builder.build()
    file = tmp_path / 'tmp_file.csv'
    expected = textwrap.dedent('''\
    SMILES,col1,col2
    C,0.0,1
    H,2.5,1
    ''')
    uut.to_csv(file)

    assert file.read_text() == expected
