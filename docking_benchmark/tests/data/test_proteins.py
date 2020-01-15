import textwrap

import pytest

import docking_benchmark.data.proteins as proteins


def test_protein_init_raises_when_protein_file_ambiguous(tmp_path):
    dummy_protein_files = (
        tmp_path / 'first_protein_file.pdb',
        tmp_path / 'second_protein_file.pdb'
    )
    for file in dummy_protein_files:
        file.write_text('dummy')

    with pytest.raises(ValueError):
        proteins.Protein('test_name', tmp_path)


def test_protein_init_raises_when_no_protein_files(tmp_path):
    with pytest.raises(ValueError):
        proteins.Protein('test_name', tmp_path)


def test_protein_init_raises_when_metadata_has_no_pocket_center(tmp_path):
    dummy_protein_file = tmp_path / 'protein_file.pdb'
    dummy_protein_file.write_text('dummy')
    metadata_file = tmp_path / 'metadata.json'
    metadata_file.write_text('{}')

    with pytest.raises(ValueError):
        proteins.Protein('test_name', tmp_path)


def test_protein_init_raises_when_pocket_center_has_invalid_length(tmp_path):
    dummy_protein_file = tmp_path / 'protein_file.pdb'
    dummy_protein_file.write_text('dummy')
    metadata_file = tmp_path / 'metadata.json'
    metadata_file.write_text('{"pocket_center": [0.0, 0.0]}')

    with pytest.raises(ValueError):
        proteins.Protein('test_name', tmp_path)


def test_protein_init_raises_when_pocket_center_is_not_coordinates(tmp_path):
    dummy_protein_file = tmp_path / 'protein_file.pdb'
    dummy_protein_file.write_text('dummy')
    metadata_file = tmp_path / 'metadata.json'
    metadata_file.write_text('{"pocket_center": ["f"]}')

    with pytest.raises(ValueError):
        proteins.Protein('test_name', tmp_path)


def test_protein_init_raises_if_directory_does_not_exist():
    with pytest.raises(ValueError):
        proteins.Protein('test_name', '/no/such/directory')


def test_protein_init_raises_if_metadata_file_does_not_exist(tmp_path):
    dummy_protein_file = tmp_path / 'protein_file.pdb'
    dummy_protein_file.write_text('dummy')

    with pytest.raises(ValueError):
        proteins.Protein('test_name', tmp_path)


def test_protein_init_happy_path(tmp_path):
    dummy_protein_file = tmp_path / 'protein_file.pdb'
    dummy_protein_file.write_text('dummy')
    metadata_file = tmp_path / 'metadata.json'
    metadata_file.write_text('{"pocket_center": [0.0, 0.0, 0.0]}')

    uut = proteins.Protein('test_name', tmp_path)

    assert uut.name == 'test_name'
    assert uut.directory == tmp_path
    assert uut.pocket_center == [0.0, 0.0, 0.0]


def mock_valid_protein(tmp_path):
    dummy_protein_file = tmp_path / 'protein_file.pdb'
    dummy_protein_file.write_text('dummy')
    metadata_file = tmp_path / 'metadata.json'
    datasets_dir = tmp_path / 'datasets'
    datasets_dir.mkdir()
    dummy_dataset = datasets_dir / 'dummy.csv'
    dummy_dataset.write_text(textwrap.dedent('''\
    SMILES,DOCKING_SCORE,A,B,()
    C,-2.3,1.0,1.0,1.0
    '''))
    metadata_file.write_text('''
    {
        "pocket_center": [0.0, 0.0, 0.0],
        "datasets": {
            "dummy": {
                "path": "datasets/dummy.csv"
            }
        }
    }
    ''')

    return proteins.Protein('mocked_protein', tmp_path)


def test_datasets_get_dataset_happy_path(tmp_path):
    protein = mock_valid_protein(tmp_path)
    uut = proteins.Datasets(protein)

    smiles, docking_scores = uut['dummy']

    assert len(smiles) == len(docking_scores)
    assert smiles[0] == 'C'
    assert docking_scores[0] == -2.3


def test_datasets_raises_when_dataset_does_not_exist(tmp_path):
    protein = mock_valid_protein(tmp_path)
    uut = proteins.Datasets(protein)

    with pytest.raises(KeyError):
        _ = uut['no_such_dataset']


def test_dataset_linear_combination_happy_path(tmp_path):
    protein = mock_valid_protein(tmp_path)
    uut = proteins.Datasets(protein)

    smiles, scores = uut.with_linear_combination_score('dummy', A=1.0, B=1.0)

    assert smiles[0] == 'C'
    assert scores[0] == 2.0


def test_dataset_linear_combination_happy_path_with_dict_passed_components(tmp_path):
    protein = mock_valid_protein(tmp_path)
    uut = proteins.Datasets(protein)
    components = {
        '()': 2.0,
        'A': -1.0,
        'B': 1.0
    }

    smiles, scores = uut.with_linear_combination_score('dummy', **components)

    assert smiles[0] == 'C'
    assert scores[0] == 2.0


def test_dataset_linear_combination_raises_with_non_existing_component(tmp_path):
    protein = mock_valid_protein(tmp_path)
    uut = proteins.Datasets(protein)

    with pytest.raises(ValueError):
        _ = uut.with_linear_combination_score('dummy', A=1.0, no_such_component=1.0)


def test_dataset_linear_combination_raises_with_no_components(tmp_path):
    protein = mock_valid_protein(tmp_path)
    uut = proteins.Datasets(protein)

    with pytest.raises(ValueError):
        _ = uut.with_linear_combination_score('dummy')
