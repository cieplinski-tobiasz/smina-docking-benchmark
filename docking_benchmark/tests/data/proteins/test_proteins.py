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


def test_protein_init_happy_path(tmp_path):
    dummy_protein_file = tmp_path / 'protein_file.pdb'
    dummy_protein_file.write_text('dummy')
    metadata_file = tmp_path / 'metadata.json'
    metadata_file.write_text('{"pocket_center": [0.0, 0.0, 0.0]}')

    uut = proteins.Protein('test_name', tmp_path)

    assert uut.name == 'test_name'
    assert uut.directory == tmp_path
    assert uut.pocket_center == [0.0, 0.0, 0.0]
