import numpy as np
from rdkit.Chem import AllChem as Chem

from docking_benchmark.utils.scripting import setup_and_get_logger

logger = setup_and_get_logger(name=__name__)


def pad_smile(string, max_len, padding='right'):
    if len(string) >= max_len:
        return string

    if padding == 'right':
        return string.ljust(max_len)

    if padding == 'left':
        return string.rjust(max_len)

    return string


def filter_valid_length(strings, max_len):
    return [s for s in strings if len(s) <= max_len]


def filter_valid_smiles_return_invalid(strings, max_len):
    filter_list = []
    new_smiles = []
    for idx, s in enumerate(strings):
        if len(s) > max_len:
            filter_list.append(idx)
        else:
            new_smiles.append(s)
    return new_smiles, filter_list


def smiles_to_hot(smiles, max_len, padding, char_to_index, n_chars):
    smiles = [
        pad_smile(i, max_len, padding)
        for i in smiles if pad_smile(i, max_len, padding)
    ]

    X = np.zeros((len(smiles), max_len, n_chars), dtype=np.float32)

    for i, smi in enumerate(smiles):
        for t, char in enumerate(smi):
            try:
                X[i, t, char_to_index[char]] = 1
            except KeyError as e:
                logger.error('Check chars file. Bad SMILES: ' + smi)
                raise e

    return X


def smiles_to_hot_filter(smiles, char_indices, max_len, with_valid_indices=False):
    filtered_smiles = []
    indices = []

    for i, smi in enumerate(smiles):
        if len(smi) > max_len:
            continue

        for char in smi:
            try:
                char_indices[char]
            except KeyError:
                break
        else:
            if with_valid_indices:
                indices.append(indices)

            filtered_smiles.append(smi)

    if with_valid_indices:
        return filtered_smiles, indices

    return filtered_smiles


def hot_to_smiles(hot_x, indices_chars):
    if type(indices_chars) is np.ndarray:
        indices_batch = np.argmax(hot_x, axis=2)
        smiles_arrays = indices_chars[indices_batch]
        smiles = [''.join(smi).strip() for smi in smiles_arrays]

        return smiles

    smiles = []

    for x in hot_x:
        temp_str = ""
        for j in x:
            index = np.argmax(j)
            temp_str += indices_chars[index]
        smiles.append(temp_str)
    return smiles


def balanced_parentheses(input_string):
    s = []
    balanced = True
    index = 0
    while index < len(input_string) and balanced:
        token = input_string[index]
        if token == "(":
            s.append(token)
        elif token == ")":
            if len(s) == 0:
                balanced = False
            else:
                s.pop()

        index += 1

    return balanced and len(s) == 0


def matched_ring(s):
    return s.count('1') % 2 == 0 and s.count('2') % 2 == 0


def fast_verify(s):
    return matched_ring(s) and balanced_parentheses(s)


def canon_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)
