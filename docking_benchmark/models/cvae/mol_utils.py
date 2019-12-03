import numpy as np
from rdkit.Chem import AllChem as Chem


def pad_smile(string, max_len, padding='right'):
    if len(string) >= max_len:
        return string

    if padding == 'right':
        return string + " " * (max_len - len(string))
    elif padding == 'left':
        return " " * (max_len - len(string)) + string

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


def smiles_to_hot(smiles, max_len, padding, char_indices, nchars):
    smiles = [pad_smile(i, max_len, padding)
              for i in smiles if pad_smile(i, max_len, padding)]

    X = np.zeros((len(smiles), max_len, nchars), dtype=np.float32)

    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                X[i, t, char_indices[char]] = 1
            except KeyError as e:
                print("ERROR: Check chars file. Bad SMILES:", smile)
                raise e
    return X


def smiles_to_hot_filter(smiles, char_indices):
    filtered_smiles = []
    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                char_indices[char]
            except KeyError:
                break
        else:
            filtered_smiles.append(smile)
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


def get_molecule_smi(mol_obj):
    return Chem.MolToSmiles(mol_obj)


def canon_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)
