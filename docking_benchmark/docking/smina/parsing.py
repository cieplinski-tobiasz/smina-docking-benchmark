from typing import Iterable, List


def parse_docking_score(smina_stdout: List[str]) -> List[float]:
    # WARNING: A Very hacky way to get score table
    table_header = '-----+------------+----------+----------'
    results_start_index = smina_stdout.index(table_header)
    score_table = smina_stdout[results_start_index + 1:-3]
    scores = [float(row.split()[1]) for row in score_table]

    return scores


def parse_score_only(smina_stdout: Iterable[str]):
    terms = []
    for line in smina_stdout:
        if line.startswith('## Name'):
            _, _, *parsed_terms = line.split()
            terms = parsed_terms
            break

    assert terms, "No terms specified in smina's scoring function"
    terms = [term.replace(',', '_') for term in terms]

    current_mode = -1
    results = []
    for line in smina_stdout:
        if line.startswith('Affinity:'):
            results.append(dict())
            current_mode += 1
            results[current_mode] = {}
            _, affinity, _ = line.split()
            results[current_mode]['affinity'] = float(affinity)
        elif line.startswith('Intramolecular energy:'):
            _, _, energy = line.split()
            results[current_mode]['intramolecular_energy'] = float(energy)
        elif line.startswith('##') and not line.startswith('## Name'):
            _, _, *term_values = line.split()
            results[current_mode]['pre_weighting_terms'] = {
                term: float(value) for term, value in zip(terms, term_values)
            }

    assert current_mode + 1 == len(results)

    return results
