import nltk
import numpy as np

import docking_benchmark.models.gvae.model_zinc as model_zinc
import docking_benchmark.models.gvae.zinc_grammar as zinc_grammar
from docking_benchmark.utils.scripting import setup_and_get_logger

NOTHING_PRODUCTION = 'Nothing'

logger = setup_and_get_logger(__name__)


def get_zinc_tokenizer(cfg):
    long_tokens = [key for key in cfg._lexical_index.keys() if len(key) > 1]
    replacements = ['$', '%', '^']  # ,'&']
    assert len(long_tokens) == len(replacements)

    for token in replacements:
        assert token not in cfg._lexical_index

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])

        tokens = []

        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens

    return tokenize


def pop_or_nothing(S):
    try:
        return S.pop()
    except:
        return NOTHING_PRODUCTION


def prods_to_eq(prods):
    seq = [prods[0].lhs()]

    for prod in prods:
        if str(prod.lhs()) == NOTHING_PRODUCTION:
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''


class ZincGrammarModel(object):

    def __init__(self, pretrained_model=None, latent_rep_size=56):
        """ Load the (trained) zinc encoder/decoder, grammar model. """
        self._grammar = zinc_grammar
        self._model = model_zinc
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {prod: ix for ix, prod in enumerate(self._productions)}
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = get_zinc_tokenizer(self._grammar.GCFG)
        self._n_chars = len(self._productions)
        self._lhs_map = {lhs: ix for ix, lhs in enumerate(self._grammar.lhs_list)}
        self.MAX_LEN = self._model.MAX_LEN

        if type(pretrained_model) is model_zinc.MoleculeVAE:
            self.vae = pretrained_model

        if type(pretrained_model) is str:
            self.vae = self._model.MoleculeVAE()
            self.vae.load(self._productions, pretrained_model, max_length=self.MAX_LEN, latent_rep_size=latent_rep_size)

    def to_one_hots(self, smiles, with_valid_indices=False):
        if type(smiles) is str:
            smiles = [smiles]

        productions_for_smiles = []
        parsable_smiles_indices = []

        for i, smi in enumerate(smiles):
            try:
                tokens = self._tokenize(smi)
                parse_tree = self._parser.parse(tokens).__next__()
                production_seq = parse_tree.productions()
                productions_for_smiles.append(np.array([self._prod_map[prod] for prod in production_seq], dtype=int))
                parsable_smiles_indices.append(i)
            except (StopIteration, ValueError):
                logger.error(f'Failed parsing for {smi}')

        one_hots = np.zeros((len(productions_for_smiles), self.MAX_LEN, self._n_chars), dtype=np.float32)
        valid = []

        try:
            for i, productions in enumerate(productions_for_smiles):
                num_productions = len(productions)
                if num_productions > 277:
                    continue
                one_hots[i][np.arange(num_productions), productions] = 1.
                one_hots[i][np.arange(num_productions, self.MAX_LEN), -1] = 1.
                valid.append(i)
        except IndexError:
            logger.error(f'Failed conversion to one hot vector')

        if with_valid_indices:
            return one_hots[valid], [parsable_smiles_indices[i] for i in valid]

        return one_hots[valid]

    def encode(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        assert type(smiles) == list

        return self.vae.encoder.predict(self.to_one_hots(smiles))

    def _sample_using_masks(self, unmasked):
        """ Samples a one-hot vector, masking at each timestep.
            This is an implementation of Algorithm ? in the paper. """
        eps = 1e-100
        X_hat = np.zeros_like(unmasked)

        # Create a stack for each input in the batch
        S = np.empty((unmasked.shape[0],), dtype=object)
        for ix in range(S.shape[0]):
            S[ix] = [str(self._grammar.start_index)]

        # Loop over time axis, sampling values and updating masks
        for t in range(unmasked.shape[1]):
            next_nonterminal = [self._lhs_map[pop_or_nothing(a)] for a in S]
            mask = self._grammar.masks[next_nonterminal]
            masked_output = np.exp(unmasked[:, t, :]) * mask + eps
            sampled_output = np.argmax(np.random.gumbel(size=masked_output.shape) + np.log(masked_output), axis=-1)
            X_hat[np.arange(unmasked.shape[0]), t, sampled_output] = 1.0

            # Identify non-terminals in RHS of selected production, and
            # push them onto the stack in reverse order
            rhs = [list(filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
                               self._productions[i].rhs()))
                   for i in sampled_output]
            for ix in range(S.shape[0]):
                S[ix].extend(list(map(str, rhs[ix]))[::-1])
        return X_hat  # , ln_p

    def one_hot_to_smiles(self, x_hat):
        prod_seq = [
            [self._productions[x_hat[index, t].argmax()]
             for t in range(x_hat.shape[1])]
            for index in range(x_hat.shape[0])
        ]

        return [prods_to_eq(prods) for prods in prod_seq]

    def decode(self, z):
        """ Sample from the grammar decoder """
        assert z.ndim == 2
        unmasked = self.vae.decoder.predict(z)
        X_hat = self._sample_using_masks(unmasked)
        return self.one_hot_to_smiles(X_hat)
