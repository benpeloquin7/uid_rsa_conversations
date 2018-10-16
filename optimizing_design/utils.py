"""Utils for RSA efficiency simulations.

See rsa_efficiency/efficiency-20180829.ipynb for use.

"""

import itertools as it
import numpy as np


#
# Objectives
#
def KL(M1, M2):
    """Kullback-Leilbler divergence.

    Note that M1 and M2 should be noramlized to form valid joint p(u, m)
    as in S(u|m)p(m) and L(m|u)p(u)

    """
    res = []
    for row in range(M1.shape[0]):
        for col, _ in enumerate(M1[row, :]):
            s = M1[row, col]
            l = M2[row, col]
            score = s * np.log2(s / l) if (s != 0.) and (l != 0.) else 0.
            res.append(score)
    return np.sum(res)


def crossEntropy(M1, M2):
    """Cross Entropy loss.

    Note that M1 and M2 should be noramlized to form valid joint p(u, m)
    as in S(u|m)p(m) and L(m|u)p(u)

    """
    res = []
    for row in range(M1.shape[0]):
        for col in range(M1.shape[1]):
            s = M1[row, col]
            l = M2[row, col]
            score = s * -np.log2(l) if (l != 0.) else 0.
            res.append(score)
    return np.sum(res)


def normalize(M):
    return M / np.sum(M)


def create_boolean_matrix(meanings, utterances):
    """Note that symbols (meanings) are columns, codes (utterances) are rows."""
    M = []
    for utt in utterances:
        M.append([1. if utt in meaning else 0. for meaning in meanings])
    return np.array(M)


#
# Matrix utils
#
def to_array(np_arr):
    return [v for v in np_arr]


def column_slice(M, select_cols):
    return M[:, select_cols]


def column_slice(M, select_cols):
    return M[:, select_cols]


def row_normalizer(M):
    """Normalize across columns (sum of each row is 1)"""
    row_totals = np.sum(M, axis=1).astype(float)
    return np.nan_to_num((M.transpose() / row_totals).transpose())


def column_normalizer(M):
    """Normalize across rows (sum of each column is 1)"""
    col_totals = np.sum(M, axis=0).astype(float)
    return np.nan_to_num(M / col_totals)


#
# RSA Agents
#
def rsa_speaker(M, depth=0):
    if depth == 0:
        return column_normalizer(M)
    for _ in range(depth):
        M = column_normalizer(row_normalizer(column_normalizer(M)))
    return M


def rsa_listener(M, depth=0):
    if depth == 0:
        return row_normalizer(M)
    for _ in range(depth):
        M = row_normalizer(column_normalizer(row_normalizer(M)))
    return M


#
# Simulation utils
#
def create_semantics_M(idxs, m=4, n=4):
    """Create all possible languages with 4 items (literal semantics)

    >>> create_semantics_M((0, 1, 2, 3))
    """
    M = np.arange(m * n)
    return np.array([1. if idx in idxs else 0. for idx in M]).reshape(m, n)


def can_express_all_meanings(M):
    """Check for valid languages..."""
    return np.all(np.sum(M, axis=0)) >= 1


def generate_languages(n_utterances=4, n_meanings=4,
                       filter=can_express_all_meanings):
    """
    Returns
    -------
    list of tuples
        Tuples contains (literal semantics mapping, code)

    """
    original = np.arange(n_utterances * n_meanings)
    Ms = [(create_semantics_M(idxs, m=n_meanings, n=n_utterances), idxs) for
          idxs in it.combinations(original, n_utterances)]
    Ms = [(M, code) for M, code in Ms if filter(M)]
    return Ms


def generate_utterance_costs(n_utterances=4):
    """

    Returns
    -------
    array of floats
        Utterance probabilities.

    """
    p_u = np.random.dirichlet([1.] * n_utterances)
    return p_u


def generate_need_probs(n_meanings=4, convert_to_surprisal=True):
    """

    Returns
    -------
    array of floats
        Utterance probabilities.

    """
    # Assign need probs
    p_m = np.random.dirichlet([1.] * n_meanings)
    if convert_to_surprisal:
        p_m = [-np.log2(v) for v in p_m]  # convert to surprisal
    return p_m


def generate_contexts(p_m, n_contexts=4, uniform=False):
    """

    Returns
    -------
    list of tuples of (float, list)
        (p_c, p(m|c))
    """
    # Create all contexts (re-arrange need probs)
    context_vals = [p_m[i:] + p_m[:i] for i in range(n_contexts)]
    context_probs = np.random.dirichlet([1.] * len(context_vals))
    if uniform:
        context_probs = [1.] * len(context_vals)
        context_probs = context_probs / np.sum(context_probs)
    contexts = zip(context_probs, context_vals)
    return list(contexts)


def generate_simulation(n_utterances, n_meanings, n_contexts,
                        convert_to_surprisal=True,
                        context_probs_uniform=False):
    p_u = generate_utterance_costs(n_utterances)
    p_m = generate_need_probs(n_meanings, convert_to_surprisal)
    contexts = generate_contexts(p_m, n_contexts, context_probs_uniform)
    return {
        'p_u': p_u,
        'p_m': p_m,
        'context': contexts
    }
