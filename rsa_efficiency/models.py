from collections import defaultdict
import itertools as it
import numpy as np


def all_meanings_available_filter(M):
    """

    Parameters
    ----------
    M: np.array
        A boolean matrix.

    Returns
    -------
    bool
        True if all meanings can be talked about (each col as some val.)

    """
    return all(np.sum(M, axis=0) > 0.)


def is_valid_matrix(M, filter_fn):
    """Filter on boolean matrices.

    E.g. we may want to make sure that we only consider
    boolean matrices that allow all meanings to be talked about.
    """
    return filter_fn(M)


def idxs2matrix(idxs, m, n):
    """Convert a list of indices to an M x N matrix.

    Example
    -------
    m, n = 3, 3
    idxs = [0, 2, 3, 8]

    idxs2matrix(idxs, m, n) -->

            [[1., 0., 1.],
             [1., 0., 0.],
             [0., 0., 1.]]

    """
    d = []
    for row in range(m):
        curr_row = []
        for col in range(n):
            if (row * m + col) in idxs:
                curr_row.append(1.)
            else:
                curr_row.append(0.)
        d.append(curr_row)
    return np.array(d)


def generate_all_boolean_matrices(m, n, n_true):
    """Generate all M x N boolean matrices with
    n_true ones and m*n-n_true zeros.
    """
    matrices = []
    idxs_range = range(m * n)
    all_subsets = it.combinations(idxs_range, n_true)
    for idxs in all_subsets:
        matrices.append((idxs, idxs2matrix(idxs, m, n)))
    return matrices


def matrix2dict(M, utterances, meanings):
    """Rows correspond to utterances, Cols correspond to meanings."""
    assert M.shape[0] == len(utterances) and M.shape[1] == len(meanings)
    d = defaultdict(list)
    for row in range(M.shape[0]):
        for col in range(M.shape[1]):
            if M[row][col] != 0.:
                d[utterances[row]].append(meanings[col])
    return d


def get_item_idx(item, arr1, arr2):
    assert len(arr1) == len(arr2)
    try:
        idx = [i for i, x in enumerate(arr1) if x == item][0]
    except:
        import pdb; pdb.set_trace();
    return arr2[idx]


def get_utterance_prob(utterance, utterances, utterance_probs):
    return get_item_idx(utterance, utterances, utterance_probs)


def get_meaning_prob(meaning, meanings, meaning_probs):
    return get_item_idx(meaning, meanings, meaning_probs)


class Semantics:
    """Literal semantics.

    """
    def __init__(self, mapper):
        self.mapper = mapper

    def delta(self, utterance, meaning):
        if utterance not in self.mapper:
            return False
        return meaning in self.mapper[utterance]


class Language:
    """Language.

    """
    def __init__(self, utterances, meanings,
                 p_utterances, p_meanings, semantics):
        self.utterances = utterances
        self.meanings = meanings
        self.p_utterances = p_utterances
        self.p_meanings = p_meanings
        self.semantics = semantics


class Agent:
    """Interlocutor agent.

    """
    def __init__(self, language, alpha=1.):
        self.language = language
        self.alpha = alpha

    def cost(self, utterance):
        """Default cost is utterance surprisal."""
        return -np.log(get_utterance_prob(utterance,
                                          self.language.utterances,
                                          self.language.p_utterances))

    def speak(self, meaning):
        """u ~ p(u|m)"""
        p_u_scores = \
            np.array([self.speaker_score(u, meaning) for u in
                      self.language.utterances])
        p_u_scores = p_u_scores / np.sum(p_u_scores)
        return p_u_scores

    def listen(self, utterance):
        """m ~ p(m|u)"""
        pass

    def speaker_score(self, utterance, meaning):
        listener_score = self.p_listen(utterance, meaning)
        if listener_score == 0.:
            return 0.
        return \
            np.exp(self.alpha * (np.log(listener_score) - self.cost(utterance)))

    def listener_score(self, utterance, meaning):
        def score_meaning(u, m):
            return int(self.language.semantics.delta(u, m)) * \
                   get_meaning_prob(m, self.language.meanings, self.language.p_meanings)
        score = score_meaning(utterance, meaning)
        return score

    def p_speak(self, utterance, meaning):
        """p(u|m) \propto exp(-alpha * (log(L(m|u)-cost))"""
        Z = np.sum(self.speaker_score(u, meaning) \
                   for u in self.language.utterances)

        if Z == 0.:
            return 0.
        else:
            return self.speaker_score(utterance, meaning) / Z

    def p_listen(self, utterance, meaning):
        """p(m|u) = delta(u, m) * p(m)"""
        listener_score_ = self.listener_score(utterance, meaning)
        Z = np.sum([self.listener_score(utterance, m) \
                    for m in self.language.meanings])
        if Z == 0.:
            return 0.
        return listener_score_ / float(Z)


def system_efficiency(speaker, listener, language):
    """Cross entropy between speaker and listener using language.

    \sum_m p(m)S(u|m)log[L(m|u)p(u)]

    """
    score = 0.
    complete_support = list(it.product(language.meanings, language.utterances))
    for m, u in complete_support:
        p_m = get_meaning_prob(m, language.meanings, language.p_meanings)
        p_s_u = speaker.p_speak(u, m)
        p_l_m = listener.p_listen(u, m)
        p_u = get_utterance_prob(u, language.utterances, language.p_utterances)
        part_a = p_m * p_s_u
        if p_l_m == 0.:
            part_b = 0.
        else:
            part_b = (np.log(p_l_m) + np.log(p_u))
        curr_score = part_a * part_b
        score += curr_score
        # print(m, u, curr_score)
    return -score


if __name__ == '__main__':
    utts = ['a', 'b', 'c']
    meanings = [1, 2, 3]
    p_utts = [0.1, 0.3, 0.6]
    p_meanings = [0.1, 0.3, 0.6]


    matrices = generate_all_boolean_matrices(3, 3, 3)
    matrices = [(idxs, m) for idxs, m in matrices if all_meanings_available_filter(m)]
    all_languages = []
    for idxs, m in matrices:
        d = matrix2dict(m, utts, meanings)
        sems = Semantics(d)
        language = Language(utts, meanings, p_utts, p_meanings, sems)
        all_languages.append((idxs, language))

    d_results = []
    for idxs, language in all_languages:
        agent = Agent(language)
        d_results.append({
            "idxs": idxs,
            "score": system_efficiency(agent, agent, language)
        })
    print(sorted(d_results, key=lambda x: x['score']))


    # M = np.array([[1, 1, 0],
    #               [1, 0, 0],
    #               [0, 0, 1]])
    # print(matrix2dict(M, utts, meanings))

    #
    # lang0_semantics = Semantics({
    #     'a': [1],
    #     'b': [2],
    #     'c': [3]
    # })
    # lang0 = Language(utts, meanings, p_utts, p_meanings, lang0_semantics)
    #
    # lang1_semantics = Semantics({
    #     'a': [2],
    #     'b': [1],
    #     'c': [3]
    # })
    # lang1 = Language(utts, meanings, p_utts, p_meanings, lang1_semantics)
    #
    #
    # lang2_semantics = Semantics({
    #     'a': [3],
    #     'b': [2],
    #     'c': [1]
    # })
    # lang2 = Language(utts, meanings, p_utts, p_meanings, lang2_semantics)
    #
    # lang3_semantics = Semantics({
    #     'a': [3],
    #     'b': [1, 2]
    # })
    # lang3 = Language(utts, meanings, p_utts, p_meanings, lang3_semantics)
    #
    # lang4_semantics = Semantics({
    #     'c': [1, 2, 3]
    # })
    # lang4 = Language(utts, meanings, p_utts, p_meanings, lang4_semantics)
    #
    # alpha = 1.
    # a0 = Agent(lang0, alpha)
    # a1 = Agent(lang1, alpha)
    # a2 = Agent(lang2, alpha)
    # a3 = Agent(lang3, alpha)
    # a4 = Agent(lang4, alpha)
    #
    #
    # print(system_efficiency(a0, a0, lang0))
    # print(system_efficiency(a1, a1, lang1))
    # print(system_efficiency(a2, a2, lang2))
    # print(system_efficiency(a3, a3, lang3))
    # print(system_efficiency(a4, a4, lang4))
