import itertools as it
import numpy as np


def get_item_idx(item, arr1, arr2):
    assert len(arr1) == len(arr2)
    idx = [i for i, x in enumerate(arr1) if x == item][0]
    return arr2[idx]


def get_utterance_prob(utterance, utterances, utterance_probs):
    return get_item_idx(utterance, utterances, utterance_probs)


def get_meaning_prob(meaning, meanings, meaning_probs):
    return get_item_idx(meaning, meanings, meaning_probs)


class Semantics:
    def __init__(self, mapper):
        self.mapper = mapper

    def delta(self, utterance, meaning):
        return meaning in self.mapper[utterance]


class Language:
    def __init__(self, utterances, meanings,
                 p_utterances, p_meanings, semantics):
        self.utterances = utterances
        self.meanings = meanings
        self.p_utterances = p_utterances
        self.p_meanings = p_meanings
        self.semantics = semantics


class Agent:
    def __init__(self, language):
        self.language = language

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
            np.exp(np.log(listener_score) - self.cost(utterance))

    def p_speak(self, utterance, meaning):
        """p(u|m)"""
        score = self.speaker_score(utterance, meaning) / \
                np.sum(self.speaker_score(u, meaning) \
                       for u in self.language.utterances)
        return score

    def p_listen(self, utterance, meaning):
        """p(m|u)"""
        return int(self.language.semantics.delta(utterance, meaning))


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
        # print('part_a {}'.format(part_a))
        part_b = np.log(p_l_m) + np.log(p_u) if p_l_m != 0. else 0.
        # print('part_b {}'.format(part_b))
        curr_score = part_a * part_b
        score += curr_score
        # print(m, u, curr_score)
    return -score


if __name__ == '__main__':
    utts = ['a', 'b', 'c']
    meanings = [1, 2, 3]
    p_utts = [0.1, 0.3, 0.6]
    p_meanings = [0.1, 0.3, 0.6]

    lang0_semantics = Semantics({
        'a': [1],
        'b': [2],
        'c': [3]
    })
    lang0 = Language(utts, meanings, p_utts, p_meanings, lang0_semantics)

    lang1_semantics = Semantics({
        'a': [2],
        'b': [1],
        'c': [3]
    })
    lang1 = Language(utts, meanings, p_utts, p_meanings, lang1_semantics)


    lang2_semantics = Semantics({
        'a': [3],
        'b': [1],
        'c': [2]
    })
    lang2 = Language(utts, meanings, p_utts, p_meanings, lang2_semantics)

    a0 = Agent(lang0)
    a1 = Agent(lang1)
    a2 = Agent(lang2)

    print(a0.p_speak('a', 1))
    print(a0.p_speak('b', 1))
    print(a0.p_speak('c', 1))

    print(system_efficiency(a0, a0, lang0))
    print(system_efficiency(a1, a1, lang1))
    print(system_efficiency(a2, a2, lang2))
