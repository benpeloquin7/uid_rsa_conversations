"""Speaker / listener agent"""
from collections import Counter
import copy
import itertools as it
import math
import numpy as np
import lm


class UtteranceException(Exception):
    pass


class LMException(Exception):
    pass


class AgentException(Exception):
    pass


def create_dict(ks, vs):
    return {k: v for k, v in zip(ks, vs)}


def merge_dicts(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


class Utterance:
    def __init__(self, content):
        self.content = content

    def set_prob(self, p):
        self.prob = p

    def update_prob(self, p):
        self.prob += p

    def get_prob(self):
        if not self.prob:
            raise UtteranceException("Utterance prob has not been set")

        return self.prob

    def has_t(self):
        return 't' in self.content

    def has_RC(self):
        return 'b' in self.content.lower()

    def __str__(self):
        return self.content

    def __eq__(self, other):
        return self.content == other.content


class LM:
    def __init__(self):
        self.cond_probs = None

    def build_lm(self, c):
        """Initialize a language model with counter c.

        c: Counter
            Counter of utterance occurances.


        """
        events = {}
        for (s, w) in c.items():
            for i in range(len(s)):
                context = s[0:i]
                event = s[i]
                if context not in events:
                    events[context] = Counter()
                events[context][event] += w
            for context in events:
                z = float(sum(events[context].values()))
                for event in events[context].keys():
                    if z == 0.0:
                        events[context][event] = 0.0
                    else:
                        events[context][event] = float(
                            events[context][event]) / z
            self.cond_probs = copy.deepcopy(events)


class Agent:
    def __init__(self, id, c, k, B_prob=0.5, t_prob=0.5):
        self.id = id
        self.c = c
        self.k = k
        self.B_prob = B_prob
        self.t_prob = t_prob
        self.iter = 0  # track conversations
        self.weighted_lexicon = Counter()   # {str, cnts}
        self.lm = LM()             # internal n-gram language model

    def get_pairs(self):
        return [(s, c) for s, c in self.weighted_lexicon.items() if
                'b' in s.lower()]

    def listen(self, inpts):
        """Update LM

        Parameters
        ----------
        inpt: Counter
            Dict of utterance -> counts

        """
        self.weighted_lexicon += inpts
        self.lm.build_lm(self.weighted_lexicon)
        return self

    def speak(self, n):
        """Provide new LM for updating"""
        utterances = self.weighted_lexicon.keys()
        cnts = np.array(self.weighted_lexicon.values()).astype(float)
        total_mass = np.sum(cnts)
        ps = cnts / total_mass
        discourse = np.random.choice(utterances, n, p=ps)
        # TODO: Incorporate RSA into speaking...
        return Counter(discourse)


if __name__ == '__main__':
    # Initialize agents
    n_agents = 10
    agents = [Agent(str(i), 1, 1) for i in range(n_agents)]
    inpts = Counter(["abc", "abc", "ac", "ac", "atbc"])
    for a in agents:
        a.listen(inpts)

    for _ in range(100):
        # Always change set of speaker / listeners
        speakers = set(
            np.random.choice(range(n_agents), n_agents / 2, replace=False))
        listeners = set(range(n_agents)) - speakers
        for s, l in zip(speakers, listeners):
            # Size of conversation 1-10
            num_utterances = np.random.randint(1, 10)
            data = agents[s].speak(num_utterances)
            agents[l].listen(data)

    import pdb; pdb.set_trace();
