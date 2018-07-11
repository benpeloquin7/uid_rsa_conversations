"""Speaker / listener agent"""
import argparse
from collections import Counter, defaultdict
import copy
import itertools as it
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import tqdm

logging.getLogger().setLevel(logging.INFO)


class UtteranceException(Exception):
    pass


class LMException(Exception):
    pass


class AgentException(Exception):
    pass


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_dict(ks, vs):
    return {k: v for k, v in zip(ks, vs)}


def merge_dicts(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


def strings():
    A = ["a", "A"]
    B = ["b", "B"]
    C = ["c", "C"]
    T = ["t", ""]
    EOS = "$"
    strings = []
    for a12 in it.product(A, A):
        for c12 in it.product(C, C):
            strings.append("".join(list(a12) + list(c12)))
            for b12 in list(it.product(B, B)):
                for t in T:
                    strings.append(
                        "".join(list(a12) + [t] + list(b12) + list(c12)))
                    # pairs.append(thispair)
    return strings


class LM:
    def __init__(self):
        self.all_strings = strings()
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
        return events

    def learn_lm(self, c):
        self.cond_probs = self.build_lm(c)
        return self

    def build_lm_ignoring_that(self, c):
        """


        """
        # Remove optionals
        pairs = get_pairs(c)  # get RC pairs
        no_t_strs = []
        RCs = set({})
        for pair in pairs:
            (alt1, alt1_weight), (alt2, alt2_weight) = pair

            RCs |= set([alt1, alt2])
            total_weight = np.sum([alt1_weight, alt2_weight])
            no_t_strs.append((rm_optional_t(alt1),
                              total_weight))  # can use either alternative here

        # Build lm
        noRCs = set(c.keys()) - set(RCs)
        d = [(utt, weight) for (utt, weight) in c.items() if utt in noRCs]
        d.extend(no_t_strs)
        new_c = {utt: cnt for utt, cnt in d}
        lm = self.build_lm(new_c)
        return lm

    def score_string(self, s, k):
        """High scores are bad"""
        result = 0
        for i in range(len(s)):
            p = self.cond_probs[s[0:i]][s[i]]
            if p == 0:
                return float('inf')
            else:
                result += (-math.log(p)) ** k
        return result


# Utils

def generate_alt(x):
    if contains_optional_t(x):
        return re.sub('t', '', x)
    else:
        return x[0] + 't' + x[1:]


def contains_optional_t(s):
    return 't' in s


def rm_optional_t(s):
    return re.sub("t", "", s)


def fill_pair(pair):
    if len(pair) != 2:  # Generate alt with cnts 0
        (alt1, alt1_weight) = pair[0]
        missing_alt = generate_alt(alt1)
        missing_alt_weight = 0
        return ((alt1, alt1_weight), (missing_alt, missing_alt_weight))
    return pair


def get_pairs(cntr, check=lambda x: 'b' in x.lower()):
    """

    Returns a list of tuples of utterance, utterance weight alternatives.

    [
        ((alt1a, alt1a_weight), (alt1b, alt1b_weight)),
        ...
        ((alt2a, alt2a_weight), (alt2b, alt2b_weight)),
    ]

    """
    RCs = [k for k in cntr.keys() if check(k)]
    d = defaultdict(list)
    for k in RCs:
        # if not isinstance(k, str):
        #     import pdb; pdb.set_trace();
        d[rm_optional_t(k)].append((k, cntr[k]))
    return [fill_pair(p) for p in d.values()]


class Agent:
    def __init__(self, id, c, k, B_prob=0.5, t_prob=0.5, alpha=1):
        self.id = id
        self.c = c
        self.k = k
        self.B_prob = B_prob
        self.t_prob = t_prob
        self.alpha = alpha
        self.iter = 0  # track conversations
        self.weighted_lexicon = Counter()  # {str, cnts}
        self.lm = {}  # internal n-gram language model
        self.lm_no_t = {}
        self.n_spoken = 0
        self.n_listen = 0
        self.all_strigs = strings()

    def listen(self, inpts):
        """Update LM

        Parameters
        ----------
        inpt: Counter
            Dict of utterance -> counts

        """
        self.n_listen += 1
        self.weighted_lexicon += inpts
        self.lm = LM().build_lm(self.weighted_lexicon)
        self.lm_no_t = LM().build_lm_ignoring_that(self.weighted_lexicon)
        return self

    def score_string(self, s):
        """k: UID exponent; c: string cost"""
        return self._score_string(s, self.k) + self.c * len(s)

    def _score_string(self, s, k):
        """High scores are bad"""
        result = 0
        for i in range(len(s)):
            p = self.lm[s[0:i]][s[i]]  # we use full LM with optional (t)
            if p == 0:
                return float('inf')
            else:
                result += (-math.log(p)) ** k
        return result

    def rsa_score(self, utt):
        return math.exp(-self.alpha * self.score_string(utt))

    def reweight_pair(self, utts, weights):
        """The RSA S1 speaker model on the alternative utterance set consisting of
        the strings in the string set, all for the same meaning.
        Reallocate total_mass among them"""
        utt1, utt2 = utts
        total_weight = np.sum(weights)
        score1 = self.rsa_score(utt1)
        score2 = self.rsa_score(utt2)
        total = score1 + score2
        new_weights = (np.array([score1, score2]) / total) * total_weight
        return (utt1, new_weights[0]), (utt2, new_weights[1])

    def S1(self):
        """
        Returns a list of tuples of utterance, utterance weight alternatives.

        [
            ((alt1a, alt1a_weight), (alt1b, alt1b_weight)),
            ...
            ((alt2a, alt2a_weight), (alt2b, alt2b_weight)),
        ]
        """
        pairs = get_pairs(self.weighted_lexicon)
        d_reweighted = []
        for ((alt1_utt, alt1_weight), (alt2_utt, alt2_weight)) in pairs:
            utts = [alt1_utt, alt2_utt]
            weights = [alt1_weight, alt2_weight]
            d_reweighted.append(self.reweight_pair(utts, weights))
        return d_reweighted

    def speak(self, n):
        """Provide new LM for updating"""
        self.n_spoken += 1  # Track speaking occurrences

        # RSA re-weighting
        weighted_pairs = self.S1()
        speaking_lex = copy.deepcopy(self.weighted_lexicon)
        for (
                (alt1_utt, alt1_weight),
                (alt2_utt, alt2_weight)) in weighted_pairs:
            speaking_lex[alt1_utt] = alt1_weight
            speaking_lex[alt2_utt] = alt2_weight

        utterances = speaking_lex.keys()
        cnts = np.array(speaking_lex.values()).astype(float)
        total_mass = np.sum(cnts)
        ps = cnts / total_mass
        discourse = np.random.choice(utterances, n, p=ps)
        return Counter(discourse)


# Corpus analysis code

def create_corpus(speakers, num_utterances_per_speaker=100):
    corpus = Counter()
    for speaker in speakers:
        corpus += speaker.speak(num_utterances_per_speaker)
    return corpus


def record_nextword_prob_and_that_use(lm_no_t, weighted_string_pairs):
    """This function will not generalize well, as written -- it is very
    specific to the length-2 context, second-in-pair-is-t-omitted case"""
    nextword_probs = []
    that_probs = []
    for x in weighted_string_pairs:
        contains_t_pos = 0 if contains_optional_t(x[0][0]) else 1
        contains_t = x[contains_t_pos]
        no_contains_t = x[abs(contains_t_pos - 1)]
        nextword_prob = lm_no_t[no_contains_t[0][0:1]][no_contains_t[0][1]]
        nextword_probs.append(nextword_prob)
        that_prob = contains_t[1] / (contains_t[1] + no_contains_t[1])
        that_probs.append(that_prob)
    return (nextword_probs, that_probs)


def overall_that_rate(wsp):
    Z = 0.0
    q = 0.0
    for p in wsp:
        q += p[0][1]
        Z += p[0][1] + p[1][1]
    return q / Z


def get_r_and_that_rate(agents):
    corpus = create_corpus(agents)
    pairs = get_pairs(corpus)
    # corpus_lm = LM().build_lm(corpus)
    corpus_lm_no_t = LM().build_lm_ignoring_that(corpus)
    next_probs = record_nextword_prob_and_that_use(corpus_lm_no_t, pairs)
    that_rate = overall_that_rate(pairs)
    r = np.corrcoef(next_probs[0], next_probs[1])[0, 1]
    return r, that_rate


if __name__ == '__main__':
    # Initialize agents
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-agents', type=int, default=10,
                        help='Number of agents [default: 10].')
    parser.add_argument('--num-rounds', type=int, default=10,
                        help='Number of conversations [default: 10]')
    parser.add_argument('--num-warm-up', type=int, default=100,
                        help='Data pre conversation [default: 100]')
    parser.add_argument('--conversation-size', type=int, default=10,
                        help='Number of conversations [default: 10]')
    parser.add_argument('--variable-conversation-size', action='store_true',
                        help='Make conversations variable in size.')
    parser.add_argument('--single-starting-lexicon', action='store_true',
                        help='Each agent starts with same language distribution.')
    parser.add_argument('--out-file', type=str, default='results.csv')
    parser.add_argument('--out-dir', type=str, default='./output/')
    parser.add_argument('--debug-mode', action='store_true',
                        help="Debug mode flag -- don't run multiprocess to "
                             "facilitate debugging.")
    args = parser.parse_args()

    agents = [Agent(str(i), 1, 2, alpha=4) for i in range(args.num_agents)]
    all_strs = strings()

    # Initialize lexicons
    inpts = [Counter(np.random.choice(all_strs, args.num_warm_up)) \
             for i in range(len(agents))]
    for i in inpts:
        for s in all_strs:
            if s not in i:
                i[s] = 0

    # Initialize Agents
    for a, i in zip(agents, range(len(inpts))):
        if args.single_starting_lexicon:
            logging.info("Each agent starts with same lexicon")
            a.listen(inpts[0])
        else:
            logging.info("Each agent starts with unique lexicon")
            a.listen(inpts[i])

    data = []
    for i in tqdm.tqdm(range(args.num_rounds)):
        # Always change set of speaker / listeners
        speakers = set(
            np.random.choice(range(args.num_agents), args.num_agents / 2,
                             replace=False))
        listeners = set(range(args.num_agents)) - speakers
        for s, l in zip(speakers, listeners):
            if args.variable_conversation_size:
                num_utterances = np.random.randint(1, args.conversation_size)
                utterances = agents[s].speak(num_utterances)
            else:
                utterances = agents[s].speak(args.conversation_size)
            agents[l].listen(utterances)

        r, that_rate = get_r_and_that_rate(agents)
        data.append({
            'round': i,
            'r': r,
            'that_rate': that_rate
        })

    # Save data
    check_dir(args.out_dir)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(args.out_dir, "results.csv"))

    # Plots
    import seaborn as sns

    sns.set_style('whitegrid')
    logging.info("Generating r plot...")
    p = sns.pointplot(x="round", y="r", data=df)
    p.figure.savefig(os.path.join(args.out_dir, "r.png"))
    p.figure.clf()
    logging.info("Generating that_rate plot...")
    p = sns.pointplot(x="round", y="that_rate", data=df)
    p.figure.savefig(os.path.join(args.out_dir, "that_rate.png"))
