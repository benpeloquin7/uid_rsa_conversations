"""simulation.py

Conversational UID-RSA modeling code.

Example usage:
    >>> python -m models.agent --num-agents 2 --num-rounds 500 --num-warm-up 100 --conversation-size 20 --k 2.0 --c 1.2 --alpha 2.0

"""
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
import random
import re
import seaborn as sns
import tqdm

sns.set_style('whitegrid')
logging.getLogger().setLevel(logging.INFO)


class LMException(Exception):
    pass


class AgentException(Exception):
    pass


class SimulationException(Exception):
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


def join_simulation_data(frames):
    """
    Parameters
    ----------
    frames: list of pd.DataFrame
        Data frames we'd like to concatenate (rbind)

    Returns
    -------
    pd.DataFrmae
        Merged dataframe.

    """
    return pd.concat(frames)


def initial_weight_strings(B_prob=0.5, t_prob=0.5):
    """Initialize weights (probabilities) for a set of strings & string pairs

    Parameters
    ----------
    B_prob: float
        RC probability.
    t_prob: float
        That probability.

    Returns
    -------
    list of tuples
        (utterance, inititial prob)
    """
    all_strings = {utt: 0 for utt in strings()}
    rc_pairs, no_rc_pairs = partition_RC_no_RC(all_strings)
    ws = np.random.dirichlet([1.0] * len(no_rc_pairs))
    no_rc_weights = [(1 - B_prob) * w for w in ws]
    utterances = []
    weights = []
    # no RC
    for ((s, cnt), w) in zip(no_rc_pairs, no_rc_weights):
        utterances.append(s)
        weights.append(w)
    # RC
    rc_weights = np.random.dirichlet([1.0] * len(rc_pairs))
    for (rc_pair, w) in zip(rc_pairs, rc_weights):
        (s1, _), (s2, _) = rc_pair
        utterances.extend([s1, s2])
        weights.extend([t_prob * B_prob * w, (1 - t_prob) * B_prob * w])

    # Checks
    assert len(utterances) == len(weights)
    assert (abs(np.sum(weights) - 1) < 1e-10)
    return utterances, weights


def strings():
    A = ["a", "A"]
    B = ["b", "B"]
    C = ["c", "C"]
    T = ["t", ""]
    EOS = "$"
    strings = []
    for a12 in it.product(A, A):
        for c12 in it.product(C, C):
            strings.append("".join(list(a12) + list(c12) + [EOS]))
            for b12 in list(it.product(B, B)):
                for t in T:
                    strings.append(
                        "".join(
                            list(a12) + [t] + list(b12) + list(c12) + [EOS]))
    return strings


def create_corpus(speakers, num_utterances_per_speaker=1000):
    corpus = Counter()
    for speaker in speakers:
        corpus += speaker.speak(num_utterances_per_speaker)
    return corpus


def generate_alt(x):
    if contains_optional_t(x):
        return re.sub('t', '', x)
    else:
        return x[:2] + 't' + x[2:]


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


def partition_RC_no_RC(lex, check=lambda x: 'b' in x.lower()):
    """Divide lexicon into wiethed RC pairs and non-RC utterances.


    Parameters
    ----------
    lex: collections.Counter
        Mapping of utterance to counts.
    check: fn
        RC criteria.

    Returns
    -------
    tuple of list
        (
            # RCs
            [
                ((alt1a, alt1a_weight), (alt1b, alt1b_weight)),
                ...
                ((alt2a, alt2a_weight), (alt2b, alt2b_weight))
            ],
            # non-RCs
            [
                (utt1, utt1_weight), (utt2, utt2_weight), (utt3, utt3_weight)
            ]
        )

    """
    # RC pairs
    RCs = [k for k in lex.keys() if check(k)]
    d = defaultdict(list)
    for k in RCs:
        d[rm_optional_t(k)].append((k, lex[k]))
    weighted_RCs = [fill_pair(p) for p in d.values()]

    # Non-RC utterrances
    noRCs = set(lex.keys()) - set(RCs)
    weighted_no_RCs = [(u, cnt) for u, cnt in lex.items() if u in noRCs]
    return weighted_RCs, weighted_no_RCs


def record_nextword_prob_and_that_use(lm_no_t, weighted_string_pairs):
    """This function will not generalize well, as written -- it is very
    specific to the length-2 context, second-in-pair-is-t-omitted case"""
    nextword_probs = []
    that_probs = []
    for x in weighted_string_pairs:
        contains_t_pos = 0 if contains_optional_t(x[0][0]) else 1
        contains_t = x[contains_t_pos]
        no_contains_t = x[abs(contains_t_pos - 1)]

        nextword_prob = lm_no_t[no_contains_t[0][0:2]][no_contains_t[0][2]]
        nextword_probs.append(nextword_prob)
        that_prob = contains_t[1] / (contains_t[1] + no_contains_t[1])
        that_probs.append(that_prob)
    return (nextword_probs, that_probs)


def overall_that_rate(wsp):
    Z = 0.0
    q = 0.0
    for p in wsp:
        contains_t_pos = 0 if contains_optional_t(p[0][0]) else 1
        contains_t = p[contains_t_pos]
        no_contains_t = p[abs(contains_t_pos - 1)]
        q += contains_t[1]
        Z += contains_t[1] + no_contains_t[1]
    if Z == 0:
        return 0.0
    return q / Z


def get_r_and_that_rate(agents):
    corpus = create_corpus(agents)
    rc_pairs, _ = partition_RC_no_RC(corpus)
    corpus_lm_no_t = LM().build_lm_ignoring_that(corpus)
    next_probs = record_nextword_prob_and_that_use(corpus_lm_no_t, rc_pairs)
    that_rate = overall_that_rate(rc_pairs)
    r = np.corrcoef(next_probs[0], next_probs[1])[0, 1]
    return r, that_rate,


def create_initial_input(B_prob, t_prob, n):
    all_utterances = strings()
    init_utts, init_weights = initial_weight_strings(B_prob, t_prob)
    d = Counter(np.random.choice(init_utts, n, p=init_weights))
    # Fill un-observerved utterances (lexicon is fully known)
    for u in all_utterances:
        if u not in d.keys():
            d[u] = 0
    return d


class LM:
    """Language modeling."""
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
        rc_pairs, no_Rcs = partition_RC_no_RC(c)  # get RC pairs
        no_t_strs = []
        RCs = set({})
        for pair in rc_pairs:
            (alt1, alt1_weight), (alt2, alt2_weight) = pair
            RCs |= set([alt1, alt2])
            total_weight = np.sum([alt1_weight, alt2_weight])
            no_t_strs.append((rm_optional_t(alt1),
                              total_weight))  # can use either alternative here

        # Build lm
        # noRCs = set(c.keys()) - set(RCs)
        # d = [(utt, weight) for (utt, weight) in c.items() if utt in noRCs]
        no_Rcs.extend(no_t_strs)
        new_c = {utt: cnt for utt, cnt in no_Rcs}
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

class Agent:
    """Speaker/Listener agents."""
    def __init__(self, id, k, c, alpha=1):
        self.id = id
        self.k = k
        self.c = c
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
        if total != 0:
            new_weights = (np.array([score1, score2]) / total) * total_weight
        else:
            new_weights = np.array([0., 0.])
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
        rc_pairs, _ = partition_RC_no_RC(self.weighted_lexicon)
        d_reweighted = []
        for ((alt1_utt, alt1_weight), (alt2_utt, alt2_weight)) in rc_pairs:
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

        utterances = list(speaking_lex.keys())
        cnts = np.array(list(speaking_lex.values())).astype(float)
        total_mass = np.sum(cnts)
        ps = cnts / total_mass
        discourse = np.random.choice(utterances, n, p=ps)
        return Counter(discourse)

class Simulation:
    """Simulation.

    Parameters
    ----------
    id: str
        Unique identifier for data logging.
    seed: int
        random seed.
    num_agents: int
        Number of agents involved in simulation.
    num_rounds: int
        Number of converation rounds.
    num_warm_up: int
        Size of speaker language prior input.
    conversation_size: int
        Number of utterances per conversation.
    variable_conversation_size: bool
        Vary the conversation size with uniform choice
        from [1, conversation_size]
    single_starting_lexicon: bool
        If true, each agent starts with a different lexicon.
    B_probs: array of floats
        List of B_probs (should be same length as num_agents).
    t_probs: array of floats
        List of t_prob (should be same length as num_agents).
    ks: array of floats
        Uniformity parameters in UID cost.
    cs: array of floats
        Length penalty parameter in UID cost.
    alphas: float
        RSA rationality parameter.
    out_dir: str
        File path to out directory.


    """
    def __init__(self, id, seed, num_agents, num_rounds, num_warm_up,
                 conversation_size, variable_conversation_size,
                 single_starting_lexicon, B_probs, t_probs, ks, cs, alphas,
                 out_dir="./outputs/"):
        self.id = id
        self.seed = seed
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.num_warm_up = num_warm_up
        self.conversation_size = conversation_size
        self.variable_conversation_size = variable_conversation_size
        self.single_starting_lexicon = single_starting_lexicon
        self.B_probs = B_probs
        self.t_probs = t_probs
        self.ks = ks
        self.cs = cs
        self.alphas = alphas
        self.out_dir = out_dir

        # Build sim
        self.agents = self.build_agents()
        self.init_inpts = self.build_init_inputs()
        self.load_agent_lexicons(self.agents, self.init_inpts, single_starting_lexicon)
        self.sims_run = 0
        self.single_agent_run = self.num_agents == 1

    def set_random_state_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def build_agents(self):
        """Create a set of agents of size num_agents."""
        if len(self.ks) == 1:
            self.ks = np.repeat(self.ks, self.num_agents)
        if len(self.cs) == 1:
            self.cs = np.repeat(self.cs, self.num_agents)
        if len(self.alphas) == 1:
            self.alphas = np.repeat(self.alphas, self.num_agents)
        if not len({self.num_agents,
                    len(self.ks), len(self.cs), len(self.alphas)}) == 1:
            raise SimulationException("Mismatch in ks, cs or alphas size"
                                      " should be: {}".format(self.num_agents))

        agents = []
        for i, k, c, alpha in zip(range(self.num_agents), self.ks, self.cs,
                                  self.alphas):
            agents.append(Agent(str(i), k, c, alpha))

        return agents

    def build_init_inputs(self):
        """Create inputs for agents"""
        if len(self.B_probs) == 1:
            self.B_probs = np.repeat(self.B_probs, self.num_agents)
        if len(self.t_probs) == 1:
            self.t_probs = np.repeat(self.t_probs, self.num_agents)
        if not len({self.num_agents,
                    len(self.B_probs), len(self.t_probs)}) == 1:
            raise SimulationException("Mismatch in B_probs t_probs size"
                                      " should be: {}".format(self.num_agents))

        inpts = []
        for _, B_prob, t_prob in zip(range(self.num_agents), self.B_probs, self.t_probs):
            inpt = create_initial_input(B_prob, t_prob, self.num_warm_up)
            inpts.append(inpt)
        return inpts

    def load_agent_lexicons(self, agents, inpts, single_starting_lex):
        if single_starting_lex:
            logging.info("Using shared lexicon")
            shared_lex = inpts[0]
            for a, i in zip(agents, range(len(inpts))):
                a.listen(shared_lex)
        else:
            logging.info("All agents receive different lexicon")
            for a, i in zip(agents, range(len(inpts))):
                a.listen(inpts[i])

    def run_simulation(self):
        """
        Parameters
        ----------

        Returns
        -------
        pd.DataFrame
            Results.

        """
        data = []
        logging.info("Running {} simulations...".format(self.num_rounds))
        for i in tqdm.tqdm(range(self.num_rounds)):
            # Single speaker simulation
            if self.single_agent_run:
                speakers = [0]
                listeners = [0]
            # Speaker / Listener simulations
            else:
                speakers = set(np.random.choice(range(self.num_agents),
                                                int(self.num_agents / 2),
                                                replace=False))
                listeners = set(range(self.num_agents)) - speakers
            for s, l in zip(speakers, listeners):
                if self.variable_conversation_size:
                    num_utterances = np.random.randint(1, self.conversation_size)
                    utterances = self.agents[s].speak(num_utterances)
                else:
                    utterances = self.agents[s].speak(self.conversation_size)
                self.agents[l].listen(utterances)

            r, that_rate = get_r_and_that_rate(self.agents)
            data.append({
                'round': i,
                'r': r,
                'that_rate': that_rate,
                'n_agents': self.num_agents,
                'n_rounds': self.num_rounds,
                'n_warmup': self.num_warm_up,
                'sim_number': self.sims_run,
                'id': self.id
            })
        self.sims_run += 1
        return pd.DataFrame(data)

    def write_csv(self, df, name=None):
        check_dir(self.out_dir)
        if not name:
            df.to_csv(
                os.path.join(
                    self.out_dir, "results-id-{}-sims-{}.csv".format(self.id, self.sims_run)))
        else:
            df.to_csv(os.path.join(self.out_dir, name))

    def cor_plot(self, df, name=None):
        check_dir(self.out_dir)
        logging.info("Generating r plot...")
        p = sns.pointplot(x="round", y="r", hue="id", data=df)
        if not name:
            p.figure.savefig(
                os.path.join(args.out_dir, "r-id-{}-sim-{}.png".format(self.id, self.sims_run)))
        else:
            p.figure.savefig(os.path.join(args.out_dir, name))
        p.figure.clf()

    def that_rate_plot(self, df, name=None):
        check_dir(self.out_dir)
        logging.info("Generating that_rate plot...")
        p = sns.pointplot(x="round", y="that_rate", hue="id", data=df)
        if not name:
            p.figure.savefig(os.path.join(args.out_dir,
                                          "that_rate-id-{}-sim-{}.png".format(self.id, self.sims_run)))
        else:
            p.figure.savefig(os.path.join(args.out_dir, name))
        p.figure.clf()

    def simulation_data(self):
        data = {
            "id": self.id,
            "num_agents": self.num_agents,
            "num_rounds": self.num_rounds,
            "num_warm_up": self.num_warm_up,
            "conversation_size": self.conversation_size,
            "variable_conversation_size": self.variable_conversation_size,
            "single_starting_lexicon": self.single_starting_lexicon,
            "sim": self.variable_conversation_size,
            "B_probs": self.B_probs,
            "t_probs": self.t_probs,
            "ks": self.ks,
            "cs": self.cs,
            "alphas": self.alphas,
            "sims_run": self.sims_run,
            "seed": self.seed
        }
        return data

    def output_sim_log(self):
        check_dir(self.out_dir)
        p = os.path.join(self.out_dir, "sim-{}.log".format(self.id))
        with open(p, 'wb') as fp:
            sim_data = self.simulation_data()
            header = ','.join(sim_data.keys()) + '\n'
            data = ','.join(map(str, sim_data.values()))
            fp.write(header)
            fp.write(data)



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
    parser.add_argument('--B-prob', type=float, default=0.5,
                        help='Initial RC probability [default: 0.5]')
    parser.add_argument('--t-prob', type=float, default=0.5,
                        help='Initial that-rate [default=: 0.5]')
    parser.add_argument('--k', type=float, default=1.,
                        help='Uniformity cost [default=: 1.]')
    parser.add_argument('--c', type=float, default=1.,
                        help='Length cost [default=: 1.]')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='RSA rationality parameter [default=: 1.]')
    parser.add_argument('--out-file', type=str, default='results.csv')
    parser.add_argument('--out-dir', type=str, default='./output/')
    parser.add_argument('--debug-mode', action='store_true',
                        help="Debug mode flag -- don't run multiprocess to "
                             "facilitate debugging.")
    args = parser.parse_args()

    B_prob = [0.5]
    t_prob = [0.5]
    k = [1.2]
    c = [1.0]
    alpha = [10]
    sim = Simulation(0, 0, 1, args.num_rounds, args.num_warm_up,
                     args.conversation_size, False, True, B_prob, t_prob,
                     k, c, alpha, args.out_file)

    data = []
    pbar = tqdm.tqdm(total=args.num_rounds, position=0)
    for i in range(args.num_rounds):
        res = sim.run_simulation()
        data.append(res)
        pbar.update()
    pbar.close()

    df_results = pd.concat(data)
    df_results.to_csv(args.out_file)

    # results = []
    # n_sims = 2
    #
    #
    # ks = [float(k)/10 for k in range(10, 21)]
    # cs = [float(k) / 10 for k in range(10, 21)]
    # total = len(list(it.product(ks, cs))) * n_sims
    #
    # pbar = tqdm.tqdm(total=total, position=0)
    #
    # id_ = 0
    # for k in ks:
    #     k = [k]
    #     for c in cs:
    #         c = [c]
    #         B_prob = [np.random.beta(1, 1)]
    #         t_prob = [np.random.beta(1, 1)]
    #         logging.info("B_prob:{}, t_prob:{}".format(B_prob, t_prob))
    #         sim = Simulation(id_, id_, 1, args.num_rounds, args.num_warm_up,
    #                          args.conversation_size, False, True, B_prob, t_prob,
    #                          k, c, [1.], args.out_file)
    #         id_ += 1
    #         for i in range(n_sims):
    #             data = sim.run_simulation()
    #             results.append(data)
    #             pbar.update()
    #
    #
    # pbar.close()
    # df_results = pd.concat(results)
    # df_results.to_csv(args.out_file)

    # n_sims = 100
    # sim_dfs = []
    # for i in range(n_sims):
    #     print("sim {}/{}".format(i, n_sims))
    #     sim = Simulation(i, i, args.num_agents, args.num_rounds, args.num_warm_up,
    #                args.conversation_size, args.variable_conversation_size,
    #                args.single_starting_lexicon, [args.B_prob], [args.t_prob],
    #                [args.k], [args.c], [args.alpha], args.out_dir)
    #     data = sim.run_simulation()
    #     sim_dfs.append(data)
    #
    #
    # df = join_simulation_data(sim_dfs)
    # sim.that_rate_plot(df, "multiple_sims_that_rate.png")
    # sim.cor_plot(df, "multiple_sims_r.png")

