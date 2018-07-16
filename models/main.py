"""main.py

Central simulation run file.

Example usage:
    >>> python -m models.main --config-path "./configs/basic_config.cfg"

"""

import timeit

from simulation import *


def run(id_, seed, num_agents, num_rounds, num_warm_up, conversation_size,
        variable_conversation_size, single_starting_lexicon, B_probs, t_probs,
        ks, cs, alphas, out_dir):
    sim = Simulation(id_, seed, num_agents, num_rounds, num_warm_up,
                     conversation_size, variable_conversation_size,
                     single_starting_lexicon, B_probs, t_probs, ks, cs, alphas,
                     out_dir)
    return sim.run_simulation()

if __name__ == '__main__':
    import argparse
    import ConfigParser
    import json
    import itertools as it
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str,
                        default='./configs/minimal_config.cfg',
                        help='Path to config file [default: ./configs/minimal_config.cfg]')
    parser.add_argument('--num-processes', type=int,
                        default=multiprocessing.cpu_count(),
                        help='Number of cores to use '
                             '[default: multiprocessing.cpu_count()]')
    parser.add_argument('--checkpoints', action='store_true',
                        help='Store checkpointed data [default every 25 simulations.]')
    parser.add_argument('--debug-mode', action='store_true',
                        help="Debug mode flag -- don't run multiprocess to "
                             "facilitate debugging.")
    args = parser.parse_args()

    config = ConfigParser.ConfigParser()
    config.read(args.config_path)

    # Experiment params
    use_RSA = config.getboolean('EXPERIMENT', 'use_RSA')
    num_simulations = config.getint('EXPERIMENT', 'num_simulations')
    num_agents = config.getint('EXPERIMENT', 'num_agents')
    num_rounds = config.getint('EXPERIMENT', 'num_rounds')
    num_warmup = config.getint('EXPERIMENT', 'num_warmup')
    conversation_size = config.getint('EXPERIMENT', 'conversation_size')
    single_starting_lexicon = \
        config.getboolean('EXPERIMENT','single_starting_lexicon')
    permute_conversations = \
        config.getboolean('EXPERIMENT', 'permute_conversations')

    # Agent params
    randomize_ks = config.getboolean('AGENTS', 'randomize_ks')
    randomize_cs = config.getboolean('AGENTS', 'randomize_cs')
    randomize_alphas = config.getboolean('AGENTS', 'randomize_alphas')

    # Language params
    randomize_B_probs = config.getboolean('LANGUAGE', 'randomize_B_probs')
    randomize_t_probs = config.getboolean('LANGUAGE', 'randomize_t_probs')

    # Output
    out_dir = config.get('OUTPUT', 'out_dir')
    out_file = config.get('OUTPUT', 'out_file')
    out_path = os.path.join(out_dir, out_file)


    def get_B_probs():
        if randomize_B_probs:
            B_probs = [np.random.beta(1, 1) for _ in range(num_agents)]
        else:
            B_probs = [config.getfloat('LANGUAGE', 'B_prob')]
        return B_probs


    def get_t_probs():
        if randomize_t_probs:
            t_probs = [np.random.beta(1, 1) for _ in range(num_agents)]
        else:
            t_probs = [config.getfloat('LANGUAGE', 't_prob')]
        return t_probs


    def get_ks():
        if randomize_ks:
            ks = [np.random.uniform(1, 2) for _ in range(num_agents)]
        else:
            ks = [config.getfloat('AGENTS', 'ks')]
        return ks


    def get_cs():
        if randomize_ks:
            cs = [np.random.uniform(0, 2) for _ in range(num_agents)]
        else:
            cs = [config.getfloat('AGENTS', 'cs')]
        return cs


    def get_alphas():
        if randomize_alphas:
            alphas = [np.random.uniform(1., 6.) \
                      for _ in range(num_agents)]
        else:
            alphas = [config.getfloat('AGENTS', 'alphas')]
        return alphas



    # Outputs dir
    check_dir(out_dir)

    pbar = tqdm.tqdm(total=num_simulations, position=0)
    results = []
    for i in range(num_simulations):
        B_probs = get_B_probs()
        t_probs = get_t_probs()
        ks = get_ks()
        cs = get_cs()
        alphas = get_alphas()

        sim = Simulation(i, i, num_agents, num_rounds, num_warmup,
                         conversation_size, permute_conversations,
                         single_starting_lexicon, B_probs, t_probs, ks, cs,
                         alphas, use_RSA, out_file)

        results.append(sim.run_simulation())
        if args.checkpoints and i % 25 == 0:
            df_results = pd.concat(results)
            df_results.to_csv(os.path.join(out_dir, out_file[:-4] + "-{}-runs.csv".format(i)))
        pbar.update()
    pbar.close()
    df_results = pd.concat(results)
    df_results.to_csv(out_path)
