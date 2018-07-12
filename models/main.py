"""main.py

Central simulation run file.

Example usage:
    >>> python -m models.main # TODO (BP) fill this in.

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
    parser.add_argument('--out-file', type=str, default='results.csv')
    parser.add_argument('--out-dir', type=str, default='./output/')
    parser.add_argument('--debug-mode', action='store_true',
                        help="Debug mode flag -- don't run multiprocess to "
                             "facilitate debugging.")
    args = parser.parse_args()

    config = ConfigParser.ConfigParser()
    config.read(args.config_path)


    # TODO (BP) This needs to generalize better from config
    # Simulation parameters
    ks = json.loads(config.get(             'AGENT', 'ks'))
    cs = json.loads(config.get(             'AGENT', 'cs'))
    alphas = json.loads(config.get(         'AGENT', 'alphas'))
    num_agents = json.loads(config.get(     'POPULATION', 'num_agents'))
    num_rounds = json.loads(config.get(          'POPULATION', 'num_rounds'))
    num_warmup = int(config.get(          'LANGUAGE', 'num_warmup'))
    single_starting_lexicon = config.get(   'LANGUAGE', 'single_starting_lexicon')
    conversation_size = int(config.get(    'LANGUAGE', 'conversation_size'))
    params = [
        num_agents,
        num_rounds,
        [num_warmup],
        [conversation_size],
        [False],   # variable_conversation_size
        [single_starting_lexicon],
        [[0.5]],   # B_probs
        [[0.5]],   # t_probs
        [[2.0]],   # ks (temp)
        [[2.0]],   # cs (temp)
        [[1.0]],   # alphas (temp)
        [args.out_dir]
    ]
    sim_params_ = it.product(*params)
    sim_params = []
    for i, p in enumerate(sim_params_):
        data = tuple([i, i] + [el for el in p])
        sim_params.append(data)


    t0 = timeit.default_timer()
    if not args.debug_mode:
        logging.info("Running multiprocessing...")
        pool = multiprocessing.Pool(processes=args.num_processes)
        results = [pool.apply(run, (a, b, c, d, e, f, g, h, i, j, k, l, m, n))
                   for a, b, c, d, e, f, g, h, i, j, k, l, m, n in sim_params]
    else:
        logging.info("Running in debug mode...")
        results = []
        for params in sim_params:
            results.append(run(*params))
    logging.info("Runtime:\t{}".format(timeit.default_timer() - t0))

    check_dir(args.out_dir)
    df_results = pd.concat(results)
    df_results.to_csv(os.path.join(args.out_dir, "results.csv"))

