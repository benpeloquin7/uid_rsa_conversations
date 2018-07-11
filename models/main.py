from simulation import *


if __name__ == '__main__':
    import argparse
    import itertools as it
    import multiprocessing


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
    parser.add_argument('--variable-conversation-size',
                        action='store_true',
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


    ks = [1, 1.5, 2]
    cs = [1, 1.5, 2]
    alphas = [1, 4, 10]
    num_agents = [2, 10, 20, 50]
    num_rounds = [10, 50, 100]
    single_starting_lexicon = np.repeat(True, len(ks))
    params = [ks, cs, alphas, num_agents, num_rounds, single_starting_lexicon]
    # for p in  [ks, cs, alphas, num_agents, num_rounds, single_starting_lexicon]:
    #     params.extend(p)


    pool = multiprocessing.Pool(processes=args.num_processes)
    sim_params = it.product(*params)



        # n_sims = 10
        # sim_dfs = []
        # for i in range(n_sims):
        #     print("sim {}/{}".format(i, n_sims))
        #     sim = Simulation(i, i, args.num_agents, args.num_rounds,
        #                      args.num_warm_up,
        #                      args.conversation_size,
        #                      args.variable_conversation_size,
        #                      args.single_starting_lexicon, [args.B_prob],
        #                      [args.t_prob],
        #                      [args.k], [args.c], [args.alpha], args.out_dir)
        #     data = sim.run_simulation()
        #     sim_dfs.append(data)
        #
        # df = join_simulation_data(sim_dfs)
        # sim.that_rate_plot(df, "multiple_sims_that_rate.png")
        # sim.cor_plot(df, "multiple_sims_r.png")