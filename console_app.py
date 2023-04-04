from algorithms import Ant, Genetic


class Console_App:
    def __init__(self, args):
        print('Hemlo from console')
        choice = {'ants': Ant, 'genetic': Genetic}
        params_names = {'ants': ['tours', 'alpha', 'beta', 'rho']}
        print(args)
        params = vars(args)
        params = {key: params[key] for key in params_names[args.algorithm]}
        self.algo = choice[args.algorithm](params)
