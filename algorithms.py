class Genetic:
    def __init__(self):
        print('Genetic')


class Ant:
    def __init__(self, console_params):
        params = {'tours': 40, 'alpha': 2, 'beta': 1, 'rho': 0.5}
        params = {key: (params[key] if console_params[key] is
                        None else console_params[key]) for key in params}
        print('Ants', params)
