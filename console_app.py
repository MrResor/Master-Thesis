from algorithms import Ant, Genetic
from datetime import datetime
from __init__ import argparse, logging, np
import pandas as pd


class Console_App:
    """ Program Class. Inside it whole utility of program is done. It further
        processes the parameters obtained from parser. First it reads data
        from provided .csv file, then runs chosen algorithm for the given data.
        \n

        Attributes:\n
        d                   -- Distance matrix obtained from csv file.\n
        size                -- Number of nodes in loaded problem.\n
        algo                -- Holds object of class matching the one chosen
        by user.\n

        Methods:\n
        load_data           -- Loads data from given .csv file.\n
        setup_algorithm     -- Based on the chosen algorithm selects
        parameters from parsed arguments and creates object of the class
        matching the algorithm with the aformentioned arguments.\n
        run                 -- Runs the prepaired algorithm.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """ Initialization of Console_App class, crates logging file, loads
            the data from .csv file and setups algorithm based on the arguments
            passed to command line. Takes Namespace of arguments as parameter.
        """

        curr_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        logging.basicConfig(filename=curr_time + '.log',
                            level=logging.INFO,
                            format='[%(asctime)s] -> {%(levelname)s} '
                            '(%(runtime)s) %(message)s')
        self.load_data(args.path)
        self.setup_algorithm(args)

    def load_data(self, path) -> None:
        """ Function reading data from .csv file, such us number of nodes and
            distances between nodes. Takes path to the file as parameter.
        """

        logging.info('Loading data from: %s',
                     path,
                     extra={'runtime': 0})
        df = pd.read_csv(path)
        self.size = df[['from', 'to']].values.max()
        self.d = np.zeros((self.size, self.size))
        for t in df.values:
            self.d[int(t[0])-1, int(t[1])-1] = t[2]
        self.d += self.d.transpose()
        logging.info('Data successfully loaded.',
                     extra={'runtime': 0})

    def setup_algorithm(self, args: argparse.Namespace) -> None:
        """ Setups the algorithm for solving TSP problem based on the parsed
            command line arguments.
        """

        choice = {'ants': Ant, 'genetic': Genetic}
        params_names = {'ants': ['tours', 'alpha', 'beta', 'rho'],
                        'genetic': ['initial_population',
                                    'children_multiplier',
                                    'mutation_probability',
                                    'generation_count'],
                        }
        params = vars(args)
        params = {key: params[key] for key in params_names[args.algorithm]}
        logging.info('Setting up %s algorithm with following parameters:\n %s',
                     args.algorithm,
                     str(params).translate({ord(i): None for i in '{}'}),
                     extra={'runtime': 0})
        self.algo = choice[args.algorithm](params)

    def run(self) -> None:
        """ Function running the prepaired algorithm.
        """

        self.algo.run(self.d, self.size)
