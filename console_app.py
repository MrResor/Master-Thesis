from algorithms import Ant, Genetic
from decorators import db_handler
from datetime import datetime
from __init__ import argparse, logging, np, sqlite3


class Console_App:
    def __init__(self, args: argparse.Namespace) -> None:
        curr_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        logging.basicConfig(filename=curr_time + '.log',
                            level=logging.INFO,
                            format='[%(asctime)s] -> {%(levelname)s} '
                            '(%(runtime)s) %(message)s')
        self.load_data(args.path)
        self.setup_algorithm(args)

    # TODO decorator to catch errors
    @db_handler
    def load_data(self, path) -> None:
        logging.info('Loading Database: %s',
                     path,
                     extra={'runtime': 0})
        # TODO sprawdzić jakoś format bazy danych
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute('SELECT * FROM Distance')
        temp = cur.fetchall()
        cur.execute('SELECT COUNT(*) FROM Cities')
        self.size = cur.fetchone()[0]
        cur.close()
        self.d = np.zeros((self.size, self.size))
        for t in temp:
            self.d[t[0]-1][t[1]-1] = t[2]
        self.d += self.d.transpose()
        logging.info('Database successfully loaded.',
                     extra={'runtime': 0})

    def setup_algorithm(self, args) -> None:
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
        self.algo.run(self.d, self.size)
