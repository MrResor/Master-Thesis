from algorithms import Ant, Genetic
from decorators import db_handler
from datetime import datetime
from __init__ import argparse, logging, np, sqlite3


class Console_App:
    """ Program Class. Inside it whole utility of program is done. It further
        processes the parameters obtained from parser. First it reads data
        from provided database, then runs chosen algorithm for the given data.
        \n

        Attributes:\n
        d                   -- Distance matrix obtained from database file.\n
        size                -- Number of nodes in database.\n
        algo                -- Holds object of class matching the one chosen
        by user.\n

        Methods:\n
        load_data           -- Loads data from given database.\n
        setup_algorithm     -- Based on the chosen algorithm selects
        parameters from parsed arguments and creates object of the class
        matching the algorithm with the aformentioned arguments.\n
        run                 -- Runs the prepaired algorithm.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """ Initialization of Console_App class, crates logging file, loads
            the data from database and setups algorithm based on the arguments
            passed to command line. Takes Namespace of arguments as parameter.
        """

        curr_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        logging.basicConfig(filename=curr_time + '.log',
                            level=logging.INFO,
                            format='[%(asctime)s] -> {%(levelname)s} '
                            '(%(runtime)s) %(message)s')
        self.load_data(args.path)
        self.setup_algorithm(args)

    @db_handler
    def load_data(self, path) -> None:
        """ Function reading data from databas, such us number of nodes and
            distances between nodes. Takes path do database file as parameter.
        """

        # TODO add possibility of reading data from csv file, due to sqlite3
        # problems on ziemowit cluster
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
        """ Function reading data from databas, such us number of nodes and
            distances between nodes. Takes path do database file as parameter.
        """

        self.algo.run(self.d, self.size)
