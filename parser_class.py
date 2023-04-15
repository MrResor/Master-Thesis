import argparse
import os


def frange(min: float, max: float):
    """ Return function handle of an argument type function for
        ArgumentParser checking a float range: min <= arg <= max.\n
        min - Minimum acceptable argument.\n
        max - Maximum acceptable argument.
    """

    def float_range_checker(arg: float) -> float:
        """ New Type function for argparse - a float within predefined range.
        """

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(
                'Must be a floating point number.')
        if f < min or f > max:
            raise argparse.ArgumentTypeError(
                f'Must be in range [{min} .. {max}].')
        return f

    return float_range_checker


def positive(num_type: type):
    """ Return function handle of an argument type function for
        ArgumentParser checking if number is of required type and
        if it is positive.\n
        num_type - numeric type of the variable.
    """
    def positive_checker(arg) -> num_type:
        try:
            f = num_type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f'Must be a{"n" if num_type.__name__ == "int" else ""} '
                f'{num_type.__name__}.')
        if f < 0:
            raise argparse.ArgumentTypeError(
                f'Must be positive {num_type.__name__}.')
        return f

    return positive_checker


def is_path(path):
    if not os.path.isabs(path):
        path = os.path.realpath(__file__).split("\\")[0:-1] + [path]
        path = "\\".join(path)
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError('Given file does not exist')
    return path


class Parser:
    def __init__(self) -> None:
        self.__p = argparse.ArgumentParser(
            prog='TSP solver',
            description='Program for solving of TSP using different methods',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False
        )

        self.__p.add_argument('-c', '--console',
                              action='store_true',
                              default=False,
                              help='flag for running program in console mode.')
        known, _ = self.__p.parse_known_args()
        self.__p.add_argument('-h', '--help', action='help',
                              default=argparse.SUPPRESS,
                              help='Show this help message and exit.')
        self.__p.add_argument('--path', required=known.console, metavar='',
                              type=is_path, help='Path to database file.')
        sub_p = self.__p.add_subparsers(title='algorithms',
                                        dest="algorithm",
                                        help='Choice of algorithm.',
                                        required=known.console)
        self.__ants_params(sub_p)
        self.__genetic_params(sub_p)

    def __ants_params(self, sub_p) -> None:
        ants = sub_p.add_parser(
            'ants', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False)
        ants.add_argument('-h', '--help', action='help',
                          default=argparse.SUPPRESS,
                          help='\b\b\b\bShow this help message and exit.')
        ants.add_argument('-t', '--tours', type=positive(int), default=40,
                          metavar='\b', help='Number of tours.')
        ants.add_argument('-a', '--alpha', type=positive(float), default=1,
                          metavar='\b', help='Variable controling impact of '
                          'pheromones between nodes.')
        ants.add_argument('-b', '--beta', type=positive(float), default=2,
                          metavar='\b', help='Variable controling impact of '
                          'distance between nodes.')
        ants.add_argument('-p', '--rho', type=frange(0, 1), default=0.5,
                          metavar='\b', help='Variable controling pheromone '
                          'evaporation.')

    def __genetic_params(self, sub_p) -> None:
        genetic = sub_p.add_parser(
            'genetic', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False)
        genetic.add_argument('-h', '--help', action='help',
                             default=argparse.SUPPRESS,
                             help='Show this help message and exit.')
        genetic.add_argument('-P', '--initial-population', type=int,
                             default=250, metavar='\b',
                             help='Size of initial population.')
        genetic.add_argument('-n', '--children-multiplier', type=float,
                             default=0.8, metavar='\b',
                             help='Multiplier for children populaion size '
                             '(P * n)')
        genetic.add_argument('-p', '--mutation-probability', type=float,
                             default=0.5, metavar='\b',
                             help='Mutation rate for offsprings.')
        genetic.add_argument('-T', '--generation-count', type=int,
                             default=500, metavar='\b',
                             help='Number of generations.')

    def parse(self) -> argparse.Namespace:
        return self.__p.parse_args()
