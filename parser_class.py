from __init__ import argparse
import os
from typing import Callable


def frange(min: float, max: float) -> Callable[..., float]:
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
                'Must be a floating point number.'
            )
        if f < min or f > max:
            raise argparse.ArgumentTypeError(
                f'Must be in range [{min} .. {max}].'
            )
        return f

    return float_range_checker


def positive(num_type: type) -> Callable[..., type]:
    """ Return function handle of an argument type function for
        ArgumentParser checking if number is of required type and
        if it is positive.\n
        num_type - numeric type of the variable.
    """

    def positive_checker(arg) -> num_type:
        """ New Type function for argparse - a positive number of passed
            num_type.
        """

        try:
            f = num_type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f'Must be a{"n" if num_type.__name__ == "int" else ""} '
                f'{num_type.__name__}.'
            )
        if f < 0:
            raise argparse.ArgumentTypeError(
                f'Must be positive {num_type.__name__}.'
            )
        return f

    return positive_checker


def is_path(path: str) -> str:
    """ New Type function for argparse - gets path, ensures it is absolute path
        and checks if the file exists.
    """

    if not os.path.isabs(path):
        path = os.path.realpath(__file__).split("\\")[0:-1] + [path]
        path = "\\".join(path)
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f'{path}\nGiven file does not exist.')
    if not path.endswith('.csv'):
        raise argparse.ArgumentTypeError(f'{path}\nPlease use .csv file.')
    return path


def sum_to_one(var: str) -> list:
    try:
        var = float(var)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a float value.')
    sum_to_one.s += var
    sum_to_one.c += 1
    if sum_to_one.c == 3 and sum_to_one.s != 1:
        raise argparse.ArgumentTypeError('Coefficients do not sum to 1.')
    return var


sum_to_one.s = 0
sum_to_one.c = 0
# def city_list(path: str):
#     cities = []
#     return cities


class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _format_action_invocation(self, action) -> str:
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        return ', '.join(action.option_strings)


class Parser:
    """ Class that holds parser and performs parsing of command line arguments
        using argparse module.\n

        Attributes:\n
        __p                 -- Holds main argument parser of the program.\n
        sub_p               -- Holds subparsers for each method.\n

        Methods:\n
        __ants_params       -- Creates subparser for ants colony algorithm's
        parameters.\n
        __genetic_params    -- Creates subparser for genetic algorithm's
        parameters.\n
        __sea_params        -- Creates subparser for smallest edge algorithm's
        parameters.
    """

    def __init__(self) -> None:
        """ Initialization of Parser class, creates main parser __p with help
            flag and path positional argumentand calls all the functions to
            create subparsers for each algorithm.
        """

        self.__p = argparse.ArgumentParser(
            prog='TSP solver',
            description='Program for solving of TSP using different methods.',
            formatter_class=lambda prog: CustomHelpFormatter(prog),
            add_help=False
        )
        self.__p.add_argument('-h',
                              '--help',
                              action='help',
                              default=argparse.SUPPRESS,
                              help='Show this help message and exit.'
                              )
        self.__p.add_argument('path',
                              type=is_path,
                              help='Path to database file.'
                              )
        # self.__p.add_argument('-c', '--cities', type=city_list, default='[]',
        #                       help='Specifies list of cities to '
        #                       'use during program runtime.')
        sub_p = self.__p.add_subparsers(title='algorithms',
                                        dest="algorithm",
                                        help='Choice of algorithm.'
                                        )
        # Outside add_subparser to not cause problems with python 3.6.8
        sub_p.required = True
        self.__ants_params(sub_p)
        self.__genetic_params(sub_p)
        self.__sea_params(sub_p)
        self.__pso_params(sub_p)

    def __ants_params(self, sub_p) -> None:
        """ Initialization of subparser for ants algorithm with help flag and
            4 optional parameters that have influence on the algorithm. Takes
            _SubParsersAction as parameter. The parameters are: tours, alpha,
            beta, rho.
        """

        ants = sub_p.add_parser('ants',
                                formatter_class=lambda prog:
                                CustomHelpFormatter(prog),
                                add_help=False
                                )
        ants.add_argument('-h',
                          '--help',
                          action='help',
                          default=argparse.SUPPRESS,
                          help='Show this help message and exit.'
                          )
        ants.add_argument('-t',
                          '--tours',
                          type=positive(int),
                          default=40,
                          help='Number of tours.'
                          )
        ants.add_argument('-a',
                          '--alpha',
                          type=positive(float),
                          default=1,
                          help='Variable controling impact of pheromones '
                          'between nodes.'
                          )
        ants.add_argument('-b',
                          '--beta',
                          type=positive(float),
                          default=2,
                          help='Variable controling impact of distance between'
                          ' nodes.')
        ants.add_argument('-p',
                          '--rho',
                          type=frange(0, 1),
                          default=0.5,
                          help='Variable controling pheromone evaporation.'
                          )

    def __genetic_params(self, sub_p) -> None:
        """ Initialization of subparser for genetic algorithm with help flag
            and 4 optional parameters that have influence on the algorithm.
            Takes _SubParsersAction as parameter. The parameters are: inital
            population size, size of children population expressed as a
            multiplier of initial population, probability of mutation, nuber
            of generations.
        """

        genetic = sub_p.add_parser('genetic',
                                   formatter_class=lambda prog:
                                   CustomHelpFormatter(
                                       prog, max_help_position=30
                                   ),
                                   add_help=False
                                   )
        genetic.add_argument('-h',
                             '--help',
                             action='help',
                             default=argparse.SUPPRESS,
                             help='Show this help message and exit.'
                             )
        genetic.add_argument('-P',
                             '--initial-population',
                             type=int,
                             default=250,
                             help='Size of initial population.'
                             )
        genetic.add_argument('-n',
                             '--children-multiplier',
                             type=float,
                             default=0.8,
                             help='Multiplier for children populaion size '
                             '(P * n)'
                             )
        genetic.add_argument('-p',
                             '--mutation-probability',
                             type=float,
                             default=0.5,
                             help='Mutation rate for offsprings.'
                             )
        genetic.add_argument('-T',
                             '--generation-count',
                             type=int,
                             default=500,
                             help='Number of generations.'
                             )

    def __sea_params(self, sub_p) -> None:
        """ Initialization of subparser for shortest edge algorithm with help
            flag. Takes _SubParsersAction as parameter.
        """

        sea = sub_p.add_parser('sea',
                               formatter_class=lambda prog:
                               CustomHelpFormatter(prog),
                               add_help=False)
        sea.add_argument('-h',
                         '--help',
                         action='help',
                         default=argparse.SUPPRESS,
                         help='Show this help message and exit.'
                         )

    def __pso_params(self, sub_p) -> None:
        """ Initialization of subparser for genetic algorithm with help flag
            and 4 optional parameters that have influence on the algorithm.
            Takes _SubParsersAction as parameter. The parameters are: inital
            population size, size of children population expressed as a
            multiplier of initial population, probability of mutation, nuber
            of generations.
        """

        pso = sub_p.add_parser('pso',
                               formatter_class=lambda prog:
                               CustomHelpFormatter(prog, max_help_position=30),
                               add_help=False
                               )
        pso.add_argument('-h',
                         '--help',
                         action='help',
                         default=argparse.SUPPRESS,
                         help='Show this help message and exit.'
                         )
        pso.add_argument('-c',
                         '--coefficients',
                         metavar='',
                         nargs=3,
                         type=sum_to_one,
                         default=[0.9, 0.05, 0.05],
                         help='Starting strenght of particle\'s own course.'
                         )
        pso.add_argument('-i',
                         '--iterations',
                         metavar='',
                         type=positive(int),
                         default=500,
                         help='Max number of iterations.'
                         )
        pso.add_argument('-n',
                         '--particles-number',
                         metavar='',
                         type=positive(int),
                         default=20,
                         help='Number of simulated particles.'
                         )

    def parse(self) -> argparse.Namespace:
        """ Parses the arguments passed in command line and returns the
            Namespace holding them.
        """

        return self.__p.parse_args()
