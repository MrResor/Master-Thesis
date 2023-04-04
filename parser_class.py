import argparse


def frange(min, max):
    """ Return function handle of an argument type function for 
        ArgumentParser checking a float range: mini <= arg <= maxi
        min - minimum acceptable argument
        max - maximum acceptable argument
    """

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < min or f > max:
            raise argparse.ArgumentTypeError(
                "must be in range [" + str(min) + " .. " + str(max)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker


class Parser:
    def __init__(self) -> None:
        self.__p = argparse.ArgumentParser(
            prog='TSP solver',
            description='Program for solving of TSP using different mehtods',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        self.__p.add_argument('-c', '--console',
                              action='store_true',
                              default=False,
                              help='flag for running program in console mode')
        sub_p = self.__p.add_subparsers(title='algorithms',
                                        dest="algorithm",
                                        help='Choice of algorithm.')
        self.__ants_params(sub_p)
        self.__genetic_params(sub_p)

    def __ants_params(self, sub_p) -> None:
        ants = sub_p.add_parser(
            'ants', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        ants.add_argument('-t', '--tours', type=int, default=40, metavar='',
                          help='Number of tours.')
        ants.add_argument('-a', '--alpha', type=float, default=1, metavar='',
                          help='Variable controling impact of pheromones '
                          'between nodes.')
        ants.add_argument('-b', '--beta', type=float, default=2, metavar='',
                          help='Variable controling impact of distance between'
                          ' nodes.')
        ants.add_argument('-p', '--rho', type=frange(0, 1), default=0.5,
                          metavar='', help='Variable controling pheromone '
                          'evaporation.')

    def __genetic_params(self, sub_p) -> None:
        genetic = sub_p.add_parser(
            'genetic', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        genetic.add_argument('-P', type=int,
                             default=250, help='Number of tours, '
                             'default is 40')
        genetic.add_argument('-n', '--children_multiplier', type=float, default=2,
                             help='Variable controling ..., default is ...')
        genetic.add_argument('-b', '--beta', type=float,
                             help='Variable controling ..., default is ...')
        genetic.add_argument('-p', '--rho', type=frange(0, 1))

    def parse(self) -> argparse.Namespace:
        return self.__p.parse_args()
