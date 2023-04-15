from parser_class import Parser
from console_app import Console_App
import sys


if __name__ == '__main__':
    # setup parser
    parser = Parser()
    args = parser.parse()
    # TODO add logger

    # run app
    run = Console_App(args)
