from parser_class import Parser
from console_app import Console_App


if __name__ == '__main__':
    # setup parser
    parser = Parser()
    args = parser.parse()
    # TODO add logger

    # run app
    app = Console_App(args)
    app.run()
