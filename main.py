from console_app import Console_App
from parser_class import Parser


if __name__ == '__main__':
    # setup parser and parse the parameters
    parser = Parser()
    args = parser.parse()

    # run app
    app = Console_App(args)
    app.run()
