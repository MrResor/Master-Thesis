from parser_class import Parser
from console_app import Console_App
from PyQt5.QtWidgets import QMessageBox, QApplication
import sys


if __name__ == '__main__':
    # setup parser
    # args = parse()
    parser = Parser()
    args = parser.parse()
    # TODO add logger

    if args.console:
        # run in console
        run = Console_App(args)
    else:
        app = QApplication(sys.argv)
        msg = QMessageBox()
        msg.setWindowTitle('TSP Solver')
        msg.setIcon(QMessageBox.Information)
        msg.setText('The UI version of this program is not yet implemented.\n'
                    'Please run the program with -h flag to know more.')
        msg.setStandardButtons(QMessageBox.Ok)
        sys.exit(msg.exec_())
