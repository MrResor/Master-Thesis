from __init__ import logging, sqlite3
from typing import Callable


def db_handler(func) -> Callable[..., None]:
    """ Decorator returning function for handling sqlite3 errors.
    """

    def db_decorator(self, path) -> None:
        """ Inner functions, catches the error and quits the program.
        """

        try:
            func(self, path)
        except sqlite3.OperationalError as err:
            logging.info(str(err).capitalize() + '.',
                         extra={'runtime': 0})
            logging.info('Quitting with error code 1.',
                         extra={'runtime': 0})
            quit(1)

    return db_decorator
