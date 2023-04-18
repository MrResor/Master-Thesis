from __init__ import logging, sqlite3


def db_handler(func):
    def db_decorator(self, path) -> None:
        try:
            func(self, path)
        except sqlite3.OperationalError as err:
            logging.info(str(err).capitalize() + '.',
                         extra={'runtime': 0})
            logging.info('Quitting with error code 1.',
                         extra={'runtime': 0})
            quit(1)

    return db_decorator
