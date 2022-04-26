from pathlib import Path
from datetime import datetime
import logging
import sys

LOG_FOLDER = Path('logs')
LOG_FOLDER.mkdir(exist_ok=True)


def setup_nb_logger(level: int = logging.INFO,
                    format_str: str = f'%(name)s - %(levelname)-8s %(message)s') -> None:
    logging.basicConfig(level=level, format=format_str)
    logger = logging.getLogger()
    logger.setLevel(level) # in case logger existed already before calling basicConfig
    c_format = logging.Formatter(format_str)
    if logger.handlers:
        handler = logger.handlers[0]
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(c_format)

def setup_logger_w_file(logger, level=logging.INFO, fname_base=None):
    """Setup logging in project. Takes a logger an creates

    Parameters
    ----------
    logger : logging.Logger
        logger instance to configre
    level : int, optional
        logging level, by default logging.INFO
    fname_base : str, optional
        filename for logging, by default None

    Returns
    -------
    logging.Logger
        Configured logger instance for logging

    Examples
    --------
    >>> import logging
    >>> logger = logging.getLogger('vaep')
    >>> _ = setup_logger_w_file(logger) # no logging to file
    >>> logger.handlers = [] # reset logger
    >>> _ = setup_logger_w_file() # 

    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = []  # remove any handler in case you reexecute the cell

    c_format = logging.Formatter(f'%(name)s - %(levelname)-8s %(message)s')

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(level)
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    if fname_base:
        date_log_file = "{:%y%m%d_%H%M}".format(datetime.now())
        f_handler = logging.FileHandler(
            LOG_FOLDER / f"{fname_base}_{date_log_file}.txt")
        f_handler.setLevel(level)
        f_handler.setFormatter(c_format)
        logger.addHandler(f_handler)

    return logger


setup_logger = setup_logger_w_file
