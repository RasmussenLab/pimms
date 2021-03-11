from pathlib import Path
from datetime import datetime
import logging

LOG_FOLDER = Path('logs')
LOG_FOLDER.mkdir(exist_ok=True)

def setup_logger_w_file(logger, level=logging.INFO, fname_base='log'):
    """Setup logging in project"""
    logger.setLevel(logging.INFO)
    logger.handlers = []  #remove any handler in case you reexecute the cell

    c_format = logging.Formatter(f'%(name)s - %(levelname)-8s %(message)s')
    
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    if fname_base:
        date_log_file = "{:%y%m%d_%H%M}".format(datetime.now())
        f_handler = logging.FileHandler(LOG_FOLDER / f"{fname_base}_{date_log_file}.txt")
        f_handler.setLevel(logging.INFO) 
        f_handler.setFormatter(c_format)
        logger.addHandler(f_handler)
    
    return logger
