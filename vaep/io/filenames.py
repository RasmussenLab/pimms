import re
import logging
import functools

logger = logging.getLogger(__name__)


def read_number_from_str(fname: str, regex: str = 'M[0-9]*', strip: int = 1) -> int:
    M = re.search(regex, fname).group()
    logger.info(f"Found: {M}")
    M = int(M[strip:])
    return M


read_M_features = functools.partial(read_number_from_str, regex='M[0-9]*', strip=1)
read_N_samples = functools.partial(read_number_from_str, regex='N[0-9]*', strip=1)
