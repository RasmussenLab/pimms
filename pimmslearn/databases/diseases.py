import logging
import requests

logger = logging.getLogger(__name__)


def get_disease_association(doid: int, limit: int = 1000):
    params = {'type1': -26,
              'id1': f'DOID:{doid}',
              'type2': 9606,
              'limit': limit,
              'format': 'json'}
    diseases_url_all = 'https://api.jensenlab.org/Integration'

    r = requests.get(diseases_url_all, params=params)
    if r.status_code == 200:
        data, is_there_more = r.json()
    else:
        raise ValueError(f"Could not get valid data back, response code: {r.status_code}")
    if is_there_more:
        logger.warning("There are more associations available")
    return data
