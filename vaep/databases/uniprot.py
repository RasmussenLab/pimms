import urllib.parse
import urllib.request

__all__ = ['query_uniprot_id_mapping']

def query_uniprot_id_mapping(query_list: list, FROM='ACC+ID', TO='GENENAME', FORMAT='tab'):
    """Query Uniprot ID mappings programatically.

    See availabe mappings: https://www.uniprot.org/help/api_idmapping
    
    Parameters
    ----------
    query_list : list
        list of strings containing queries in format specified 
        in FROM parameter.
    FROM : str, optional
        Format of string-ids in query_list, by default 'ACC+ID'
    TO : str, optional
        Format to which strings-ids should be matched with, by default 'GENENAME'
    FORMAT : str, optional
        Separator for Uniprot-ID, by default 'tab'
        
    Returns
    -------
    list:
        List of tuples of type (FROM, TO)
    """

    url = 'https://www.uniprot.org/uploadlists/'

    params = {
        'from': FROM,
        'to': TO,
        'format': FORMAT,
        'query': ' '.join(query_list)[:100000].strip()
    }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()
    _l = [line.split('\t')
          for line in response.decode('utf-8').split('\n')[1:-1]]
    return {_to: _from for _to, _from in _l}
