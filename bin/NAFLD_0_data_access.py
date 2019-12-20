"""
Accessing PRIDE Archive using bioservices client

- bioservices: [Quick start documentation](https://bioservices.readthedocs.io/en/master/quickstart.html)
- [Client-API for PRIDE](https://bioservices.readthedocs.io/en/master/references.html#module-bioservices.pride)


# PRIDE Archives from Lili's paper:
- [PXD011839](https://www.ebi.ac.uk/pride/archive/projects/PXD011839)
- [PXD012056](https://www.ebi.ac.uk/pride/archive/projects/PXD012056)

"""
import os

import bioservices 
from bioservices.pride import PRIDE

client = PRIDE(verbose=True, cache=False)
# 

archives = [
    "PXD011839",
    "PXD012056"
]


identifier = archives[0]

#
client.get_file_count(identifier)
files = client.get_file_list(identifier)
infos = client.get_project(identifier)

tissue = infos['tisues'][0]
species = infos['species'][0]

ftp_link = files[0]['downloadLink']

import ftplib
ftp = ftplib.FTP()

fname = os.path.join('data', species, tissue)
with open(fname, 'wb') as f:
    ftp.retrbinary()
files

## General functions
_query = 'Hela'
n_projects = client.get_project_count(query=_query)
results = client.get_project_list(query=_query, show=10, page=0)

