"""
Based on notebook received by [Annelaura Bach](https://www.cpr.ku.dk/staff/mann-group/?pure=en/persons/443836)
and created by Johannes B. Müller
[scholar](https://scholar.google.com/citations?user=Rn1OS8oAAAAJ&hl=de),
[MPI Biochemistry](https://www.biochem.mpg.de/person/93696/2253)
"""
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_n_miscleaved(pep_sequences: list, num_missed: int):
    """Build miscleaved peptide sequences for a number of missed cleaving sides.
    Call function recusively if you want not only a specific number of miscleavages."""
    _miscleaved = []
    for i in range(len(pep_sequences)):
        if i >= num_missed:
            _miscleaved.append(''.join(pep_sequences[i - num_missed:i + 1]))
    return _miscleaved


def cleave_to_tryptic(seq, num_missed_cleavages=1, reversed=False, add_rxk=False):
    """Takes a sequence and returns an array of peptides cleaved C-term to R  and K.

    Parameters
    ----------
    seq : str, optional
        sequence of amino acids as string
    num_missed_cleavages : int, optional
        number of missed cleavages to consider, by default 1
    reversed : bool, optional
        if reversed decoy peptide sequences should be added, by default False
    """

    seq.replace(' ', '')  # remove white spaces
    seq = seq.upper()
    # add whitespace behind K and R for splitting
    seq = seq.replace('K', 'K ').replace('R', 'R ').split()

    peps_seq = [seq, ]

    for i in range(1, num_missed_cleavages + 1):
        _seq = get_n_miscleaved(seq, num_missed=i)
        peps_seq.append(_seq)

    if add_rxk and num_missed_cleavages < 2:
        _seq = find_rxk_peptides(seq)
        peps_seq.append(_seq)

    return peps_seq


def find_rxk_peptides(l_peptides):
    """Combine 3 peptides to one, if the first is an
    'RxK'-peptide: RX, XR, KX, XK - where the X can
    be any other amino-acid.

    Returns
    -------
    list
        list of miscleaved peptides. Can be empty.
    """
    if len(l_peptides) >= 3:
        rdx_peptides = []
        for i in range(len(l_peptides) - 2):
            if len(l_peptides[i]) <= 2:
                rdx_peptides.append(
                    ''.join(l_peptides[i:i + 3])
                )
        return rdx_peptides
    else:
        return []


def read_fasta(fp):
    """Read a fasta file and yield continously header and sequences."""
    header, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if header:
                yield (header, ''.join(seq))
            header, seq = line, []
        else:
            seq.append(line)
    if header:
        yield (header, ''.join(seq))


def iterFlatten(root):
    """Flatten a nested structure."""
    if isinstance(root, (list, tuple)):
        for element in root:
            for e in iterFlatten(element):
                yield e
    else:
        yield root


def count_peptide_matches(peptide_to_proteinID: dict,
                          protein_to_gene: dict = None,
                          level: str = 'protein_id') -> dict:
    """Count the number of matches of a peptide to the specified level.
    Provides the basis for summary statistics (a counter of matches).

    Possibly to be extended in a class handling the matches of proteinIDs to
    peptide sequences.

    Parameters
    ----------
    peptide_to_proteinID : dict
        mapping of peptides to a list of proteinIDs.
    protein_to_gene : dict, optional
        Uniport mapping of protein IDs (no isotopes ending "-2", "-3", etc)
        in case of level='gene', by default None
    level : str, optional
        to which level the peptides should be matched.
        'protein_id', 'protein' or 'gene', by default 'protein_id'

    Returns
    -------
    dict
        Counter of number of matches of peptides to the specified level items.
        {1 : 3, 3: 5} means that 3 peptides have been match uniquly to one item (e.g. gene)
        and 5 peptides have been matched to 3 items (e.g. genes).

    Raises
    ------
    KeyError
        [description]
    """
    assert level in ['protein_id', 'protein', 'gene'], ValueError(
        'Specify one of the three (in order of aggregation level): {}'.format(
            ', '.join(['prot_id', 'prot', 'gene'])))
    if level == 'gene':
        assert protein_to_gene is not None, "Please provide protein to gene level name"
        _set_missing_entires = set()
    n_peptides_mapped_to_level = {}

    for _pep, _protein_ids in tqdm(peptide_to_proteinID.items()):

        if level == 'protein_id':
            n_peptides_mapped_to_level[_pep] = len(_protein_ids)
        elif level == 'protein':
            n_peptides_mapped_to_level[_pep] = len(
                {x.split('-')[0] for x in peptide_to_proteinID[_pep]})
        elif level == 'gene':
            _proteins = {x.split('-')[0] for x in peptide_to_proteinID[_pep]}
            _set_genes = set()
            for _prot in _proteins:
                try:
                    _set_genes.add(protein_to_gene[_prot])
                except KeyError:
                    _set_missing_entires.add(_prot)
            n_peptides_mapped_to_level[_pep] = len(_set_genes)
        else:
            raise KeyError(f'unknown level: {level}')
    if level == 'gene':
        logger.warning(
            f'Missing protein to gene mapppings: {len(set(_set_missing_entires))}')
    return n_peptides_mapped_to_level
