from vaep.fasta import get_n_miscleaved
from vaep.fasta import find_rxk_peptides


def test_get_n_miscleaved_miss_1():
    pep_seqs = ['MSSHEGGK', 'K', 'K', 'ALK', 'QPK', 'K', 'QAK', 'EMDEEEK', 'AFK',
                'QK', 'QK', 'EEQK', 'K', 'LEVLK', 'AK', 'VVGK', 'GPLATGGIK', 'K', 'SGK', 'K']

    true_result = set(['MSSHEGGKK', 'KK', 'KALK', 'ALKQPK', 'QPKK', 'KQAK', 'QAKEMDEEEK', 'EMDEEEKAFK', 'AFKQK',
                       'QKQK', 'QKEEQK', 'EEQKK', 'KLEVLK', 'LEVLKAK', 'AKVVGK', 'VVGKGPLATGGIK', 'GPLATGGIKK', 'KSGK', 'SGKK'])

    result = get_n_miscleaved(pep_sequences=pep_seqs, num_missed=1)

    assert len(true_result.difference(result)) == 0, print(
        true_result.difference(result))


def test_find_rxk_peptides():

    pep_seqs = ['MSSHEGGK', 'K', 'K', 'ALK', 'QPK', 'K', 'QAK', 'EMDEEEK', 'AFK',
                'QK', 'QK', 'EEQK', 'K', 'LEVLK', 'AK', 'VVGK', 'GPLATGGIK', 'K', 'SGK', 'K']

    true_rdx_peps = ['KKALK', 'KALKQPK', 'KQAKEMDEEEK', 'QKQKEEQK',
                     'QKEEQKK', 'KLEVLKAK', 'AKVVGKGPLATGGIK', 'KSGKK']

    assert true_rdx_peps == find_rxk_peptides(
        pep_seqs), 'Build sequence: {}'.format(repr(find_rxk_peptides(pep_seqs)))
