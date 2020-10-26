from vaep.fasta import get_n_miscleaved
from vaep.fasta import find_rxk_peptides
from vaep.fasta import count_peptide_matches


def test_get_n_miscleaved_miss_1():
    pep_seqs = ['MSSHEGGK', 'K', 'K', 'ALK', 'QPK', 'K', 'QAK',
                'EMDEEEK', 'AFK',
                'QK', 'QK', 'EEQK', 'K', 'LEVLK', 'AK',
                'VVGK', 'GPLATGGIK', 'K', 'SGK', 'K']

    true_result = set(['MSSHEGGKK', 'KK', 'KALK', 'ALKQPK', 'QPKK', 'KQAK', 'QAKEMDEEEK',
                       'EMDEEEKAFK', 'AFKQK',
                       'QKQK', 'QKEEQK', 'EEQKK', 'KLEVLK', 'LEVLKAK', 'AKVVGK',
                       'VVGKGPLATGGIK', 'GPLATGGIKK', 'KSGK', 'SGKK'])

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


def test_count_pep_mapped_to_gene():
    test_peptide_to_proteinID = {'LMHIQPPK': [
        'A0JLT2', 'A0A2R8YDL4', 'A0A494C0G4', 'A0A494C0Y4', 'A0JLT2-2', 'J3KR33']}
    test_protein_to_gene = {'A0JLT2': 'MED19', 'J3KR33': 'MED19'}

    assert count_peptide_matches(
        test_peptide_to_proteinID, test_protein_to_gene) == {'LMHIQPPK': 6}

    assert count_peptide_matches(test_peptide_to_proteinID, test_protein_to_gene, level='protein') == {
        'LMHIQPPK': 5}, 'Error for canonical protein level (combination of all isotopes in UniProt)'

    assert count_peptide_matches(
        test_peptide_to_proteinID, test_protein_to_gene, level='gene') == {'LMHIQPPK': 1}
