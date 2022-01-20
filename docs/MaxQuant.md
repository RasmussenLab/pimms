# MaxQuant

Output tables [overview of MQ](http://www.coxdocs.org/doku.php?id=maxquant:table:directory)

## summary.txt

columns | Description
--- | ---
Raw File | The raw file processed. 
Protease | The protease used to digest the protein sample.
Protease first search | The protease used for the first search.
Use protease first search |  Marked with '+' when a different protease setup was used for the first search.
Fixed modifications | The fixed modification(s) used during the identification of peptides.
Variable modifications | The variable modification(s) used during the identification of peptides.
Variable modifications first search | The variable modification(s) used during the first search.
Use variable modifications firstsearch | Marked with '+' when different variable modifications were used for the first search.
Multiplicity | The number of labels used.
Max. missed cleavages | The maximum allowed number of missed cleavages.
Labels0 | The labels used in the SILAC experiment. Allowed values for X: 0=light; 1=medium; 2=heavy SILAC partner.
LC-MS run type |The type of LC-MS run. Usually it will be 'Standard' which refers to a conventional shotgun proteomics run with datadependent MS/MS. 
Time-dependent recalibration | When marked with ‘+’, time-dependent recalibration was applied to improve the data quality.
MS | The number of MS spectra recorded in this raw file.
MS/MS | The number of tandem MS spectra recorded in this raw file.
MS/MS Submitted | The number of tandem MS spectra submitted for analysis.
MS/MS Submitted (SIL) | The number of tandem MS spectra submitted for analysis, where the precursor ion was detected as part of a SILAC cluster.
MS/MS Submitted (ISO) | The number of tandem MS spectra submitted for analysis, where the precursor ion was detected as an isotopic pattern.
MS/MS Submitted (PEAK) | The number of tandem MS spectra submitted for analysis, where the precursor ion was detected as a single peak.
MS/MS on Polymers | The number of tandem MS spectra, where the precursor ion was a polymer.
MS/MS Identified | The total number of identified tandem MS spectra.
MS/MS Identified (SIL) | The total number of identified tandem MS spectra, where the precursor ion was detected as part of a SILAC cluster.
MS/MS Identified (ISO) | The total number of identified tandem MS spectra, where the precursor ion was detected as an isotopic pattern.
MS/MS Identified (PEAK) | The total number of identified tandem MS spectra, where the precursor ion was detected as a single peak.
MS/MS Identified [%] | The percentage of identified tandem MS spectra.
MS/MS Identified (SIL) [%] | The percentage of identified tandem MS spectra, where the precursor ion was detected as part of a SILAC cluster.
MS/MS Identified (ISO) [%] | The percentage of identified tandem MS spectra, where the precursor ion was detected as an isotopic pattern.
MS/MS Identified (PEAK) [%] | The percentage of identified tandem MS spectra, where the precursor ion was detected as a single peak.
Peptide Sequences Identified | The total number of unique peptide amino acid sequences identified from the recorded tandem mass spectra.
Peaks | The total number of peaks detected in the full scans.
Peaks Sequenced | The total number of peaks sequenced by tandem MS.
Peaks Sequenced [%] | The percentage of peaks sequenced by tandem MS.
Peaks Repeatedly Sequenced | The total number of peaks repeatedly sequenced (i.e. 1 or more times) by tandem MS.
Peaks Repeatedly Sequenced [%] | The percentage of peaks repeatedly sequenced (i.e. 1 or more times) by tandem MS.
Isotope Patterns | The total number of detected isotope patterns.
Isotope Patterns Sequenced          | The total number of isotope patterns sequenced by tandem MS.
Isotope Patterns Sequenced (z>1)    | The total number of isotope patterns sequenced by tandem MS with a charge state of 2 or more.
Isotope Patterns Sequenced [%]      | The percentage of isotope patterns sequenced by tandem MS.
Isotope Patterns Sequenced (z>1) [%] | The percentage of isotope patterns sequenced by tandem MS with a charge state of 2 or more.
Isotope Patterns Repeatedly Sequenced | The total number of isotope patterns repeatedly sequenced (i.e. 1 or more times) by tandem MS.
Isotope Patterns Repeatedly Sequenced [%] | The percentage of isotope patterns repeatedly sequenced (i.e.1 or more times) by tandem MS.
Recalibrated |  When marked with '+', the masses taken from the raw file were recalibrated.
Av. Absolute Mass Deviation  | The average absolute mass deviation found comparing to the identification mass.
Mass Standard Deviation |The standard deviation of the mass deviation found comparing to the identification mass.
