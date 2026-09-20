# FragPipe reference-search configuration

These files preserve the reference-search settings used for the ultradeep HeLa phosphoproteome comparison described in the OmniNovo manuscript:

- `fragpipe.workflow` is an exported FragPipe 22.0 workflow.
- `fragger.params` is the corresponding MSFragger parameter file.

The files contain portable placeholders in place of the original machine-specific paths. Before running FragPipe, replace the database, FragPipe tools, Python interpreter and work-directory paths in `fragpipe.workflow`, and replace `database_name` in `fragger.params`. The required raw spectra, FASTA database, FragPipe, MSFragger, Philosopher, Percolator and IonQuant installations are not included in this repository.
