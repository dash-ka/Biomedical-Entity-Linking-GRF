# Biomedical-Entity-Linking-GRF

## Installation

Run the following commands:

1. Preprocess the dictionaries first

```bash
cd preprocess/kb
python --dictionary_path ../../kbs/CTD_diseases.tsv.gz --output_dir ../../kbs/ctd-disease
python --dictionary_path ../../kbs/CTD_chemicals.tsv.gz --output_dir ../../kbs/ctd-chemical
python --dictionary_path ../../kbs/nlm_gene_subset.txt --output_dir ../../kbs/ncbi-gene/nlm
python --dictionary_path ../../kbs/ncbi_gene_subset.txt --output_dir ../../kbs/ncbi-gene/gnormplus
python --dictionary_path ../../kbs/ncbitaxon.obo --output_dir ../../kbs/ncbi-taxonomy
```

2. Preprocess datasets

3. Get unseen queries.json file for GRF generation.

