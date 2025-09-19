# Biomedical-Entity-Linking-GRF

## Installation

Run the following commands:

1. Preprocess the dictionaries first

```bash
cd preprocess/kb
python --dictionary_path ../../kbs/CTD_diseases.tsv.gz --output_dir ../../datasets/ncbi-disease
python --dictionary_path ../../kbs/CTD_diseases.tsv.gz --output_dir ../../datasets/bc5cdr-disease
python --dictionary_path ../../kbs/CTD_chemicals.tsv.gz --output_dir ../../datasets/bc5cdr-chemical
python --dictionary_path ../../kbs/CTD_chemicals.tsv.gz --output_dir ../../datasets/nlm-chemical
python --dictionary_path ../../kbs/nlm_gene_subset.txt --output_dir ../../datasets/nlm-gene
python --dictionary_path ../../kbs/ncbi_gene_subset.txt --output_dir ../../datasets/ncbi-gene
python --dictionary_path ../../kbs/ncbitaxon.obo --output_dir ../../datasets/ncbi-taxon
python --dictionary_path ../../kbs/ncbitaxon.obo --output_dir ../../datasets/s800
```

2. Preprocess datasets

3. Get unseen queries.json file for GRF generation.

