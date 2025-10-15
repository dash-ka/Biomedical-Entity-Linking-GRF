# Biomedical-Entity-Linking-GRF

## Installation

Run the following commands:

1. Download datasets and kbs from

2. Unzip the resources in 'submission' folder.
   
3. Place the preprocessing scripts in 'submission' folder.
   
4. Preprocess the dictionaries first

```bash
cd preprocess/kb
python --dictionary_path ../../kb/CTD_diseases.tsv.gz --output_dir ../../kbs/ctd-disease
python --dictionary_path ../../kb/CTD_chemicals.tsv.gz --output_dir ../../kbs/ctd-chemical
python --dictionary_path ../../kb/nlm_gene_subset.txt --output_dir ../../kbs/ncbi-gene/nlm
python --dictionary_path ../../kb/ncbi_gene_subset.txt --output_dir ../../kbs/ncbi-gene/gnormplus
python --dictionary_path ../../kb/ncbitaxon.obo --output_dir ../../kbs/ncbi-taxonomy
```

5. Preprocess datasets

6. Get unseen queries.json file for GRF generation.

```bash
cd ../..
python get_unseen_queries.py --input_dir ../submission/datasets/ncbi-disease/processed_data/
python get_unseen_queries.py --input_dir ../submission/datasets/bc5cdr-disease/processed_data/
python get_unseen_queries.py --input_dir ../submission/datasets/bc5cdr-chemical/processed_data/
python get_unseen_queries.py --input_dir ../submission/datasets/nlm-chemical/processed_data/
python get_unseen_queries.py --input_dir ../submission/datasets/ncbi-gene/processed_data/
python get_unseen_queries.py --input_dir ../submission/datasets/nlm-gene/processed_data/
python get_unseen_queries.py --input_dir ../submission/datasets/s800/processed_data/
python get_unseen_queries.py --input_dir ../submission/datasets/ncbi-taxon/processed_data/
```
