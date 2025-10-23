# Biomedical-Entity-Linking-GRF

## Installation

Run the following commands:

1. Download datasets and kbs from https://zenodo.org/records/17359566 

3. Unzip the resources in `submission` folder.
   
4. Place the preprocessing scripts in `submission` folder.
   
5. Preprocess the dictionaries first

```bash
cd preprocess/kb
python process_kb_ctd.py --dictionary_path ../../kb/CTD_diseases.tsv.gz --output_dir ../../kbs/ctd-disease
python process_kb_ctd.py --dictionary_path ../../kb/CTD_chemicals.tsv.gz --output_dir ../../kbs/ctd-chemical
python process_kb_ncbi_gene.py --dictionary_path ../../kb/nlm_gene_subset.txt --output_dir ../../kbs/ncbi-gene/nlm
python process_kb_ncbi_gene.py --dictionary_path ../../kb/ncbi_gene_subset.txt --output_dir ../../kbs/ncbi-gene/gnormplus
python process_kb_ncbi_taxonomy.py --dictionary_path ../../kb/ncbitaxon.obo --output_dir ../../kbs/ncbi-taxonomy
```

5. Preprocess datasets

```bash
cd preprocess/datasets
python ncbi_disease_preprocess.py --corpus_dir ../../datasets/ncbi-disease/ --dictionary_dir ../../kbs/ctd-disease --output_dir ../../datasets/ncbi-disease/processed_data/
python bc5cdr_preprocess.py --corpus_dir ../../datasets/bc5cdr-disease/ --dictionary_dir ../../kbs/ctd-disease --output_dir ../../datasets/bc5cdr-disease/processed_data/ --type disease
python bc5cdr_preprocess.py --corpus_dir ../../datasets/bc5cdr-chemical/ --dictionary_dir ../../kbs/ctd-chemical --output_dir ../../datasets/bc5cdr-chemical/processed_data/ --type chemical
python nlm_chemical_preprocess.py --corpus_dir ../../datasets/nlm-chemical/ --dictionary_dir ../../kbs/ctd-chemical --output_dir ../../datasets/nlm-chemical/processed_data/
python gnormplus_gene_preprocess.py --corpus_dir ../../datasets/gnormplus/ --dictionary_dir ../../kbs/ncbi-gene/gnormplus/ --output_dir ../../datasets/gnormplus/processed_data/
python nlm_gene_preprocess.py --corpus_dir ../../datasets/nlm-gene/ --dictionary_dir ../../kbs/ncbi-gene/nlm/ --output_dir ../../datasets/nlm-gene/processed_data/
python s800_preprocess.py --corpus_dir ../../datasets/s800/ --dictionary_dir ../../kbs/ncbi-taxonomy/ --output_dir ../../datasets/s800/processed_data/
python linnaeus_preprocess.py  --dictionary_dir ../../kbs/ncbi-taxonomy/ --output_dir ../../datasets/linnaeus/processed_data/
```

6. Build unseen test queries `queries.json` file for GRF generation.

```bash
cd preprocess
python get_unseen_queries.py --input_dir ../datasets/ncbi-disease/processed_data/
python get_unseen_queries.py --input_dir ../datasets/bc5cdr-disease/processed_data/
python get_unseen_queries.py --input_dir ../datasets/bc5cdr-chemical/processed_data/
python get_unseen_queries.py --input_dir ../datasets/nlm-chemical/processed_data/
python get_unseen_queries.py --input_dir ../datasets/gnormplus/processed_data/
python get_unseen_queries.py --input_dir ../datasets/nlm-gene/processed_data/
python get_unseen_queries.py --input_dir ../datasets/s800/processed_data/
python get_unseen_queries.py --input_dir ../datasets/linnaeus/processed_data/
```
7.1 Generate Feedback with an LLM using OpenAI.
```bash
python generate_feedback_openai.py --OPENAI_API_KEY "YOUR_OPENAI_KEY" --model_name "gpt-4o-2024-08-06" --queries_path ../datasets/ncbi-disease/raw_queries.json --dataset_name ncbi-disease
```

7.2 Generate Feedback with an open-source LLM using vLLM.
```bash
python generate_feedback_vllm.py --HF_TOKEN "YOUR_HF_TOKEN" --model_name "Qwen/Qwen3-14B" --queries_path ../datasets/ncbi-disease/raw_queries.json --dataset_name ncbi-disease
```
   
8. To reproduce the GRF results run the `run_grf_experiment.py` script for each dataset. 
