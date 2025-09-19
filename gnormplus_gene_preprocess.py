#!/usr/bin/env python
"""
Input: NCBI(train/development/test)set_corpus.txt
https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/
Output: (train/dev/test)/*.concept, *.txt
"""

import os, json, re
import argparse
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

def main(args):
    input_dir = args.input_dir
    main_dir = args.output_dir    
    
    if args.human_only:
        alt_ids_file_name = "alt_ids2cui_human.json" 
    elif args.qualify:
        alt_ids_file_name = "alt_ids2cui.json" 
    else:
        alt_ids_file_name = "alt_ids2cui.json"

    with open(os.path.join(input_dir, alt_ids_file_name)) as f:
        alt_ids2cui = json.load(f)

    def load_cui_set(path):
        cui_set = set()
        with open(path, 'r') as f:
            for line in f:
                cui, *_ = line.strip().split('||')
                for c in cui.split('|'):
                    c = c.strip()
                    cui_set.add(c)
        return cui_set

    if args.human_only:
        dictionary_file_name = "train_dictionary_human.txt" 
    elif args.qualify:
        dictionary_file_name = "train_dictionary.txt"
    else:
        dictionary_file_name = "train_dictionary.txt"

    cui_set = load_cui_set(os.path.join(input_dir, dictionary_file_name))
    print("CUI set size: ", len(cui_set))

    # loading knowledge base
    with open(os.path.join(input_dir, "kb.json")) as file:
        kb = json.load(file)

    # read lines from raw file
    split_files = {
        "traindev":"BC2GNtrain.PubTator.txt",
        "test": "BC2GNtest.PubTator.txt"
    }

    for split in split_files:

        with open(os.path.join(input_dir, "Corpus", split_files[split]), 'r') as f:
            lines = f.readlines()

        # create directory if it doesn't exist
        output_dir = Path(main_dir) / split
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def extract_tax_id(text):
            match = re.search(r'\(Tax:(\d+)\)', text)
            return match.group(1) if match else "TaxID not specified"

        human_only = args.human_only
        queries = []
        pmids = []
        lines = lines + ['\n']
        num_docs = 0
        num_queries = 0
        tax2gene = defaultdict(int)
        composite, oov = 0, 0

        for line in tqdm(lines):
            line = line.strip()
            if '|t|' in line:
                title = line.split("|", maxsplit=2)[2]
            elif '|a|' in line:
                abstract = line.split("|", maxsplit=2)[2]
            elif '\t' in line:
                line = line.split("\t")

                if len(line) == 6:
                    pmid, start, end, mention, _class, cui = line

                    cui = cui.strip("-")
                    for sep in [";", "-"]:
                        cui = re.sub(sep, ",", cui)
                
                    if human_only:
                        clean_cuis = [c.split("(Tax:")[0] for c in cui.split(",") if extract_tax_id(c) in ["TaxID not specified", "9606"]]
                        if not clean_cuis:
                            continue
                    else:
                        clean_cuis = [c.split("(Tax:")[0] for c in cui.split(",")]

                    clean_cuis = [alt_ids2cui.get(c, c) for c in clean_cuis]
                
                else: # not a gene but "FamilyMotif", "DomainMotif" etc...
                    continue

                if len(clean_cuis) > 1:
                    composite += 1
                    continue

                cui = clean_cuis[0]
                if cui in cui_set:
                    clean_mention = mention.strip().replace("\n", " ")

                    taxon = kb[cui]["organism"]
                    if args.qualify:
                        clean_mention = f"{clean_mention} ({taxon})"

                    query = pmid + "||" + start +"|" + end + "||" + taxon + "||" + clean_mention + "||" + cui
                    queries.append(query)
                    tax2gene[taxon] +=1
                else:
                    oov +=1

            elif len(queries): 
                if pmid in pmids:
                    print(pmid)
                    queries = []
                    title = ""
                    abstract = ""
                    continue

                context = title + "\n" + abstract + "\n"
                concept = "\n".join(queries) + "\n"
                output_context_file = output_dir / "{}.txt".format(pmid)
                output_concept_file = output_dir / "{}.concept".format(pmid)
                
                with open(output_context_file, 'w', encoding="utf-8") as f:
                    f.write(context)
                with open(output_concept_file, 'w', encoding="utf-8") as f:
                    f.write(concept)

                num_docs +=1
                num_queries += len(queries)
                pmids.append(pmid)
                queries = []
                title = ""
                abstract = ""

        total_mentions = num_queries + composite + oov
        
        parent_dir = output_dir.parent
        with open(parent_dir / f"statistics_{split}.json", "w") as file:

            json.dump({
                "docs":num_docs,
                "filtered mentions": f"{num_queries} ({num_queries/total_mentions * 100:.2f})",
                "composite": f"{composite} ({composite/total_mentions * 100:.2f})",
                "oov": f"{oov} ({oov/total_mentions * 100:.2f})",
                "total mentions": total_mentions,
                "taxa": tax2gene
            }, file, indent=4)
        
        print("{} {} {}".format(output_dir, num_docs, num_queries))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                    help='path of input folder')
    parser.add_argument('--output_dir', type=str,
                    help='path of output directionary')
    parser.add_argument('--human_only', action="store_true",
                        help="If specified, only include human genes")
    parser.add_argument('--qualify', action="store_true",
                        help="If specified, qualify mentions with organism name")
    args = parser.parse_args()
    
    main(args)
