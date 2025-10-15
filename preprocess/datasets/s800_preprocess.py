#!/usr/bin/env python

import os, json
import argparse
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path 

def load_cui_set(path):
        cui_set = set()
        with open(path, 'r') as f:
            for line in f:
                cui, *_ = line.strip().split('||')
                for c in cui.split('|'):
                    c = c.strip()
                    cui_set.add(c)
        return cui_set

def main(args):
    input_dir = args.corpus_dir
    dictionary_dir = args.dictionary_dir
    output_dir = args.output_dir

    # create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(dictionary_dir, "alt_ids2cui.json")) as f:
        alt_ids2cui = json.load(f)

    cui_set = load_cui_set(os.path.join(dictionary_dir, "train_dictionary.txt"))
    print("CUI set size: ", len(cui_set))

    num_docs = 0
    num_queries = 0
    stats = defaultdict(lambda:defaultdict(int))

    print("Processing the corpus:")
    with open(os.path.join(input_dir, "Corpus/S800.tsv")) as file:
        lines = file.readlines()

    data = defaultdict(list)
    for line in lines:
        if line:
            cui, pmid_raw, start, end, mention = line.split("\t")
            pmid = pmid_raw.split(":")[0]
            data[pmid].append({
                "cui":cui, 
                "mention": mention.strip(), 
                "start": int(start), 
                "end": int(end)+1
                })
            
    for idx, pmid in tqdm(enumerate(data), total=len(data)):

        split = "traindev" if idx < 500 else "test" 

        # collect context:
        filepath = os.path.join(input_dir, "Corpus/abstracts", pmid+".txt")
        with open(filepath) as fin:
            context = fin.read()

        queries = []
        # collect queries
        for entry in data[pmid]:
            cui = entry["cui"]
            mention = entry["mention"]
            start, end = entry["start"], entry["end"]

            if any(map(lambda x: x in cui, ["|", "+", ",", ";"])):
                stats[split]["composite"] +=1
                continue

            # replace alternative cuis 
            cui = alt_ids2cui.get(cui, cui) 

            if mention != context[start:end]:
                print(f"Different mention in pmid {pmid}:\n", mention, context[start:end])
                continue

            if cui in cui_set:
                clean_mention = mention.strip().replace("\n", " ")
                query = f"{pmid}||{start}|{end}||species||{clean_mention}||{cui}"
                queries.append(query)
                stats[split]["n_mentions"] +=1
                
            else:
                stats[split]["oov"] += 1   
        
        concept = "\n".join(queries) + "\n"

        # Writing files
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        output_context_file = os.path.join(output_dir, split, "{}.txt".format(pmid))
        output_concept_file = os.path.join(output_dir, split, "{}.concept".format(pmid))

        with open(output_context_file, 'w', encoding="utf-8") as f:
            f.write(context)
        with open(output_concept_file, 'w', encoding="utf-8") as f:
            f.write(concept)
        
        num_docs +=1
        stats[split]["n_docs"] +=1
        
    for spl in stats:
        composite, oov = stats[spl]["composite"], stats[spl]["oov"]
        num_queries = stats[spl]["n_mentions"]
        total_mentions = num_queries + composite + oov
        
        with open(Path(output_dir) / f"statistics_{spl}.json", "w") as file:

            json.dump({
            "docs":stats[spl]["n_docs"],
            "filtered mentions": f"{num_queries} ({num_queries/total_mentions * 100:.2f})",
            "composite": f"{composite} ({composite/total_mentions * 100:.2f})",
            "oov": f"{oov} ({oov/total_mentions * 100:.2f})",
            "total mentions": total_mentions
        }, file, indent=4)
    
    print("{} {} {}".format(output_dir, num_docs, num_queries))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, required=True,
                    help='path to the folder with corpus file')
    parser.add_argument('--dictionary_dir', type=str, required=True,
                    help='path to the folder with train_dictionary.txt file')
    parser.add_argument('--output_dir', type=str, required=True,
                    help='path of output directionary')
    args = parser.parse_args()
    
    main(args)
