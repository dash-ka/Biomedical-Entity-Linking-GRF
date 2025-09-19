#!/usr/bin/env python
"""
Input: CDR_(Training/Development/Test)Set.PubTator.txt
https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/
Output: (train/dev/test)/*.concept, *.txt
"""

import argparse, json, os, shutil
from tqdm import tqdm
from pathlib import Path

def main(args):
    input_dir = args.input_dir
    main_dir = args.output_dir
    entity_type = args.type

    def load_cui_set(path):
        cui_set = set()
        with open(path, 'r') as f:
            for line in f:
                cui, *_ = line.strip().split('||')
                for c in cui.split('|'):
                    c = c.strip()
                    cui_set.add(c)
        return cui_set

    # load cui_set
    cui_set = load_cui_set(os.path.join(input_dir, "train_dictionary.txt"))
    print("CUI set size: ", len(cui_set))

    # load alt_ids2cui
    with open(os.path.join(input_dir, "alt_ids2cui.json")) as f:
        alt_ids2cui = json.load(f)

    # read lines from raw file
    split_files = {
        "train": "CDR_TrainingSet.PubTator.txt",
        "dev": "CDR_DevelopmentSet.PubTator.txt",
        "test":"CDR_TestSet.PubTator.txt",
    }

    for split in split_files:

        output_dir = Path(main_dir) / split
        # create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(input_dir, "Corpus", split_files[split]), 'r') as f:
            lines = f.readlines()
            print(len(lines))

        queries = []
        pmids = []
        lines = lines + ['\n']
        num_docs = 0
        num_queries = 0
        composite = 0
        oov = 0

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
                    if any(map(lambda x: cui.startswith(x), ["C", "D"])):
                        cui = "MESH:"+ cui
                    elif ":" not in cui:
                        cui = "OMIM:" + cui

                    # replace alternative cuis 
                    cui = alt_ids2cui.get(cui, cui) 

                elif len(line) == 7: # Composite mention
                    pmid, start, end, mention, _class, cui, composite_mentions = line
                    if any(map(lambda x: x in cui, ["|", "+"])) and _class.lower() in [entity_type.lower()]:
                        composite +=1
                    continue
                else:
                    continue
                
                # verify if the mention is_oov
                if cui in cui_set:
                    clean_mention = mention.strip().replace("\n", " ")
                    if clean_mention:
                        query = pmid + "||" + start +"|" + end + "||" + _class + "||" + clean_mention + "||" + cui
                        queries.append(query)
                else:
                    if _class.lower() in [entity_type.lower()]:
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
        
        parent_dir = output_dir.parent  # parent directory
        
        with open(parent_dir / f"statistics_{split}.json", "w") as file:

            json.dump({
                "docs":num_docs,
                "filtered mentions": f"{num_queries} ({num_queries/total_mentions * 100:.2f})",
                "composite": f"{composite} ({composite/total_mentions * 100:.2f})",
                "oov": f"{oov} ({oov/total_mentions * 100:.2f})",
                "total mentions": total_mentions
            }, file, indent=4)

        print("{} {} {}".format(output_dir, num_docs,num_queries))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                    help='path to the corpus files')
    parser.add_argument('--output_dir', type=str,
                    help='path of output directionary')
    parser.add_argument('--type', type=str, choices=["chemical", "disease"])

    args = parser.parse_args()
    
    main(args)

    # --- Merge train and dev folders into traindev --

    train_dir = os.path.join(args.output_dir, "train")
    dev_dir = os.path.join(args.output_dir, "dev")
    traindev_dir = os.path.join(args.output_dir, "traindev")
    os.makedirs(traindev_dir, exist_ok=True)

    # copy all files from train and dev into traindev
    for src_dir in [train_dir, dev_dir]:
        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(traindev_dir, fname)
            shutil.copy2(src_path, dst_path)
