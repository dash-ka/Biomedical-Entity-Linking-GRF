#!/usr/bin/env python

import argparse, json, os, shutil
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

def process_passage( passage):

    p_offset = passage["offset"]
    p_text = passage["text"]

    annotation_list = []
    for a in passage["annotations"]:
        a_type = a["infons"].get("type")
        if a_type.lower() == "chemical": 
            mesh_id = a["infons"].get("identifier")
            loc = a.get("locations", [])
            if loc:
                offset = loc[0].get("offset") 
                length = loc[0].get("length") 
                if a["text"].strip():
                    annotation_list.append(
                    {
                        "type": a_type,
                        "id": mesh_id,
                        "start": offset - p_offset,
                        "end": offset + length - p_offset,
                        "mention": a["text"]
                    }
                )

    return str(p_offset), p_text, annotation_list

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

    # load alt_ids2cui
    with open(os.path.join(dictionary_dir, "alt_ids2cui.json")) as f:
        alt_ids2cui = json.load(f)

    # load cui_set
    cui_set = load_cui_set(os.path.join(dictionary_dir, "train_dictionary.txt"))
    print("CUI set size: ", len(cui_set))
    
    stats = defaultdict(lambda:defaultdict(int))

    for split_file in os.listdir(Path(input_dir)/"Corpus"):

        with open(os.path.join(input_dir, "Corpus", split_file), encoding="utf-8") as file:
            content = json.load(file)

        num_docs = 0
        num_queries = 0
        split = split_file.split(".")[0]
        for doc in tqdm(content["documents"], total=len(content["documents"]), desc=f"Processing {split} split..."):
            
            header = doc["passages"][0]
            pmid = header["infons"]["article-id_pmid"]

            for passage_idx, passage in enumerate(doc["passages"]):
                
                queries = []
                offset, context, annotations = process_passage(passage)   
                for a in annotations:
                    cui = a["id"]
                    mention = a["mention"]
                    start, end = a["start"], a["end"]
                    etype = a["type"] 

                    # avoid adding composite queries
                    if any(map(lambda x: x in cui, ["|", "+", ",", ";"])):
                        stats[split]["composite"] +=1
                        continue
                    
                    # replace alternative cuis 
                    cui = alt_ids2cui.get(cui, cui) 

                    if cui in cui_set:
                        clean_mention = mention.strip().replace("\n", " ")
                        if clean_mention:
                            query = f"{pmid}||{start}|{end}||{etype}||{clean_mention}||{cui}"
                            queries.append(query)
                    else:
                        stats[split]["oov"] +=1

                concept = "\n".join(queries) + "\n"
                stats[split]["n_mentions"] += len(queries)

                os.makedirs(os.path.join(output_dir, split), exist_ok=True)
                output_context_file = os.path.join(output_dir, split, "{}.txt".format("_".join([pmid, str(passage_idx)])))
                output_concept_file = os.path.join(output_dir, split, "{}.concept".format("_".join([pmid, str(passage_idx)])))
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
    
    print("{} {}".format(output_dir, num_docs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, required=True,
                    help='path of input file')
    parser.add_argument('--dictionary_dir', type=str, required=True,
                    help='path to the folder with train_dictionary.txt')
    parser.add_argument('--output_dir', type=str, required=True,
                    help='path of output directionary')
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
