#!/usr/bin/env python

import os, json, re
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

def parse_gene_nlm(path):

    corpus = dict()
    for item in list(Path(path).iterdir()):
        tree = ET.parse(item)
        root = tree.getroot()

        if len(root.findall("document")) > 1:
            print(f"Warning: more than one docs in the file {item.name}")

        for document in root.findall("document"):
            doc_id = document.findtext("id")
    
            texts, annotations = [], []
            max_len = 0
            for passage in document.findall("passage"):
                passage_type = passage.findtext("infon[@key='type']")
                offset_p = int(passage.findtext("offset"))
                texts.append(passage.findtext("text"))
                #if "isoform" in passage.findtext("text"):
                #    print(doc_id)

                for annotation in passage.findall("annotation"):
                    ann_type = annotation.findtext("infon[@key='type']")
                    location = annotation.find("location")
                    offset = int(location.attrib["offset"])
                    length = int(location.attrib["length"])
                    mention = annotation.findtext("text")
                    identifier = annotation.findtext("infon[@key='NCBI Gene identifier']") 
                    
                    if (ann_type.lower() != "other")  and (identifier is not None) and ("-" not in identifier):
                        annotations.append({
                        "cui": identifier.strip(),
                        "type": ann_type,
                        "start": offset ,
                        "end": offset + length,
                        "mention": mention
                    })

            corpus[doc_id] = {
                "text": " ".join(texts),
                "annotations": annotations
                }
            
            if len(" ".join(texts)) > max_len:
                max_len = len(" ".join(texts))
    
    print(f"Parsed {len(corpus)} items.")
    return corpus

def main(args):
    input_dir = args.corpus_dir
    dictionary_dir = args.dictionary_dir
    main_dir = args.output_dir    
    
    if args.human_only:
        alt_ids_file_name = "alt_ids2cui_human.json" 
    elif args.qualify:
        alt_ids_file_name = "alt_ids2cui.json" 
    else:
        alt_ids_file_name = "alt_ids2cui.json"

    with open(os.path.join(dictionary_dir, alt_ids_file_name)) as f:
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

    cui_set = load_cui_set(os.path.join(dictionary_dir, dictionary_file_name))
    print("CUI set size: ", len(cui_set))

    # loading knowledge base
    with open(os.path.join(dictionary_dir, "kb.json")) as file:
        kb = json.load(file)

    # load corpus
    for split in ["Train", "Test"]:
        
        nlm_gene = os.path.join(input_dir, "Corpus", split)
        gene_corpus = parse_gene_nlm(nlm_gene)

        output_dir = Path(main_dir) / "traindev" if split in ["Train"] else  Path(main_dir) / "test" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # set parameters
        human_only = args.human_only
        num_docs = 0
        num_queries = 0
        composite, oov = 0, 0
        tax2gene = defaultdict(int)

        for pmid in tqdm(gene_corpus):

            # processing queries
            queries = []
            for item in gene_corpus[pmid]["annotations"]:

                mention = item["mention"]
                start, end = str(item["start"]), str(item["end"])
                cui = item["cui"].strip("-")
                for sep in [";", "-"]:
                    cui = re.sub(sep, ",", cui)
                
                if human_only:
                    clean_cuis = [c for c in cui.split(",") if kb.get(c, {}).get("organism", "").lower() in ["homo sapiens"]]
                    if not clean_cuis:
                        continue
                else:
                    clean_cuis = cui.split(",")

                clean_cuis = [alt_ids2cui.get(c, c) for c in clean_cuis]
                    
                if len(clean_cuis) > 1:
                    composite += 1
                    continue

                cui = clean_cuis[0]
                if cui in cui_set:
                    clean_mention = mention.strip().replace("\n", " ")
                    taxon = kb.get(cui, {}).get("organism", "").lower()
                    if args.qualify: 
                        clean_mention = f"{clean_mention} ({taxon})"
                    query = pmid + "||" + start +"|" + end + "||" + taxon + "||" + clean_mention + "||" + cui
                    tax2gene[taxon] +=1
                    queries.append(query)
                else:
                    oov += 1

            context = gene_corpus[pmid]["text"]
            concept = "\n".join(queries) + "\n"
            output_context_file = output_dir / "{}.txt".format(pmid)
            output_concept_file = output_dir / "{}.concept".format(pmid)

            with open(output_context_file, 'w', encoding="utf-8") as f:
                f.write(context)
            with open(output_concept_file, 'w', encoding="utf-8") as f:
                f.write(concept)
            
            num_docs +=1
            num_queries += len(queries)

        total_mentions = num_queries + composite + oov
        with open(output_dir / f"statistics_{split}.json", "w") as file:

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
    parser.add_argument('--corpus_dir', type=str, required=True,
                    help='path to the folder with corpus files')
    parser.add_argument('--dictionary_dir', type=str, required=True,
                    help='path to the forder with train_dictionary.txt')
    parser.add_argument('--output_dir', type=str,
                    help='path of output directionary')
    parser.add_argument('--human_only', action="store_true",
                        help="If specified, only include human genes")
    parser.add_argument('--qualify', action="store_true",
                        help="If specified, qualify mentions with organism name")
    args = parser.parse_args()
    
    main(args)
