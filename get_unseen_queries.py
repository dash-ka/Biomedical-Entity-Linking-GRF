import gzip, json, re, os, pysbd, argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from glob import glob
from pathlib import Path

def get_zero_shot_mentions(test_dataset, train_dataset):
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)
    refined_set = test_df[~test_df.entity_text.str.lower().isin(train_df.entity_text)]
    return refined_set.drop_duplicates(subset=["entity_text", "cui"]).to_dict('records')

def get_zero_shot_concepts(test_dataset, train_dataset):
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)
    refined_set = test_df[~test_df.cui.isin(train_df.cui)]
    return refined_set.drop_duplicates(subset=["entity_text", "cui"]).to_dict('records')

def read_annotation_file(ann_file_path):
    data = []
    with open(ann_file_path, encoding='utf-8') as input_stream:
        for row_id, line in enumerate(input_stream):
            if not line.strip() :
                continue
            splitted_line = line.strip().split('||')
            mention = splitted_line[-2]
            concept_id = splitted_line[-1]
            offsets = splitted_line[1].strip().split('|')
            data.append({'entity_text': mention.lower(), 'cui':concept_id, "start": offsets[0], "end": offsets[-1], "pmid":splitted_line[0]})
        
    return data


def read_dataset(dataset_folder):
    ann_file_pattern: str = os.path.join(dataset_folder, '*.concept')
    dataset= []
    for ann_file_path in glob(ann_file_pattern):
        dataset += read_annotation_file(ann_file_path)
    return dataset


def recalculate_spans(sentences, entities):

        abstract_data = list()
        
        # split into sentences
        current_position = 0
        sentence_spans = []
        for sentence in sentences:
            sentence_length = len(sentence)
            sentence_spans.append((current_position, current_position + sentence_length))
            current_position += sentence_length # + 1
        
        # re-annotate at the sentence level
        for i, (sentence_start, sentence_end) in enumerate(sentence_spans):
            sentence_text = sentences[i]
            
            annotations = []
            for entity in entities:

                entity_cui = entity.get("cui")
                entity_start = int(entity["start"])
                entity_end = int(entity["end"])
                entity_mention = entity["mention"]
                entity_type = entity.get("type").lower() if "type" in entity else None

                if sentence_start <= entity_start < sentence_end:
                    new_start = entity_start - sentence_start
                    new_end = entity_end - sentence_start
                    if sentence_text[new_start:new_end]!=entity_mention:
                         
                         print(sentences)
                         print(sentence_text)
                         print(entity_start, entity_end, entity_cui)
                         print(new_start, new_end, current_position)
                         print(sentence_start, sentence_end)
                         print(sentence_text[new_start:new_end], entity_mention)
                         #raise Exception
                    annotations.append({
                        "cui": entity_cui,
                        "start": str(new_start),
                        "end": str(new_end),
                        "mention": entity_mention,
                        "type": entity_type
                    })

                
            abstract_data.append({
                "text": sentence_text,
                "annotations": annotations
            })

        return abstract_data


def merge_bad_splits(sentences, special_starts=None):
    """
    Merge sentences where the first ends with '.' and the next starts with a lowercase letter.
    """
    merged = []
    i = 0
    if special_starts is None:
        special_starts = set()
    else:
        special_starts = {w.lower() for w in special_starts}
    mutation_start_pattern = re.compile(r"^[ACGT]>[ACGT]")
    while i < len(sentences):
        # If not the last sentence and the bad split pattern matches
        
        if (
            i < len(sentences) - 1 and sentences[i+1].strip()
            and (
                 sentences[i].strip().endswith(".") 
                    and (sentences[i+1].lstrip()[0].islower() 
                         or any(sentences[i+1].lstrip().lower().startswith(s) for s in special_starts)
                         ) 
                or (mutation_start_pattern.match(sentences[i+1].lstrip())))
        ):
            # Merge them with a space
            merged.append(sentences[i].rstrip() + " " + sentences[i+1].lstrip())
            i += 2  # Skip next sentence (already merged)
        else:
            merged.append(sentences[i])
            i += 1
    return merged

def split_and_respan(json_obj, use_spacy=True, merge_bad=False):
            
    text = json_obj["text"]
    entities = json_obj["annotations"]
    if use_spacy:
        seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
        sentences = seg.segment(text)
        if sentences:
            first_sentence = sentences[0]
            if first_sentence.start != 0:
                sentences[0].sent = " "*first_sentence.start + first_sentence.sent
        sentences = [s.sent for s in sentences]

        # only for s800
        if merge_bad:
            sentences = merge_bad_splits(sentences, special_starts=["JA12", "Guimeiren","Tochiotome","Sachinoka", "Benihoppe", "C>T"])

    else:
        re_sentences = re.split(r'(?<=[.]) (?=[A-Z])', text)
        sentences = [sentence for sentence in re_sentences if sentence]
    
    try:
        abstract_parsed = recalculate_spans(sentences, entities)
    except:
        print("Problem with recalculating the spans.")

    entity_level_data = []
    for item in abstract_parsed:
        for a in item["annotations"]:
            if item["text"]:
                entity_level_data.append(
                    {
                        "entity_text" : a["mention"].lower(), 
                        "cui" : a["cui"],
                        "start": int(a["start"]),
                        "end": int(a["end"]),
                        "sentence": item["text"]
                    })
            else:
                print(item)
            
    return entity_level_data

def main(args):

    dataset_folder = args.input_dir # "/submission/datasets/s800/processed_data"
    output_dir = Path(dataset_folder).parent / "predictions"
    stats_path = Path(dataset_folder)/ "statistics_test.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_folder = Path(dataset_folder)
    test_folder = dataset_folder / "test"

    test = read_dataset(dataset_folder / "test")
    traindev = read_dataset(dataset_folder /"traindev")
    
    unseen_concepts = get_zero_shot_concepts(test, traindev)
    unseen_mentions = get_zero_shot_mentions(test, traindev)

    # Read, update, write back
    with stats_path.open() as file:
        stats = json.load(file)
        stats["zero_shot_mentions"] = len(unseen_mentions)
        stats["zero_shot_concepts"] = len(unseen_concepts)

    with stats_path.open("w") as file:
        json.dump(stats, file, indent=4)

    RESPAN = True
    test_data = []
    
    for file in glob(os.path.join(test_folder, '*.txt')):

        with open(file, encoding="utf-8") as f:
            text = f.read()

        filename = os.path.basename(file)
        annotations = []

        with open(os.path.join(test_folder, filename.replace(".txt", ".concept")), encoding="utf-8") as f:
            for line in f.readlines():
                if not line.strip() :
                    continue
                pid, position, role, mention, cuis = line.strip().split("||")
                annotations.append({
                        "cui": cuis,
                        "start": position.split("|")[0],
                        "end": position.split("|")[1],
                        "mention": mention
                    })
        if RESPAN:
            merge_bad = True if "s800" in str(test_folder) else False
            entity_level_data = split_and_respan(
                {"text": text, "annotations": annotations}, 
                merge_bad=merge_bad
                )
            test_data.extend(entity_level_data)
        else:
            for a in annotations:
                test_data.append(
                    {
                        "entity_text":a["mention"].lower(),
                        "cui": a["cui"], 
                        "start":a["start"],
                        "end":a["end"], 
                        "sentence": text
                        }
                    )
                
    contextualized_test = pd.DataFrame(test_data)

    unseen_queries = pd.DataFrame(unseen_mentions)[["entity_text", "cui"]]\
        .merge(contextualized_test, how="left", on=["entity_text", "cui"])\
            .drop_duplicates(subset=["entity_text", "cui"]).to_dict("records")
    
    with open(output_dir / "queries.json", "w", encoding="utf-8") as file:
        json.dump(unseen_queries, file, indent=4)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                    help='path to the processed test folder')
    args = parser.parse_args()
    
    main(args)