import numpy as np
import pandas as pd
import torch, json, faiss, random, os, re
from time import time
from tqdm import tqdm
from glob import glob
from collections import defaultdict, OrderedDict
from scipy.stats import t
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    default_data_collator
)

def reciprocal_rank_fusion(results: list[list], weights=None, k=1, alpha=None):
    fused_scores = {}
    if weights is None:
        weights = [1] * len(results)

    if alpha is not None:
        assert len(results) == 2, "RRF with alpha is only supported for two rankings"
        weights = [alpha, 1 - alpha]

    for i, ranking in enumerate(results):
        if weights[i] == 0:
            continue  # Skip this ranking entirely if its weight is 0

        for rank, doc_id in enumerate(ranking):
            score = (1 / (rank + k)) * weights[i]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score

    reranked_results = [
        doc_id for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results, fused_scores
    
def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)


class NamesDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
      
    
def embed_dense(tokenizer, encoder, names, max_length, show_progress=True, prompt=None, use_cuda=True, index=None):
        """
        Embedding data into dense representations

        Parameters
        ----------
        names : np.array or list
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        encoder.eval() # prevent dropout
        batch_size=1024
        dense_embeds = []

        if isinstance(names, np.ndarray):
            names = names.tolist()        

        name_encodings = tokenizer(names, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")

        if use_cuda:
            name_encodings = name_encodings.to('cuda')

        name_dataset = NamesDataset(name_encodings)
        name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size)

        if index is not None:
            with torch.no_grad():
                for batch in tqdm(name_dataloader, disable=not show_progress, desc='embedding dictionary'):
                    outputs = encoder(**batch)
                    batch_dense_embeds = outputs[0][:, 0].cpu().detach()#.numpy()
                    index.add(batch_dense_embeds.numpy())
            print(f"Total embeddings in index: {index.ntotal}")
            return index
        else:
            with torch.no_grad():
                for batch in tqdm(name_dataloader, disable=not show_progress, desc='embedding dictionary'):
                    outputs = encoder(**batch)
                    batch_dense_embeds = outputs[0][:, 0].cpu().detach()#.numpy()
                    dense_embeds.append(batch_dense_embeds.numpy())
            dense_embeds = np.concatenate(dense_embeds, axis=0)
            return dense_embeds
        

def get_sim(dense_embeddings):

    norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
    normalized_embeds = dense_embeddings / norms
    similarity_matrix = np.matmul(normalized_embeds, normalized_embeds.T)
    sims = similarity_matrix[0] 
    return sims

def retrieve_candidate(score_matrix, topk):
        
    def indexing_2d(arr, cols):
        rows = np.repeat(np.arange(0,cols.shape[0])[:, np.newaxis],cols.shape[1],axis=1)
        return arr[rows, cols]

    # get topk indexes without sorting
    topk_idxs = np.argpartition(score_matrix,-topk)[:, -topk:]

    # get topk indexes with sorting
    topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
    topk_argidxs = np.argsort(-topk_score_matrix) 
    topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

    return topk_idxs


def compute_metrics(query_rankings, golden_cuis, cutoffs, verbose=True):

    cutoffs = [1, 2, 3, 5, 10, 15, 20]
    recall_at_k = {k: [] for k in cutoffs}
    ndcg_at_k = {k: [] for k in cutoffs}
        
    for ranking_cuis, golden_cui in zip(query_rankings, golden_cuis):
    
        # compute recall@k
        for k_val in cutoffs:
            hits = np.intersect1d(golden_cui, ranking_cuis[:k_val]) 
            recall_at_k[k_val].append(len(hits) / max(min(k_val, len(golden_cui)), 1))

        # compute ndcg@k
        def compute_dcg_at_k(relevances, k):
            dcg = 0
            for i in range(min(len(relevances), k)):
                dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
            return dcg
        
        for k_val in cutoffs:
                predicted_relevance = [
                    1 if cui in golden_cui else 0 for cui in ranking_cuis[:k_val]
                ]
                true_relevances = [1] * len(golden_cui)
                ndcg = compute_dcg_at_k(predicted_relevance, k_val)
                indcg = compute_dcg_at_k(true_relevances, k_val)
                ndcg_at_k[k_val].append(ndcg / indcg)

    # Compute averages
    for k in recall_at_k:
        recall_at_k[k] = np.mean(recall_at_k[k] )

        if verbose:
            print(f"recall@{k}: {recall_at_k[k]}")

    for k in ndcg_at_k:
        ndcg_at_k[k] = np.mean(ndcg_at_k[k])

        if verbose:
            print(f"ndcg@{k}: {ndcg_at_k[k]}")

    return {
            "recall@k": recall_at_k,
            "ndcg@k": ndcg_at_k,
        }


def get_recall(predictions, ground_truth, cutoffs):

    def compute_recall(preds, truths, cutoffs):
        recalls = np.zeros(len(cutoffs))
        
        for (pred_cuis, true_cui) in zip(preds, truths):
            for i, c in enumerate(cutoffs):
                hits = np.intersect1d(true_cui, pred_cuis[:c])                    
                recalls[i] += len(hits) / max(min(c, len(true_cui)), 1)
        recalls /= len(preds)
        return recalls

    recalls, missed = compute_recall(predictions, ground_truth, cutoffs)
    for i, c in enumerate(cutoffs):
        print(f"recall@{c}: {recalls[i]}")

    return missed, {c:recalls[i] for i, c in enumerate(cutoffs)}


model_name_or_path = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
#model_name_or_path = "BioSyn/tmp/biosyn-sapbert-bc5cdr-disease-grf"
#model_name_or_path = 'BAAI/bge-base-en-v1.5'
#model_name_or_path = 'dmis-lab/biobert-base-cased-v1.1'


encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
encoder = encoder.to("cuda")

# ---- Experiment Settings ----
embed_dim = 768
cutoffs = [1, 2, 3, 5, 10, 15, 20]
sampling_ranges = [10]
alphas = [0.5] #np.round(np.arange(0, 1.1, 0.1), 1)
num_runs = 5
k = 40
use_faiss = True


#main_folder = Path("../prime/llm_gen/gemma14b/s800") #-mesh
main_folder = Path("BioSyn/SapBERT/gpt4o/ncbi-disease")
#QUERIES_PATH = main_folder / "test_unseen_standard_name"
QUERIES_PATH = main_folder / "raw_queries.json"

# loading training dictionary
with open("BioSyn/SapBERT/ncbi-disease/train_dictionary.txt") as file:
    dictionary = file.readlines()

cui2names = defaultdict(list)
name2cui = dict()
for line in dictionary:
    cui, name = line.split("||")
    cui2names[cui].append(name.strip())
    name2cui[name.strip()] = cui

# Eval dictionary & embeddings
eval_dictionary = np.array(list(name2cui.items()))
index = faiss.index_factory(embed_dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
index = embed_dense(tokenizer, encoder, list(name2cui.keys()), max_length=50, use_cuda=True, index=index)


def get_feedback(query: dict, fk: str, depth: int):
    """
    Extracts feedback from a query dictionary based on the feedback key (fk).
    
    Parameters:
        query (dict): Dictionary with keys like 'synonyms', 'definition', 'standard_name'.
        fk (str): Feedback key ('synonyms', 'definition', 'standard_name').
        depth (int): Number of items to sample when fk == 'synonyms'.
        
    Returns:
        str | list: Cleaned feedback (string for 'definition'/'standard_name', 
                    list for 'synonyms').
    """
    if fk == "synonyms":
        
        raw_synonyms = query.get("synonyms", "")
        entity_text = query.get("entity_text", "")
        feedback_synonyms = [
            re.sub(r"\n", " ", s.strip()) if s.strip() else entity_text
            for s in raw_synonyms.split(":")
        ][:20]  # cap at 20
        
        n_available = len(feedback_synonyms)
        n_sample = min(depth, n_available)
        
        if n_available < depth:
            print(f"WARNING: Subsampling due to insufficient elements: {n_available}")
        
        feedback = random.sample(feedback_synonyms, n_sample)
    
    elif fk in ("definition", "standard_name"):
    
        text = query.get(fk, "")
        feedback = re.sub(r"\n", " ", text.strip())
    
    else:
        raise ValueError(f"Unsupported feedback key: {fk}")
    
    return feedback

def apply_rank_grf(queries_list, feedback_keys, alpha, depth=None, k=None):
    
    """
    Apply rank-based feedback for entity linking.

    Parameters:
        queries_list (list[dict]): List of queries containing at least 'entity_text' and 'cui'.
        feedback_keys (str): Feedback keys separated by '+', e.g. "synonyms+definition".
        alpha (float): Weight for mention embedding (used in Rocchio).
        depth (int, optional): Depth for sampling feedback (only used for synonyms).
        k (int): Top-k candidates to retrieve from index.

    Returns:
        candidate_rankings (list[list[str]]): Ranked list of unique candidate CUIs per query.
        gold_cuis (list[list[str]]): Corresponding gold CUIs.
    """

    # needs eval_dictionary, tokenizer, encoder, index
    candidate_rankings, gold_cuis = [], []
    for query in tqdm(queries_list, desc=f"Running RF with alpha={alpha} feedback depth={depth}"):

        feedback_embeddings = []
        for fk in feedback_keys.split("+"):

            feedback = get_feedback(query, fk, depth) # either a str or a list
            if isinstance(feedback, list):
                assert fk == "synonyms"
                embeddings = embed_dense(tokenizer, encoder, feedback, max_length=50, show_progress=False)
                embeddings = np.mean(embeddings, axis=0, keepdims=True) # (1, d)
            elif isinstance(feedback, str):
                embeddings = embed_dense(tokenizer, encoder, feedback, max_length=50, show_progress=False)
            else:
                raise ValueError(f"Unsupported feedback type: {type(feedback)}")
            
            # store as 1D vector
            feedback_embeddings.append(embeddings[0]) 
        
        # Compute mention embedding
        mention_embedding = embed_dense(tokenizer, encoder, query["entity_text"], max_length=50, show_progress=False)[0]
        mention_embedding = mention_embedding.reshape(1, -1)

        all_embeddings = np.concatenate([mention_embedding, feedback_embeddings], axis=0)
          
        # Search index
        dists, candidate_idxs = index.search(all_embeddings, k=k)

        ranked_lists = []
        for query_candidate_idx in candidate_idxs:

            candidates = eval_dictionary[query_candidate_idx].squeeze() # ranked candidate list
        
            # Deduplicate CUIs while preserving ranking order
            golden_cui = query["cui"]
            unique_candidates = OrderedDict() 
            for name, cui in candidates:
                if cui not in unique_candidates:
                    unique_candidates[cui] = {
                        "name": name,
                        "cui": cui,
                        "label": check_label(cui, golden_cui)
                    }
            ranked_lists.append([c for c in unique_candidates])

        # Fuse all ranked list via RRF
        fused_ranking, _ = reciprocal_rank_fusion(ranked_lists)
        candidate_rankings.append(fused_ranking)
        gold_cuis.append([golden_cui])

    return candidate_rankings, gold_cuis


def apply_vector_grf(queries_list, feedback_keys, alpha, depth=None, rocchio=False, k=None):
    
    """
    Apply vector-based feedback (average or Rocchio) for entity linking.

    Parameters:
        queries_list (list[dict]): List of queries containing at least 'entity_text' and 'cui'.
        feedback_keys (str): Feedback keys separated by '+', e.g. "synonyms+definition".
        alpha (float): Weight for mention embedding (used in Rocchio).
        depth (int, optional): Depth for sampling feedback (only used for synonyms).
        rocchio (bool): If True, use Rocchio weighting; else simple average.
        k (int): Top-k candidates to retrieve from index.

    Returns:
        candidate_rankings (list[list[str]]): Ranked list of unique candidate CUIs per query.
        gold_cuis (list[list[str]]): Corresponding gold CUIs.
    """

    # needs eval_dictionary, tokenizer, encoder, index
    candidate_rankings, gold_cuis = [], []
    for query in tqdm(queries_list, desc=f"Running VF-{'rocchio' if rocchio else 'average'} with alpha={alpha} feedback depth={depth}"):

        feedback_embeddings = []
        for fk in feedback_keys.split("+"):

            feedback = get_feedback(query, fk, depth) # either a str or a list
            if isinstance(feedback, list):
                assert fk == "synonyms"
                embeddings = embed_dense(tokenizer, encoder, feedback, max_length=50, show_progress=False)
                embeddings = np.mean(embeddings, axis=0, keepdims=True) # (1, d)
            elif isinstance(feedback, str):
                embeddings = embed_dense(tokenizer, encoder, feedback, max_length=50, show_progress=False)
            else:
                raise ValueError(f"Unsupported feedback type: {type(feedback)}")
            
            # store as 1D vector
            feedback_embeddings.append(embeddings[0]) 
        
        # Compute mention embedding
        mention_embedding = embed_dense(tokenizer, encoder, query["entity_text"], max_length=50, show_progress=False)[0]
        mention_embedding = mention_embedding.reshape(1, -1)

        if feedback_embeddings:
            if rocchio:
                feedback_embeddings = np.concatenate([feedback_embeddings], axis=0)
                feedback_embedding = np.mean(feedback_embeddings, axis=0, keepdims=True) 
                query_embedding = alpha * mention_embedding + (1 - alpha) * feedback_embedding
                query_embedding = query_embedding.reshape(1, -1)

            else:
                all_embeddings = np.concatenate([mention_embedding, feedback_embeddings], axis=0)
                query_embedding = np.mean(all_embeddings, axis=0).reshape(1, -1) 
        else:
            print("No feedback provided, setting the query to mention span.")
            query_embedding = mention_embedding
        
        # Search index
        dists, candidate_idxs = index.search(query_embedding, k=k)
        candidates = eval_dictionary[candidate_idxs[0]].squeeze() # ranked candidate list
        
        # Deduplicate CUIs while preserving ranking order
        golden_cui = query["cui"]
        unique_candidates = OrderedDict() 
        for name, cui in candidates:
            if cui not in unique_candidates:
                unique_candidates[cui] = {
                    "name": name,
                    "cui": cui,
                    "label": check_label(cui, golden_cui)
                }
        candidate_rankings.append([c for c in unique_candidates])
        gold_cuis.append([golden_cui])

    return candidate_rankings, gold_cuis
    

def apply_text_grf(queries_list, feedback_keys, depth=None, k=None):

    """
    Apply text-based feedback for entity linking.

    Parameters:
        queries_list (list[dict]): List of queries containing at least 'entity_text' and 'cui'.
        feedback_keys (str): Feedback keys separated by '+', e.g. "synonyms+definition".
        depth (int, optional): Depth for sampling feedback (only used for synonyms).
        k (int): Top-k candidates to retrieve from index.

    Returns:
        candidate_rankings (list[list[str]]): Ranked list of unique candidate CUIs per query.
        gold_cuis (list[list[str]]): Corresponding gold CUIs.
    """
    
    # needs eval_dictionary, tokenizer, encoder, index
    candidate_rankings, gold_cuis = [], []
    for query in tqdm(queries_list, desc=f"Running TF with feedback depth={depth}"):

        feedbacks = []
        for fk in feedback_keys.split("+"):

            feedback = get_feedback(query, fk, depth) # either a str or a list
            if isinstance(feedback, list):
                assert fk == "synonyms"
                feedbacks.extend(feedback)
            elif isinstance(feedback, str):
                feedbacks.append(feedback)
            else:
                raise ValueError

        mention_texts = "; ".join([query["entity_text"]] + feedbacks)
        query_embedding = embed_dense(tokenizer, encoder, mention_texts, max_length=50, show_progress=False)
        query_embedding = query_embedding[0].reshape(1, -1)

        # Search index
        dists, candidate_idxs = index.search(query_embedding, k=k)
        candidates = eval_dictionary[candidate_idxs[0]].squeeze() # ranked candidate list
        
        # Deduplicate CUIs while preserving ranking order
        golden_cui = query["cui"]
        unique_candidates = OrderedDict() 
        for name, cui in candidates:
            if cui not in unique_candidates:
                unique_candidates[cui] = {
                    "name": name,
                    "cui": cui,
                    "label": check_label(cui, golden_cui)
                }
        candidate_rankings.append([c for c in unique_candidates])
        gold_cuis.append([golden_cui])

    return candidate_rankings, gold_cuis


def eval_rank_grf(queries_path, feedback_type, alphas=None, feedback_depths=None, run_seed=None, k=20):
    print(f"\n🔁 Running rank-based GRF with seed {run_seed}")
    random.seed(run_seed)

    if feedback_depths is None:
        feedback_depths = [1]

    with open(queries_path, encoding='utf-8') as file:
        queries = json.load(file)

    run_results = []

    for w in alphas:
        for depth in feedback_depths:
            candidate_rankings, gold_cuis = apply_rank_grf(
                queries, 
                feedback_keys=feedback_type, 
                alpha=w, 
                depth=depth, 
                k=k
                )
            
            metrics = compute_metrics(candidate_rankings, golden_cuis=gold_cuis, cutoffs=cutoffs)
            for c in cutoffs:
                run_results.append({
                    "strategy": "RF-GRF" + "_" + feedback_type,
                    "alpha": w,
                    "depth": depth,
                    "cutoff": c,
                    "recall@cutoff": metrics["recall@k"][c],
                    "ndcg@cutoff": metrics["ndcg@k"][c],
                    "run": run_seed
                })
    return run_results

def eval_vector_grf(queries_path, feedback_type, alphas=None, feedback_depths=None, rocchio=False, run_seed=None, k=20):
    print(f"\n🔁 Running vector-based GRF with seed {run_seed}")
    random.seed(run_seed)

    with open(queries_path, encoding='utf-8') as file:
        queries = json.load(file)

    if feedback_depths is None:
        feedback_depths = [1]

    run_results = []

    for w in alphas:
        for depth in feedback_depths:
            candidate_rankings, gold_cuis = apply_vector_grf(
                queries, 
                feedback_keys=feedback_type, 
                alpha=w, 
                depth=depth, 
                rocchio=rocchio,
                k=k
                )
            
            metrics = compute_metrics(candidate_rankings, golden_cuis=gold_cuis, cutoffs=cutoffs)
            for c in cutoffs:
                run_results.append({
                    "strategy": "VF-GRF" + "_" + feedback_type,
                    "alpha": w,
                    "depth": depth,
                    "cutoff": c,
                    "recall@cutoff": metrics["recall@k"][c],
                    "ndcg@cutoff": metrics["ndcg@k"][c],
                    "run": run_seed
                })
    return run_results

def eval_text_grf(queries_path, feedback_type, feedback_depths=None,run_seed=None, k=20):
    print(f"\n🔁 Running text-based GRF with seed {run_seed}")
    random.seed(run_seed)

    with open(queries_path, encoding='utf-8') as file:
        queries = json.load(file)

    run_results = []

    if feedback_depths is None:
        feedback_depths = [1]

    for depth in feedback_depths:
        candidate_rankings, gold_cuis = apply_text_grf(queries, feedback_keys=feedback_type, depth=depth, k=k)
            
        metrics = compute_metrics(candidate_rankings, golden_cuis=gold_cuis, cutoffs=cutoffs)
        for c in cutoffs:
            run_results.append({
                "strategy": "TF-GRF" + "_" + feedback_type,
                "alpha": -1,
                "depth": depth,
                "cutoff": c,
                "recall@cutoff": metrics["recall@k"][c],
                "ndcg@cutoff": metrics["ndcg@k"][c],
                "run": run_seed
            })
    return run_results


def grf_vector_fusion(queries_path, run_seed, index=None):
    print(f"\n🔁 Running vector-based GRF with seed {run_seed}")
    random.seed(run_seed)

    with open(queries_path, encoding='utf-8') as file:
        queries = json.load(file)

    run_results = []

    for w in alphas:
        for depth in sampling_ranges:
            true_cuis, concept_rankings = [], []

            for query_data in tqdm(queries,
                desc=f"Running VF for epoch={run_seed} with alpha={w} and feedback depth={depth}"):
                

                query = query_data["entity_text"]
                golden_cui = query_data["cui"]
                feedback_synonyms = query_data.get("synonyms")
                feedback_synonyms = [re.sub("\n", " ", s.strip()) if s.strip() else query for s in feedback_synonyms.split(":")][:20]
                
                
                try:
                    sampled_feedback = random.sample(feedback_synonyms, depth)
                except:
                    print(f"WARNING: Subsumpling for insufficient n. elements {len(feedback_synonyms)}")
                    sampled_feedback = random.sample(feedback_synonyms, len(feedback_synonyms))

                mention_texts = [query] + sampled_feedback

                # embed query and feedback 
                embeddings = embed_dense(tokenizer, encoder, mention_texts, max_length=50, show_progress=False)
                mention_embed = embeddings[0].reshape(1, -1)
                feedback_embeds = embeddings[1:]
                
                feedback_embed = np.average(feedback_embeds, axis=0, weights=None).reshape(1, -1)
                query_embeds = w * mention_embed + (1 - w) * feedback_embed

                dists, candidate_idxs = index.search(query_embeds, k=k)
                candidates = eval_dictionary[candidate_idxs[0]].squeeze()

                seen_cuis = set()
                grf_candidates = []
                for name, cui in candidates:
                    if cui not in seen_cuis:
                        seen_cuis.add(cui)
                        grf_candidates.append({
                        "name": name,
                        "cui": cui,
                        "label": check_label(cui, golden_cui)
                    })

                true_cuis.append([golden_cui])
                concept_rankings.append([c["cui"] for c in grf_candidates])
                
            metrics = compute_metrics(concept_rankings, golden_cuis=true_cuis, cutoffs=cutoffs)

            for c in cutoffs:
                run_results.append({
                    "strategy": "GRF",
                    "alpha": w,
                    "depth": depth,
                    "cutoff": c,
                    "recall@cutoff": metrics["recall@k"][c],
                    "ndcg@cutoff": metrics["ndcg@k"][c],
                    "run": run_seed
                })
    return run_results


# running...
all_results = []

for seed in tqdm(range(num_runs), total = num_runs):    
    
    #all_results.extend(grf_vector_fusion(QUERIES_PATH, seed, index=index))
    
    all_results.extend(
        eval_text_grf(
            QUERIES_PATH, run_seed=seed, 
            feedback_type="synonyms", 
            feedback_depths = [1, 3, 5, 10]
        ))
    
    all_results.extend(
         eval_vector_grf(
             QUERIES_PATH, run_seed=seed, 
             feedback_type="synonyms", 
             feedback_depths = [1, 3, 5, 10],
             alphas = [0.5],
             rocchio=True
         ))

    all_results.extend(
         eval_vector_grf(
             QUERIES_PATH, run_seed=seed, 
             feedback_type="synonyms+definition+standard_name", 
             feedback_depths = [1, 3, 5, 10],
             alphas = [0.5],
             rocchio=False
         ))
    
    all_results.extend(
         eval_rank_grf(
             QUERIES_PATH, run_seed=seed, 
             feedback_type="synonyms", 
             feedback_depths = [1, 3, 5, 10],
             alphas = [0.5]
         ))
    
    all_results.extend(
         eval_rank_grf(
             QUERIES_PATH, run_seed=seed, 
             feedback_type="synonyms+definition+standard_name", 
             feedback_depths = [1, 3, 5, 10],
             alphas = [0.5]
         ))

all_results.extend(
    eval_text_grf(
        QUERIES_PATH,  
        feedback_type="definition"
    ))

all_results.extend(
    eval_text_grf(
        QUERIES_PATH,  
        feedback_type="standard_name"
    ))

all_results.extend(
     eval_vector_grf(
         QUERIES_PATH,
         feedback_type="definition", 
         rocchio=True,
         alphas = [0.5]
     ))

all_results.extend(
     eval_vector_grf(
         QUERIES_PATH,  
         feedback_type="standard_name", 
         rocchio=True,
         alphas = [0.5, 1]
     ))

all_results.extend(
     eval_rank_grf(
         QUERIES_PATH,  
         feedback_type="definition", 
         alphas = [0.5]
     ))

all_results.extend(
     eval_rank_grf(
         QUERIES_PATH,  
         feedback_type="standard_name", 
         alphas = [0.5]
     ))

df = pd.DataFrame(all_results)

def compute_ci(series):
    n = len(series)
    mean = np.mean(series)
    std_err = np.std(series, ddof=1) / np.sqrt(n)
    return pd.Series({"mean": mean, "se": std_err})

agg_recall = (
    df.groupby(["strategy", "alpha", "depth", "cutoff"], dropna=False)["recall@cutoff"]
    .agg(mean=lambda x: np.mean(x), se=lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    .reset_index()
)
agg_recall["metric"] = "recall@cutoff"

agg_ndcg = (
    df.groupby(["strategy", "alpha", "depth", "cutoff"], dropna=False)["ndcg@cutoff"]
    .agg(mean=lambda x: np.mean(x), se=lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    .reset_index()
)
agg_ndcg["metric"] = "ndcg@cutoff"

agg_long = pd.concat([
    agg_recall[["strategy", "alpha", "depth", "cutoff", "metric", "mean", "se"]],
    agg_ndcg[["strategy", "alpha", "depth", "cutoff", "metric", "mean", "se"]]
])

agg_recall.to_csv(main_folder / "results_all_sapbert.csv", index=False)
print("✅ All strategy results with confidence intervals saved.")

