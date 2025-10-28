import numpy as np
import pandas as pd
import torch, json, faiss, random, os, re, argparse
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

def reciprocal_rank_fusion(results: list[dict], weights=None, k=1, alpha=None):
    fused_scores = {}
    if weights is None:
        weights = [1] * len(results)

    if alpha >=0:
        assert len(results) == 2, "RRF with alpha is only supported for two rankings"
        weights = [alpha, 1 - alpha]

    for i, ranking in enumerate(results):
        if weights[i] == 0:
            continue  # Skip this ranking entirely if its weight is 0

        for rank, candidate in enumerate(ranking):

            score = (1 / (rank + k)) * weights[i]
            if candidate not in fused_scores:
                fused_scores[candidate] = ranking[candidate] | {"score":0}
            fused_scores[candidate]["score"] += score

    reranked_results = {cui:candidate for cui, candidate
        in sorted(fused_scores.items(), key=lambda x: x[1]["score"], reverse=True)
                        }
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


def compute_metrics(query_rankings, golden_cuis, cutoffs=None, verbose=True):

    if cutoffs is None:
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



    

@staticmethod
def apply_rank_grf(tokenizer, encoder, index, eval_dictionary, queries_list, feedback_keys, alpha, depth=None, k=None):
    
    """
    Apply rank-based feedback for entity linking.

    Parameters:
        tokenizer: Tokenizer for the encoder model.
        encoder: Encoder model to generate embeddings.
        index: FAISS index for nearest neighbor search.
        eval_dictionary: List of (name, cui) tuples for candidates.
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
            ranked_lists.append(unique_candidates)

        # Fuse all ranked list via RRF
        fused_ranking, _ = reciprocal_rank_fusion(ranked_lists, alpha=alpha)
        candidate_rankings.append(fused_ranking)
        gold_cuis.append([golden_cui])

    return candidate_rankings, gold_cuis

@staticmethod
def apply_vector_grf(
    tokenizer, 
    encoder, 
    index, 
    eval_dictionary, 
    queries_list, 
    feedback_keys, 
    alpha, 
    depth=None, 
    rocchio=False, 
    k=None
    ):
    
    """
    Apply vector-based feedback (average or Rocchio) for entity linking.

    Parameters:
        tokenizer: Tokenizer for the encoder model.
        encoder: Encoder model to generate embeddings.
        index: FAISS index for nearest neighbor search.
        eval_dictionary: List of (name, cui) tuples for candidates.
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
        candidate_rankings.append(unique_candidates)
        gold_cuis.append([golden_cui])

    return candidate_rankings, gold_cuis
    
@staticmethod
def apply_text_grf(tokenizer, encoder, index, eval_dictionary, queries_list, feedback_keys, depth=None, k=None):

    """
    Apply text-based feedback for entity linking.

    Parameters:
        tokenizer: Tokenizer for the encoder model.
        encoder: Encoder model to generate embeddings.
        index: FAISS index for nearest neighbor search.
        eval_dictionary: List of (name, cui) tuples for candidates.
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
        candidate_rankings.append(unique_candidates)
        gold_cuis.append([golden_cui])

    return candidate_rankings, gold_cuis


class GRFExperiment:

    def __init__(self, args):
        self.model_path = args.model_path
        self.dictionary_path = args.dictionary_path
        self.queries_path = args.queries_path
        self.output_path = Path(args.output_path)
        if not os.path.exists(self.output_path.parent):
            os.makedirs(self.output_path.parent)
        self.use_cuda = args.use_cuda

        # load model
        self.encoder = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.use_cuda:
            self.encoder = self.encoder.to("cuda")
        self.embed_dim = 768

        # load terminology
        with open(self.dictionary_path) as file:
            dictionary = file.readlines()

        cui2names = defaultdict(list)
        name2cui = dict()
        for line in dictionary:
            cui, name = line.split("||")
            cui2names[cui].append(name.strip())
            name2cui[name.strip()] = cui

        
        self.eval_dictionary = np.array(list(name2cui.items()))

        # create faiss index and embed terminology
        index = faiss.index_factory(self.embed_dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
        self.index = embed_dense(self.tokenizer, self.encoder, list(name2cui.keys()), max_length=50, use_cuda=self.use_cuda, index=index)
        

    def eval_rank_grf(self,
            feedback_type, 
            alphas=None, 
            feedback_depths=None, 
            run_seed=None, k=20,
            save_predictions=False
            ):
        print(f"\n🔁 Running rank-based GRF with seed {run_seed}")
        random.seed(run_seed)

        if feedback_depths is None:
            feedback_depths = [1]

        if alphas is None:
            alphas = [-1]

        with open(self.queries_path, encoding='utf-8') as file:
            queries = json.load(file)

        run_results = []

        for w in alphas:
            for depth in feedback_depths:
                candidate_rankings, gold_cuis = apply_rank_grf(
                    self.tokenizer, self.encoder, self.index, self.eval_dictionary,
                    queries, 
                    feedback_keys=feedback_type, 
                    alpha=w, 
                    depth=depth, 
                    k=k
                    )
                
                cuis_rankings = []
                for ranking in candidate_rankings:
                    cuis_rankings.append([cui for cui in ranking])
                
                cutoffs = [1, 2, 3, 5, 10, 15, 20]  
                metrics = compute_metrics(cuis_rankings, golden_cuis=gold_cuis, cutoffs=cutoffs)
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
        if save_predictions:
            folder = Path(self.output_path).parent
            pred_folder =  folder/"predictions"
            if not os.path.exists(pred_folder):
                os.makedirs(pred_folder)

            ft = feedback_type.replace("+", "_")
            with open(pred_folder/f"rank_{ft}.json", "w", encoding="utf-8") as file:
                json.dump(candidate_rankings, file, indent=4)

        return run_results

    def eval_vector_grf(self, 
            feedback_type, 
            alphas=None, 
            feedback_depths=None, 
            rocchio=False, 
            run_seed=None, k=20,
            save_predictions=False
            ):
        
        print(f"\n🔁 Running vector-based GRF with seed {run_seed}")
        random.seed(run_seed)

        with open(self.queries_path, encoding='utf-8') as file:
            queries = json.load(file)

        if feedback_depths is None:
            feedback_depths = [1]

        run_results = []

        for w in alphas:
            for depth in feedback_depths:
                candidate_rankings, gold_cuis = apply_vector_grf(
                    self.tokenizer, self.encoder, self.index, self.eval_dictionary,
                    queries, 
                    feedback_keys=feedback_type, 
                    alpha=w, 
                    depth=depth, 
                    rocchio=rocchio,
                    k=k
                    )
                
                cuis_rankings = []
                for ranking in candidate_rankings:
                    cuis_rankings.append([cui for cui in ranking])
                
                cutoffs = [1, 2, 3, 5, 10, 15, 20]
                metrics = compute_metrics(cuis_rankings, golden_cuis=gold_cuis, cutoffs=cutoffs)
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
        if save_predictions:
            folder = Path(self.output_path).parent
            pred_folder =  folder/"predictions"
            if not os.path.exists(pred_folder):
                os.makedirs(pred_folder)

            ft = feedback_type.replace("+", "_")
            with open(pred_folder/f"vector_{ft}.json", "w", encoding="utf-8") as file:
                json.dump(candidate_rankings, file, indent=4)
        
        return run_results

    def eval_text_grf(self, feedback_type, feedback_depths=None, run_seed=None, k=20):
        print(f"\n🔁 Running text-based GRF with seed {run_seed}")
        random.seed(run_seed)

        with open(self.queries_path, encoding='utf-8') as file:
            queries = json.load(file)

        run_results = []

        if feedback_depths is None:
            feedback_depths = [1]

        for depth in feedback_depths:
            candidate_rankings, gold_cuis = apply_text_grf(
                self.tokenizer, self.encoder, self.index, self.eval_dictionary,
                queries, feedback_keys=feedback_type, depth=depth, k=k
            )
            
            cuis_rankings = []
            for ranking in candidate_rankings:
                cuis_rankings.append([cui for cui in ranking])

            cutoffs = [1, 2, 3, 5, 10, 15, 20]
            metrics = compute_metrics(cuis_rankings, golden_cuis=gold_cuis, cutoffs=cutoffs)
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


def main(args):
    num_runs = args.num_runs
    output_path = args.output_path
    
    grf = GRFExperiment(args)
    # running...
    all_results = []

    for seed in tqdm(range(num_runs), total = num_runs):    
        
        all_results.extend(
            grf.eval_text_grf(
                run_seed=seed, 
                feedback_type="synonyms", 
                feedback_depths = [1, 3, 5, 10]
            ))
        
        all_results.extend(
            grf.eval_vector_grf(
                run_seed=seed, 
                feedback_type="synonyms", 
                feedback_depths = [5, 10],
                alphas = [0.5],#np.round(np.arange(0, 1.1, 0.1), 1),#[0.5],
                rocchio=True
            ))

        all_results.extend(
            grf.eval_vector_grf(
                run_seed=seed, 
                feedback_type="synonyms+definition+standard_name", 
                feedback_depths = [10],#[1, 3, 5, 10],
                alphas = [0.5],
                rocchio=False,
                save_predictions=False
            ))
        
        #all_results.extend(
        #     eval_vector_grf(
        #         run_seed=seed, 
        #         feedback_type="synonyms+definition", 
        #         feedback_depths = [10],
        #         alphas = [0.5],
        #         rocchio=False,
        #         save_predictions=False
        #     ))

        #all_results.extend(
        #     eval_vector_grf(
        #         run_seed=seed, 
        #         feedback_type="synonyms+standard_name", 
        #         feedback_depths = [10],
        #         alphas = [0.5],
        #         rocchio=False,
        #         save_predictions=False
        #     ))

        
        all_results.extend(
            grf.eval_rank_grf(
                run_seed=seed, 
                feedback_type="synonyms", 
                feedback_depths = [5, 10],
                alphas = [0.5]#np.round(np.arange(0, 1.1, 0.1), 1)#[0.5]
            ))
        
        all_results.extend(
            grf.eval_rank_grf(
                run_seed=seed, 
                feedback_type="synonyms+definition+standard_name", 
                feedback_depths = [10],#[1, 3, 5, 10],
                save_predictions=False
            ))

        #all_results.extend(
        #     grf.eval_rank_grf(
        #         run_seed=seed, 
        #         feedback_type="synonyms+definition", 
        #         feedback_depths = [10],
        #         save_predictions=False
        #     ))

        #all_results.extend(
        #     grf.eval_rank_grf(
        #         run_seed=seed, 
        #         feedback_type="synonyms+standard_name", 
        #         feedback_depths = [10],
        #         save_predictions=False
        #     ))

    #all_results.extend(
    #     grf.eval_vector_grf(
    #         run_seed=seed, 
    #         feedback_type="definition+standard_name", 
    #         feedback_depths = [10],
    #         alphas = [0.5],
    #         save_predictions=False
    #     ))

    #all_results.extend(
    #         grf.eval_rank_grf(
    #         run_seed=seed, 
    #         feedback_type="definition+standard_name", 
    #         feedback_depths = [10],
    #         save_predictions=False
    #     ))

    all_results.extend(
        grf.eval_text_grf(
            feedback_type="definition"
        ))

    all_results.extend(
        grf.eval_text_grf(
            feedback_type="standard_name"
        ))

    all_results.extend(
        grf.eval_vector_grf(
            feedback_type="definition", 
            rocchio=True,
            alphas = [0.5]#np.round(np.arange(0, 1.1, 0.1), 1)#[0.5]
        ))

    all_results.extend(
        grf.eval_vector_grf(
            feedback_type="standard_name", 
            rocchio=True,
            alphas = [0.5]#np.round(np.arange(0, 1.1, 0.1), 1)#[0.5]
        ))

    all_results.extend(
        grf.eval_rank_grf(  
            feedback_type="definition", 
            alphas = [0.5]#np.round(np.arange(0, 1.1, 0.1), 1)#[0.5]
        ))

    all_results.extend(
        grf.eval_rank_grf( 
            feedback_type="standard_name", 
            alphas = [0.5]#np.round(np.arange(0, 1.1, 0.1), 1)#[0.5]
        ))

    df = pd.DataFrame(all_results)

    agg_recall = (
        df.groupby(["strategy", "alpha", "depth", "cutoff"], dropna=False)["recall@cutoff"]
        .agg(mean=lambda x: np.mean(x), se=lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
        .reset_index()
    )
    agg_recall["metric"] = "recall@cutoff"

    if not os.path.exists(output_path.parent):
        os.makedirs(output_path.parent)

    agg_recall.to_csv(output_path, index=False)
    print("✅ All strategy results with confidence intervals saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                    help='path to the folder with trained model')
    parser.add_argument('--num_runs', type=int, default=5,
                    help='number of runs with different random seeds')
    parser.add_argument('--queries_path', type=str, required=True,
                    help='path of the json file with queries with feedback')
    parser.add_argument('--dictionary_path', type=str, required=True,
                    help='path of the train_dictionary.txt file')
    parser.add_argument('--output_path', type=str,
                    help='path to the output file with results', default="grf_results.csv")
    args = parser.parse_args()
    
    main(args)


