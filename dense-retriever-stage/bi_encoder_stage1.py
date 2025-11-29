import pyterrier as pt
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ir_measures
from ir_measures import calc_aggregate, nDCG, R, P, RR, Success
from torch.utils.data import DataLoader, Dataset
import gc

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
# High RAM allocation for PyTerrier MetaIndex
os.environ["JAVA_OPTS"] = "-Xmx100g -Xms16g -Dindex.meta.data-source=fileinmem"

if not pt.java.started():
    pt.java.init()

# GPU Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()
print(f"Hardware Detected: {NUM_GPUS} GPUs available.")

# ==========================================
# 2. CONFIGURATION
# ==========================================
# Local Model Directory
LOCAL_MODEL_DIR = "/media/12TB/shared/models"

MODELS_CONFIG = {
    "colbertv2.0": {
        "hf_id": "colbert-ir/colbertv2.0",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "colbertv2.0"),
        "type": "colbert",
        "batch_size": 128  # Per GPU
    },
    "contriever": {
        "hf_id": "facebook/contriever",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "contriever"),
        "type": "dot_product",
        "batch_size": 256
    },
    "e5-large": {
        "hf_id": "intfloat/e5-large",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "e5-large"),
        "type": "e5", # Requires specific prefixing
        "batch_size": 128
    }
}

K_GRID = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
MAX_K = 5000 # We infer on this many, then filter

# Datasets used for Robust Tuning
TUNING_DATASETS = ["train", "dev1", "dev2"]
TEST_DATASET = "dev3"
ALL_DATASETS = TUNING_DATASETS + [TEST_DATASET]

MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10]

# Robustness Parameter (Alpha)
# Score = Mean - Alpha * Std
ALPHA = 0.5 

# ==========================================
# 3. HELPERS & MODEL CLASSES
# ==========================================

def load_env(env_path="env.json"):
    if not os.path.exists(env_path):
        env_path = "../env.json" # Fallback
    with open(env_path, 'r') as f:
        return json.load(f)

def get_paths(env):
    return {
        "index": env['paths']['index_path'],
        "input": env['dense-retrieval']['dense_input_files'],
        "output_run": env['dense-retrieval']['dense_run_files'],
        "output_eval": env['dense-retrieval']['dense_eval_files'],
        "fusion_dir": env['paths']['fusion_run_directory']
    }

def load_queries(env, dataset, query_variant="mistral"):
    """Load queries for a dataset from query files."""
    # Map dataset names to file patterns
    dataset_map = {
        "train": "train",
        "dev1": "dev-1", 
        "dev2": "dev-2",
        "dev3": "dev-3",
        "test": "test"
    }
    
    dataset_file = dataset_map.get(dataset, dataset)
    query_path = f"./rewritten-queries/{query_variant}_{dataset_file}_rewritten_queries.jsonl"
    
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Query file not found: {query_path}")
    
    queries_df = pd.read_json(query_path, lines=True)
    
    # Normalize column names
    if 'query_id' in queries_df.columns:
        queries_df = queries_df.rename(columns={'query_id': 'qid'})
    if 'text' in queries_df.columns:
        queries_df = queries_df.rename(columns={'text': 'query'})
    
    # Ensure qid is string type for merging
    queries_df['qid'] = queries_df['qid'].astype(str)
    
    return queries_df[['qid', 'query']]

class ReRankDataset(Dataset):
    def __init__(self, queries, docs, tokenizer, max_len=512, model_type="dot_product"):
        self.queries = queries
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_type = model_type

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q_text = self.queries[idx]
        d_text = self.docs[idx]

        # E5 Specific Formatting
        if self.model_type == "e5":
            q_text = f"query: {q_text}"
            d_text = f"passage: {d_text}"
        
        return q_text, d_text

class BiEncoderScorer:
    def __init__(self, model_key, config):
        self.model_name = model_key
        self.config = config
        self.hf_id = config['hf_id']
        self.local_path = config.get('local_path', self.hf_id)
        self.type = config['type']
        self.batch_size = config['batch_size'] * max(1, NUM_GPUS)
        
        # Try local path first, fallback to HF hub
        model_path = self.local_path if os.path.exists(self.local_path) else self.hf_id
        print(f"Loading {self.model_name} from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
        if NUM_GPUS > 1:
            print(f"Wrapping model in DataParallel for {NUM_GPUS} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(DEVICE)
        self.model.eval()

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def score(self, query_texts, doc_texts):
        """
        Scores a list of (Query, Doc) pairs.
        Returns a numpy array of scores.
        """
        dataset = ReRankDataset(query_texts, doc_texts, self.tokenizer, model_type=self.type)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_scores = []

        with torch.no_grad():
            for q_batch, d_batch in tqdm(loader, desc=f"Scoring {self.model_name}", leave=False):
                # Tokenize
                q_inputs = self.tokenizer(list(q_batch), padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
                d_inputs = self.tokenizer(list(d_batch), padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)

                if self.type == "colbert":
                    # ColBERT Interaction (MaxSim)
                    q_out = self.model(**q_inputs)[0] 
                    d_out = self.model(**d_inputs)[0]
                    
                    q_out = torch.nn.functional.normalize(q_out, p=2, dim=2)
                    d_out = torch.nn.functional.normalize(d_out, p=2, dim=2)

                    sim_matrix = torch.bmm(q_out, d_out.transpose(1, 2))
                    max_sim_values, _ = torch.max(sim_matrix, dim=2) 
                    scores = torch.sum(max_sim_values, dim=1) 
                    
                else:
                    # Dot Product Models
                    q_out = self.model(**q_inputs)[0]
                    d_out = self.model(**d_inputs)[0]
                    
                    q_emb = self.mean_pooling(q_out, q_inputs['attention_mask'])
                    d_emb = self.mean_pooling(d_out, d_inputs['attention_mask'])
                    
                    q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1)
                    d_emb = torch.nn.functional.normalize(d_emb, p=2, dim=1)
                    
                    scores = torch.sum(q_emb * d_emb, dim=1)

                all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def pandas_rrf(df_list, k=60):
    fused_parts = []
    for df in df_list:
        df_sorted = df.sort_values(["qid", "score"], ascending=[True, False])
        df_sorted["rank"] = df_sorted.groupby("qid").cumcount() + 1
        df_sorted["rrf_score_part"] = 1.0 / (k + df_sorted["rank"])
        fused_parts.append(df_sorted[["qid", "docno", "rrf_score_part"]])
    combined = pd.concat(fused_parts)
    fused_df = combined.groupby(["qid", "docno"], as_index=False)["rrf_score_part"].sum()
    fused_df = fused_df.rename(columns={"rrf_score_part": "score"})
    fused_df = fused_df.sort_values(["qid", "score"], ascending=[True, False])
    fused_df["rank"] = fused_df.groupby("qid").cumcount() + 1
    fused_df["Q0"] = "Q0"
    fused_df["system"] = "RRF_fusion"
    return fused_df

def main():
    env = load_env()
    paths = get_paths(env)
    
    # Ensure dirs exist
    os.makedirs(paths['output_run'], exist_ok=True)
    bi_encoder_eval_dir = os.path.join(paths['output_eval'], "bi_encoder")
    os.makedirs(bi_encoder_eval_dir, exist_ok=True)
    
    # Directory for Master 5k Files (Backup)
    full_5k_dir = os.path.join(paths['output_run'], "full_5k")
    os.makedirs(full_5k_dir, exist_ok=True)
    
    print("Loading Index...")
    index = pt.IndexFactory.of(paths['index'])
    
    def get_qrels(ds):
        key_map = {
            "train": "train_qrels_path", "dev1": "dev1_qrels_path",
            "dev2": "dev2_qrels_path", "dev3": "dev3_qrels_path"
        }
        return pt.io.read_qrels(env['paths'][key_map[ds]])

    # -----------------------------------------------
    # PHASE 1: GENERATE DENSE SCORES & TUNE K
    # -----------------------------------------------
    
    final_dense_runs = {ds: {} for ds in ALL_DATASETS} 

    for model_name, config in MODELS_CONFIG.items():
        print(f"\n{'#'*60}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'#'*60}")
        
        scorer = BiEncoderScorer(model_name, config)
        
        # We store recall matrices to perform Robust K selection
        # Format: { k_value: [recall_train, recall_dev1, recall_dev2] }
        tuning_matrix = {k: [] for k in K_GRID}
        
        # Store Master Dataframes in memory to avoid re-loading for final generation
        master_dfs = {} 
        
        metrics_log = []

        # 1. Inference & Virtual Sweep on Tuning Sets
        for dataset in ALL_DATASETS:
            print(f"\n>>> Dataset: {dataset}")
            
            # Load Sparse Run
            run_path = os.path.join(paths['input'], f"{dataset}.run")
            if not os.path.exists(run_path):
                print(f"  [ERR] Run file not found: {run_path}")
                continue
                
            sparse_run = pt.io.read_results(run_path)
            
            # Load Queries
            print("  Loading queries...")
            queries_df = load_queries(env, dataset)
            
            # Merge queries with sparse run
            sparse_run['qid'] = sparse_run['qid'].astype(str)
            sparse_run = sparse_run.merge(queries_df, on='qid', how='left')
            
            # Filter to Top-5000
            sparse_run = sparse_run.sort_values(["qid", "score"], ascending=[True, False])
            sparse_run["rank"] = sparse_run.groupby("qid").cumcount() + 1
            candidates = sparse_run[sparse_run["rank"] <= MAX_K].copy()
            
            # Fetch Text
            print("  Fetching text...")
            text_pipeline = pt.text.get_text(index, "text")
            candidates_with_text = text_pipeline.transform(candidates)
            
            # Score
            print("  Re-ranking...")
            dense_scores = scorer.score(
                candidates_with_text["query"].tolist(), 
                candidates_with_text["text"].tolist()
            )
            
            # Create Master Dense DataFrame
            master_dense = candidates_with_text.copy()
            master_dense["score"] = dense_scores
            master_dense = master_dense.rename(columns={"rank": "sparse_rank"}) 
            
            # [CRITICAL] Save Full 5k List for Offline Analysis
            master_fname = f"{model_name}_{dataset}_FULL_5k.run"
            master_path = os.path.join(full_5k_dir, master_fname)
            
            master_save = master_dense.sort_values(["qid", "score"], ascending=[True, False])
            master_save["rank"] = master_save.groupby("qid").cumcount() + 1
            master_save["Q0"] = "Q0"
            master_save["system"] = f"{model_name}_dense_5k"
            
            pt.io.write_results(master_save, master_path)
            print(f"  Saved Backup 5k Run: {master_fname}")
            
            # Store in memory for later use (avoid re-read)
            master_dfs[dataset] = master_dense

            # Virtual K Tuning (Only on Tuning Sets)
            if dataset in TUNING_DATASETS:
                qrels = get_qrels(dataset)
                formatted_qrels = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
                
                print("  Virtual Sweep (Collecting Recall Stats)...")
                for k in K_GRID:
                    # Filter: Keep docs that were in the top K of the SPARSE list
                    subset = master_dense[master_dense["sparse_rank"] <= k].copy()
                    
                    # Evaluate Recall (R@1000 is usually equivalent to coverage for ToT)
                    # We use R@1000 as the proxy for "Recall@K" since we are re-ranking 
                    # whatever survived the K cut.
                    subset = subset.sort_values(["qid", "score"], ascending=[True, False])
                    formatted_res = subset.rename(columns={"qid":"query_id", "docno":"doc_id"})
                    
                    # Compute just Recall for speed in loop
                    # Using ir_measures
                    m = calc_aggregate([R@1000], formatted_qrels, formatted_res)
                    recall_val = m.get(R@1000, 0)
                    
                    # Store for aggregation
                    tuning_matrix[k].append(recall_val)
                    
                    metrics_log.append({
                        "model": model_name,
                        "dataset": dataset,
                        "k": k,
                        "R@1000": recall_val
                    })

        # 2. Robust K Selection
        print(f"\n>>> Robust K Selection ({model_name})")
        print(f"    Strategy: Maximize (Mean_Recall - {ALPHA} * Std_Dev)")
        
        best_k_robust = 1000
        best_robust_score = -1.0
        
        selection_log = []
        
        for k in K_GRID:
            recalls = tuning_matrix[k]
            if not recalls: continue
            
            mean_r = np.mean(recalls)
            std_r = np.std(recalls)
            robust_score = mean_r - (ALPHA * std_r)
            
            selection_log.append({
                "k": k, "mean": mean_r, "std": std_r, "score": robust_score
            })
            
            if robust_score > best_robust_score:
                best_robust_score = robust_score
                best_k_robust = k
                
        print(f"    Winner: K={best_k_robust} (Score: {best_robust_score:.4f})")
        
        # Save Selection Log
        pd.DataFrame(selection_log).to_csv(
            os.path.join(bi_encoder_eval_dir, f"{model_name}_robust_selection.csv"), index=False
        )

        # 3. Generate Final Run Files (Using Robust K)
        print(f"\n>>> Generating Final Run Files (K={best_k_robust})...")
        
        for dataset in ALL_DATASETS:
            # Retrieve from memory
            if dataset not in master_dfs: continue
            master_dense = master_dfs[dataset]
            
            # Apply Robust K Cut
            final_subset = master_dense[master_dense["sparse_rank"] <= best_k_robust].copy()
            final_subset = final_subset.sort_values(["qid", "score"], ascending=[True, False])
            final_subset["rank"] = final_subset.groupby("qid").cumcount() + 1
            final_subset["Q0"] = "Q0"
            final_subset["system"] = f"{model_name}_dense"
            
            # Save Run
            fname = f"{model_name}_{dataset}.run"
            fpath = os.path.join(paths['output_run'], fname)
            pt.io.write_results(final_subset, fpath)
            final_dense_runs[dataset][model_name] = fpath
            print(f"  {dataset}: Saved {fname}")
            
            # Full Evaluation for this final run
            qrels = get_qrels(dataset)
            q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
            r_std = final_subset.rename(columns={"qid":"query_id", "docno":"doc_id"})
            
            final_metrics = calc_aggregate(MEASURES, q_std, r_std)
            
            # Append to main log (optional, or just save separate summary)
            # We will rely on the fusion step summary for comparison

        # Cleanup
        del scorer
        torch.cuda.empty_cache()
        gc.collect()

    # -----------------------------------------------
    # PHASE 2: RRF FUSION
    # -----------------------------------------------
    print(f"\n{'#'*60}")
    print(f"DENSE FUSION (RRF)")
    print(f"{'#'*60}")
    
    fusion_log = []
    
    for dataset in ALL_DATASETS:
        runs_map = final_dense_runs[dataset]
        if len(runs_map) < 3:
            print(f"Skipping fusion for {dataset}, incomplete runs.")
            continue
            
        print(f"Fusing {dataset}...")
        
        dfs = []
        for m_name, path in runs_map.items():
            dfs.append(pt.io.read_results(path))
        
        fused_res = pandas_rrf(dfs, k=60)
        
        qrels = get_qrels(dataset)
        q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
        r_std = fused_res.rename(columns={"qid":"query_id", "docno":"doc_id"})
        
        metrics = calc_aggregate(MEASURES, q_std, r_std)
        
        fused_path = os.path.join(paths['output_run'], f"dense_fusion_{dataset}.run")
        pt.io.write_results(fused_res, fused_path)
        
        record = {
            "dataset": dataset,
            "system": "RRF_Dense_Fusion",
            "nDCG@10": metrics.get(nDCG@10, 0),
            "R@1000": metrics.get(R@1000, 0),
            "Success@10": metrics.get(Success@10, 0)
        }
        fusion_log.append(record)
        print(f"  Fusion Score nDCG@10: {record['nDCG@10']:.4f}")

    pd.DataFrame(fusion_log).to_csv(os.path.join(bi_encoder_eval_dir, "dense_fusion_results.csv"), index=False)
    print("\nDense Pipeline Complete.")

if __name__ == "__main__":
    main()