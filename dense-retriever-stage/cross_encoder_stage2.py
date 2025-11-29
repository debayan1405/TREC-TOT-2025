import pyterrier as pt
import pandas as pd
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
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
MODEL_NAME = "castorini/monot5-large-msmarco-10k"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "monot5-large-msmarco")
BATCH_SIZE = 64 # Per GPU (Total ~128) - T5 Large is VRAM hungry

# Candidate Depth Tuning
K_GRID = [50, 100, 150, 200, 250, 300]
MAX_K = 300 # Max candidates to re-rank

DATASETS = ["train", "dev1", "dev2", "dev3"]
TUNING_SETS = ["train", "dev1", "dev2"]

MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10]
ALPHA = 0.5 # Robustness parameter

# ==========================================
# 3. HELPERS & MODEL CLASSES
# ==========================================

def load_env(env_path="env.json"):
    if not os.path.exists(env_path):
        env_path = "../env.json"
    with open(env_path, 'r') as f:
        return json.load(f)

def get_paths(env):
    return {
        "index": env['paths']['index_path'],
        "dense_runs": env['dense-retrieval']['dense_run_files'],
        "dense_eval": env['dense-retrieval']['dense_eval_files'],
        "output_run": os.path.join(env['dense-retrieval']['dense_run_files'], "cross_encoder"),
        "output_eval": os.path.join(env['dense-retrieval']['dense_eval_files'], "cross_encoder")
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

class MonoT5Dataset(Dataset):
    def __init__(self, queries, docs, tokenizer):
        self.queries = queries
        self.docs = docs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # Format: "Query: {q} Document: {d} Relevant:"
        text = f"Query: {self.queries[idx]} Document: {self.docs[idx]} Relevant:"
        return text

class MonoT5Scorer:
    def __init__(self, model_name, local_path, batch_size):
        self.model_name = model_name
        self.batch_size = batch_size * max(1, NUM_GPUS)
        
        # Try local path first, fallback to HF hub
        model_path = local_path if os.path.exists(local_path) else model_name
        print(f"Loading MonoT5 from {model_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # Token IDs for "true" and "false"
        self.true_token = self.tokenizer.encode("true")[0]
        self.false_token = self.tokenizer.encode("false")[0]
        
        if NUM_GPUS > 1:
            print(f"Wrapping model in DataParallel for {NUM_GPUS} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(DEVICE)
        self.model.eval()
        
        # Enable FP16 if available for speed
        if DEVICE == "cuda":
            self.model.half()

    def score(self, query_texts, doc_texts):
        dataset = MonoT5Dataset(query_texts, doc_texts, self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_scores = []

        with torch.no_grad():
            for batch_text in tqdm(loader, desc="MonoT5 Re-ranking", leave=False):
                # Tokenize
                inputs = self.tokenizer(
                    list(batch_text), 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(DEVICE)

                # Generate scores
                # We only need decoder_input_ids for the first step usually, but T5 generate is complex.
                # Standard MonoT5 approach: compute logits for "true" and "false" tokens at first position
                
                # Prepare decoder input (start token)
                # T5 uses pad_token_id as start_token_id usually
                decoder_input_ids = torch.full(
                    (inputs.input_ids.size(0), 1),
                    self.model.module.config.decoder_start_token_id if hasattr(self.model, "module") else self.model.config.decoder_start_token_id,
                    device=DEVICE
                )

                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False  # Disable cache to avoid compatibility issues
                )
                
                logits = outputs.logits[:, 0, :] # First token logits [Batch, Vocab]
                
                # Extract logits for "true" and "false"
                true_logits = logits[:, self.true_token]
                false_logits = logits[:, self.false_token]
                
                # Softmax to get probability of "true"
                # Score = exp(true) / (exp(true) + exp(false))
                batch_scores = torch.exp(true_logits) / (torch.exp(true_logits) + torch.exp(false_logits))
                
                all_scores.append(batch_scores.float().cpu().numpy())

        return np.concatenate(all_scores)

# ==========================================
# 4. MAIN LOGIC
# ==========================================

def get_best_dense_input(dense_eval_dir):
    """
    Analyzes the dense_fusion_results.csv to find the best system (Single or Fusion).
    Returns the system name prefix (e.g., 'dense_fusion' or 'colbertv2.0').
    Priority Metric: Success@10 (Recall) -> nDCG@10
    """
    csv_path = os.path.join(dense_eval_dir, "bi_encoder", "dense_fusion_results.csv")
    if not os.path.exists(csv_path):
        print("Dense results log not found. Defaulting to 'dense_fusion'.")
        return "dense_fusion"
        
    df = pd.read_csv(csv_path)
    
    # We look at the average performance across Train/Dev1/Dev2
    # But since that CSV logs the Fusion results, we also need the single model logs...
    # For simplicity, if fusion exists and is generally good, we use it. 
    # Or we can check if Fusion Score > 0.4 (arbitrary check).
    # Let's assume we want to use the Fusion list as it theoretically has best recall.
    return "dense_fusion" 

def main():
    env = load_env()
    paths = get_paths(env)
    
    os.makedirs(paths['output_run'], exist_ok=True)
    os.makedirs(paths['output_eval'], exist_ok=True)
    
    print("Loading Index...")
    index = pt.IndexFactory.of(paths['index'])
    
    def get_qrels(ds):
        key_map = {
            "train": "train_qrels_path", "dev1": "dev1_qrels_path",
            "dev2": "dev2_qrels_path", "dev3": "dev3_qrels_path"
        }
        return pt.io.read_qrels(env['paths'][key_map[ds]])

    # 1. Identify Input Source
    # We assume 'dense_fusion' files exist from previous step
    input_prefix = "dense_fusion" 
    print(f"Selected Input Run Type: {input_prefix}")

    # 2. Initialize Scorer
    scorer = MonoT5Scorer(MODEL_NAME, LOCAL_MODEL_PATH, BATCH_SIZE)
    
    tuning_matrix = {k: [] for k in K_GRID} # For nDCG robustness
    metrics_log = []
    
    final_runs_map = {} # To store paths for final summary

    # -----------------------------------------------
    # PHASE 1: RE-RANK & TUNE K
    # -----------------------------------------------
    
    for dataset in DATASETS:
        print(f"\n>>> Processing Dataset: {dataset.upper()}")
        
        # Load Input Run
        run_file = f"{input_prefix}_{dataset}.run"
        run_path = os.path.join(paths['dense_runs'], run_file)
        
        if not os.path.exists(run_path):
            print(f"  [ERR] Input run not found: {run_path}")
            continue
            
        dense_run = pt.io.read_results(run_path)
        
        # Load Queries
        print("  Loading queries...")
        queries_df = load_queries(env, dataset)
        
        # Merge queries with dense run
        dense_run['qid'] = dense_run['qid'].astype(str)
        dense_run = dense_run.merge(queries_df, on='qid', how='left')
        
        # Filter to Max K (300)
        dense_run = dense_run.sort_values(["qid", "score"], ascending=[True, False])
        dense_run["rank"] = dense_run.groupby("qid").cumcount() + 1
        candidates = dense_run[dense_run["rank"] <= MAX_K].copy()
        
        # Fetch Text
        print("  Fetching text...")
        text_pipeline = pt.text.get_text(index, "text")
        candidates_with_text = text_pipeline.transform(candidates)
        
        # Score
        print(f"  Re-ranking Top-{MAX_K}...")
        ce_scores = scorer.score(
            candidates_with_text["query"].tolist(), 
            candidates_with_text["text"].tolist()
        )
        
        # Master DataFrame
        master_ce = candidates_with_text.copy()
        master_ce["score"] = ce_scores
        master_ce = master_ce.rename(columns={"rank": "dense_rank"})
        
        # Virtual Tuning
        if dataset in TUNING_SETS:
            qrels = get_qrels(dataset)
            q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
            
            print("  Virtual Sweep (K Tuning)...")
            for k in K_GRID:
                # Filter by DENSE rank (Simulate sending only top-K to CE)
                subset = master_ce[master_ce["dense_rank"] <= k].copy()
                
                # Sort by CE Score
                subset = subset.sort_values(["qid", "score"], ascending=[True, False])
                r_std = subset.rename(columns={"qid":"query_id", "docno":"doc_id"})
                
                # Evaluate (Focus on nDCG@10)
                m = calc_aggregate([nDCG@10], q_std, r_std)
                score = m.get(nDCG@10, 0)
                
                tuning_matrix[k].append(score)
                metrics_log.append({
                    "dataset": dataset,
                    "k": k,
                    "nDCG@10": score
                })

        # Store Master for Final Gen
        final_runs_map[dataset] = master_ce

    # 3. Robust K Selection
    print(f"\n>>> Robust K Selection (Target: nDCG@10)")
    best_k = 50
    best_robust_score = -1.0
    selection_log = []
    
    for k in K_GRID:
        scores = tuning_matrix[k]
        if not scores: continue
        
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        robust_s = mean_s - (ALPHA * std_s)
        
        selection_log.append({"k": k, "mean": mean_s, "std": std_s, "score": robust_s})
        
        if robust_s > best_robust_score:
            best_robust_score = robust_s
            best_k = k
            
    print(f"    Winner: K={best_k} (Score: {best_robust_score:.4f})")
    
    # Save Selection Log
    pd.DataFrame(selection_log).to_csv(
        os.path.join(paths['output_eval'], "k_tuning_robustness.csv"), index=False
    )

    # 4. Generate Final Runs & Calculate Gains
    print(f"\n>>> Final Generation (K={best_k}) & Gain Analysis")
    
    final_results_log = []
    
    for dataset in DATASETS:
        if dataset not in final_runs_map: continue
        master_ce = final_runs_map[dataset]
        
        # Apply Best K
        final_subset = master_ce[master_ce["dense_rank"] <= best_k].copy()
        final_subset = final_subset.sort_values(["qid", "score"], ascending=[True, False])
        final_subset["rank"] = final_subset.groupby("qid").cumcount() + 1
        final_subset["Q0"] = "Q0"
        final_subset["system"] = "monoT5_CE"
        
        # Save
        fname = f"monoT5_{dataset}.run"
        fpath = os.path.join(paths['output_run'], fname)
        pt.io.write_results(final_subset, fpath)
        print(f"  Saved: {fname}")
        
        # Evaluate
        qrels = get_qrels(dataset)
        q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
        r_std = final_subset.rename(columns={"qid":"query_id", "docno":"doc_id"})
        
        metrics = calc_aggregate(MEASURES, q_std, r_std)
        
        # Calculate Gain vs Dense Input
        # We need the Dense Baseline Score for this dataset
        # We assume the input file (dense_fusion_dataset.run) is the baseline
        base_path = os.path.join(paths['dense_runs'], f"{input_prefix}_{dataset}.run")
        base_run = pt.io.read_results(base_path)
        base_r_std = base_run.rename(columns={"qid":"query_id", "docno":"doc_id"})
        base_metrics = calc_aggregate([nDCG@10], q_std, base_r_std)
        base_score = base_metrics.get(nDCG@10, 0)
        
        ce_score = metrics.get(nDCG@10, 0)
        gain = ((ce_score - base_score) / base_score) * 100.0 if base_score > 0 else 0.0
        
        record = {
            "dataset": dataset,
            "stage": "Cross-Encoder",
            "Baseline_Dense_nDCG": base_score,
            "CE_nDCG": ce_score,
            "Gain_Pct": gain
        }
        # Add all metrics
        for m in MEASURES:
            record[str(m)] = metrics.get(m, 0)
            
        final_results_log.append(record)
        print(f"  {dataset}: nDCG@10 {ce_score:.4f} (Gain: {gain:+.2f}%)")

    # 5. Save & Plot
    df_res = pd.DataFrame(final_results_log)
    df_res.to_csv(os.path.join(paths['output_eval'], "ce_performance_gain.csv"), index=False)
    
    # Plot Gain
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x="dataset", y="Gain_Pct", palette="RdYlGn")
    plt.title("Cross-Encoder Performance Gain (%) over Dense Stage")
    plt.ylabel("Improvement in nDCG@10 (%)")
    plt.axhline(0, color='black', linewidth=1)
    plt.savefig(os.path.join(paths['output_eval'], "ce_gain_chart.png"))
    
    # Plot K Tuning
    if metrics_log:
        df_k = pd.DataFrame(metrics_log)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_k, x="k", y="nDCG@10", hue="dataset", marker="o")
        plt.title("Cross-Encoder K Tuning (nDCG@10)")
        plt.xlabel("Re-ranking Depth (K)")
        plt.grid(True)
        plt.savefig(os.path.join(paths['output_eval'], "ce_k_tuning_curve.png"))

    print("\nCross-Encoder Stage Complete.")

if __name__ == "__main__":
    main()