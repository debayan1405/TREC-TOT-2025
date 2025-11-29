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
MODELS_CONFIG = {
    "colbertv2.0": {
        "hf_id": "colbert-ir/colbertv2.0",
        "type": "colbert",
        "batch_size": 128  # Per GPU
    },
    "contriever": {
        "hf_id": "facebook/contriever",
        "type": "dot_product",
        "batch_size": 256
    },
    "e5-large": {
        "hf_id": "intfloat/e5-large",
        "type": "e5", # Requires specific prefixing
        "batch_size": 128
    }
}

K_GRID = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
MAX_K = 5000 # We infer on this many, then filter

DATASETS = ["train", "dev1", "dev2", "dev3"]

MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10]

# ==========================================
# 3. HELPERS & MODEL CLASSES
# ==========================================

def load_env(env_path="env.json"):
    if not os.path.exists(env_path):
        env_path = "/content/env.json" # Fallback
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
        self.type = config['type']
        self.batch_size = config['batch_size'] * max(1, NUM_GPUS)
        
        print(f"Loading {self.model_name} ({self.hf_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self.model = AutoModel.from_pretrained(self.hf_id)
        
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
                # Tokenize on main process to avoid pickling overhead issues in DP
                # For optimal speed, tokenization should ideally be inside the loop or dataset
                # Here we do it per batch to keep memory low
                
                # Encode Queries
                q_inputs = self.tokenizer(list(q_batch), padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
                d_inputs = self.tokenizer(list(d_batch), padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)

                if self.type == "colbert":
                    # ColBERT Interaction (MaxSim)
                    # We manually handle ColBERT scoring to avoid dependency hell
                    # Q: [B, Lq, D], D: [B, Ld, D]
                    
                    q_out = self.model(**q_inputs)[0] # last_hidden_state
                    d_out = self.model(**d_inputs)[0]
                    
                    # Normalize (ColBERT uses L2 norm usually, or dot product if pre-normed)
                    # Standard ColBERT v2 checks:
                    q_out = torch.nn.functional.normalize(q_out, p=2, dim=2)
                    d_out = torch.nn.functional.normalize(d_out, p=2, dim=2)

                    # MaxSim: For every Q token, find max sim in D tokens, then sum
                    # Similarity matrix: [B, Lq, Ld]
                    sim_matrix = torch.bmm(q_out, d_out.transpose(1, 2))
                    max_sim_values, _ = torch.max(sim_matrix, dim=2) # [B, Lq]
                    scores = torch.sum(max_sim_values, dim=1) # [B]
                    
                else:
                    # Dot Product Models (Contriever, E5)
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

def get_text_mapping(index, docids):
    """
    Fetches text for a list of DocIDs using PyTerrier MetaIndex.
    """
    # Unique docs to save lookup time
    unique_docids = list(set(docids))
    # PyTerrier text retrieval
    # docids must be integers (internal IDs) or strings (external docnos)?
    # pt.text.get_text uses docnos usually if input is dataframe
    pass 
    # Logic moved to main loop to use pt.text.get_text transformer

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
    
    # Load Index
    print("Loading Index...")
    index = pt.IndexFactory.of(paths['index'])
    
    # Qrels lookup
    def get_qrels(ds):
        key_map = {
            "train": "train_qrels_path", "dev1": "dev1_qrels_path",
            "dev2": "dev2_qrels_path", "dev3": "dev3_qrels_path"
        }
        return pt.io.read_qrels(env['paths'][key_map[ds]])

    # -----------------------------------------------
    # PHASE 1: GENERATE DENSE SCORES (For all datasets & models)
    # -----------------------------------------------
    
    # Store paths to final best run files for Phase 2 (Fusion)
    final_dense_runs = {ds: {} for ds in DATASETS} # {train: {colbert: path, ...}}

    for model_name, config in MODELS_CONFIG.items():
        print(f"\n{'#'*60}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'#'*60}")
        
        scorer = BiEncoderScorer(model_name, config)
        
        metrics_log = []

        # Optimal K tracking
        best_k_global = 1000 # Default
        
        for dataset in DATASETS:
            print(f"\n>>> Dataset: {dataset}")
            
            # 1. Load Sparse Run (Standardized names expected: train.run, dev1.run...)
            run_path = os.path.join(paths['input'], f"{dataset}.run")
            if not os.path.exists(run_path):
                print(f"  [ERR] Run file not found: {run_path}")
                continue
                
            sparse_run = pt.io.read_results(run_path)
            
            # 2. Filter to Top-5000 (Max Candidate Set)
            # Efficient: sort and head
            # But pt results are usually not sorted by rank in dataframe? Safe to sort.
            sparse_run = sparse_run.sort_values(["qid", "score"], ascending=[True, False])
            sparse_run["rank"] = sparse_run.groupby("qid").cumcount() + 1
            candidates = sparse_run[sparse_run["rank"] <= MAX_K].copy()
            
            # 3. Fetch Text
            print("  Fetching text...")
            # Use PyTerrier transformer for efficiency
            text_pipeline = pt.text.get_text(index, "text")
            candidates_with_text = text_pipeline.transform(candidates)
            
            # 4. Score
            print("  Re-ranking...")
            dense_scores = scorer.score(
                candidates_with_text["query"].tolist(), 
                candidates_with_text["text"].tolist()
            )
            
            # Create Master Dense DataFrame
            master_dense = candidates_with_text.copy()
            master_dense["score"] = dense_scores
            # Preserve sparse rank for K-filtering logic
            master_dense = master_dense.rename(columns={"rank": "sparse_rank"}) 
            
            # 5. Virtual K Tuning
            qrels = get_qrels(dataset)
            formatted_qrels = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
            
            best_k_dataset = 500
            best_recall_dataset = -1.0
            
            # Only tune/log metrics if not dev3 (Dev3 is blind test)
            if dataset != "dev3":
                print("  Tuning K (Virtual Sweep)...")
                for k in K_GRID:
                    # Filter: Keep docs that were in the top K of the SPARSE list
                    # This simulates if we had only sent K docs to the re-ranker
                    subset = master_dense[master_dense["sparse_rank"] <= k].copy()
                    
                    # Now rank this subset by DENSE score
                    subset = subset.sort_values(["qid", "score"], ascending=[True, False])
                    subset["rank"] = subset.groupby("qid").cumcount() + 1
                    
                    # Evaluate
                    formatted_res = subset.rename(columns={"qid":"query_id", "docno":"doc_id"})
                    res_metrics = calc_aggregate(MEASURES, formatted_qrels, formatted_res)
                    
                    # Log
                    rec_val = res_metrics.get(R@1000, 0)
                    ndcg_val = res_metrics.get(nDCG@10, 0)
                    
                    metrics_log.append({
                        "model": model_name,
                        "dataset": dataset,
                        "k": k,
                        "R@1000": rec_val,
                        "nDCG@10": ndcg_val,
                        "P@10": res_metrics.get(P@10, 0),
                        "Success@10": res_metrics.get(Success@10, 0)
                    })
                    
                    if rec_val > best_recall_dataset:
                        best_recall_dataset = rec_val
                        best_k_dataset = k
                
                print(f"  Best K (Recall) for {dataset}: {best_k_dataset}")
                
                # Update Global Decision: For simplicity, we use Train's best K for downstream
                if dataset == "train":
                    best_k_global = best_k_dataset
            
            else:
                # For Dev3, we just use the determined Global Best K
                print(f"  Applying Train-Optimal K={best_k_global} to Dev3...")
            
            # 6. Generate FINAL Run File for this Dataset (Using Best K Global)
            # Note: For Dev1/Dev2, we also save the run using the Train-determined K 
            # to be consistent with a real pipeline flow.
            final_subset = master_dense[master_dense["sparse_rank"] <= best_k_global].copy()
            final_subset = final_subset.sort_values(["qid", "score"], ascending=[True, False])
            final_subset["rank"] = final_subset.groupby("qid").cumcount() + 1
            final_subset["Q0"] = "Q0"
            final_subset["system"] = f"{model_name}_dense"
            
            # Save Run
            fname = f"{model_name}_{dataset}.run"
            fpath = os.path.join(paths['output_run'], fname)
            pt.io.write_results(final_subset, fpath)
            final_dense_runs[dataset][model_name] = fpath
            print(f"  Saved Final Run: {fname}")

        # Save Metrics Log for this Model
        if metrics_log:
            df_log = pd.DataFrame(metrics_log)
            csv_name = f"{model_name}_tuning.csv"
            df_log.to_csv(os.path.join(bi_encoder_eval_dir, csv_name), index=False)
            
            # Plotting
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_log, x="k", y="R@1000", hue="dataset", marker="o")
            plt.title(f"{model_name}: Recall@1000 vs Candidate K")
            plt.grid(True)
            plt.savefig(os.path.join(bi_encoder_eval_dir, f"{model_name}_recall_curve.png"))
            plt.close()

        # Clear GPU Memory between models
        del scorer
        torch.cuda.empty_cache()
        gc.collect()

    # -----------------------------------------------
    # PHASE 2: RRF FUSION OF DENSE MODELS
    # -----------------------------------------------
    print(f"\n{'#'*60}")
    print(f"DENSE FUSION (RRF)")
    print(f"{'#'*60}")
    
    fusion_log = []
    
    for dataset in DATASETS:
        runs_map = final_dense_runs[dataset]
        if len(runs_map) < 3:
            print(f"Skipping fusion for {dataset}, incomplete runs.")
            continue
            
        print(f"Fusing {dataset}...")
        
        # Load the 3 dense runs
        dfs = []
        for m_name, path in runs_map.items():
            dfs.append(pt.io.read_results(path))
        
        # Fuse
        fused_res = pandas_rrf(dfs, k=60)
        
        # Evaluate Fusion
        qrels = get_qrels(dataset)
        formatted_qrels = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
        formatted_res = fused_res.rename(columns={"qid":"query_id", "docno":"doc_id"})
        
        metrics = calc_aggregate(MEASURES, formatted_qrels, formatted_res)
        
        # Save Fused Run
        fused_path = os.path.join(paths['output_run'], f"dense_fusion_{dataset}.run")
        pt.io.write_results(fused_res, fused_path)
        
        # Log comparison logic (vs Best Single Dense)
        # For brevity, we just log the Fusion Score here. 
        # A separate analysis script can compare vs single models if needed.
        record = {
            "dataset": dataset,
            "system": "RRF_Dense_Fusion",
            "nDCG@10": metrics.get(nDCG@10, 0),
            "R@1000": metrics.get(R@1000, 0),
            "Success@10": metrics.get(Success@10, 0)
        }
        fusion_log.append(record)
        print(f"  Fusion Score nDCG@10: {record['nDCG@10']:.4f}")

    # Save Fusion Log
    pd.DataFrame(fusion_log).to_csv(os.path.join(bi_encoder_eval_dir, "dense_fusion_results.csv"), index=False)
    print("\nDense Pipeline Complete.")

if __name__ == "__main__":
    main()