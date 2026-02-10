import pyterrier as pt
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
import json
import numpy as np
import argparse
import re
import sys
import gc
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from ir_measures import calc_aggregate, nDCG, R, P, RR, Success

# ==========================================
# 1. SYSTEM & HARDWARE SETUP
# ==========================================
# High RAM allocation for PyTerrier MetaIndex (storing text)
os.environ["JAVA_OPTS"] = "-Xmx100g -Xms16g -Dindex.meta.data-source=fileinmem"

if not pt.java.started():
    pt.java.init()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()

# ==========================================
# 2. PATHS & CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Absolute Paths (Large Assets)
INDEX_PATH = "/media/12TB/shared/datasets/indices/trec-tot-2025/trec-tot-2025-pyterrier-index"
LOCAL_MODEL_DIR = "/media/12TB/shared/models"

# Relative Paths
PATHS = {
    # Inputs
    "sparse_runs_dir": (SCRIPT_DIR / "../runs/sparse-retrieval-stage-1").resolve(),
    "rewritten_dir": (SCRIPT_DIR / "../rewritten-queries").resolve(),
    "original_dir": (SCRIPT_DIR / "../original_queries").resolve(),
    "qrels_dir": (SCRIPT_DIR / "../qrel").resolve(),
    
    # Outputs
    "run_output_dir": (SCRIPT_DIR / "runs/be-rerank-stage-2").resolve(),
    "eval_output_dir": (SCRIPT_DIR / "evaluations/be-rerank-stage-2").resolve(),
    "backup_5k_dir": (SCRIPT_DIR / "runs/be-rerank-stage-2/full_5k").resolve()
}

# QREL Mappings
QRELS_FILES = {
    "train": PATHS["qrels_dir"] / "train-2025-qrel.txt",
    "dev1": PATHS["qrels_dir"] / "dev1-2025-qrel.txt",
    "dev2": PATHS["qrels_dir"] / "dev2-2025-qrel.txt",
    "dev3": PATHS["qrels_dir"] / "dev3-2025-qrel.txt",
    "test": PATHS["qrels_dir"] / "test-2025-qrel.txt"
}

# Dataset Key Normalization (Filename -> Internal Key)
DATASET_MAP = {
    "train": "train",
    "dev-1": "dev1",
    "dev-2": "dev2", 
    "dev-3": "dev3",
    "test": "test",
    "dev1": "dev1", # Handle direct matches too
    "dev2": "dev2",
    "dev3": "dev3"
}

# Models Configuration
MODELS_CONFIG = {
    "colbertv2.0": {
        "hf_id": "colbert-ir/colbertv2.0",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "colbertv2.0"),
        "type": "colbert",
        "batch_size": 128
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
        "type": "e5", 
        "batch_size": 128
    }
}

# Tuning Settings
K_GRID = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
MAX_K_INFERENCE = 5000
TUNING_DATASETS = ["train", "dev1", "dev2"]
TEST_DATASET = "dev3" # or test
DATASET_SEQUENCE = TUNING_DATASETS + [TEST_DATASET]
ALPHA = 0.5 # Robustness parameter

# Metrics
MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10]

# ==========================================
# 3. HELPER CLASSES & FUNCTIONS
# ==========================================

class ReRankDataset(Dataset):
    def __init__(self, queries, docs, tokenizer, model_type="dot_product"):
        self.queries = queries
        self.docs = docs
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q_text = str(self.queries[idx])
        d_text = str(self.docs[idx])

        # E5 Specific Prefixing
        if self.model_type == "e5":
            q_text = f"query: {q_text}"
            d_text = f"passage: {d_text}"
        
        return q_text, d_text

class BiEncoderScorer:
    def __init__(self, model_key, config):
        self.model_name = model_key
        self.type = config['type']
        self.batch_size = config['batch_size'] * max(1, NUM_GPUS)
        
        # Resolve Path (Local vs HF)
        model_path = config['local_path'] if os.path.exists(config['local_path']) else config['hf_id']
        print(f"Loading {self.model_name} from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
        if NUM_GPUS > 1:
            self.model = nn.DataParallel(self.model)
        
        self.model.to(DEVICE)
        self.model.eval()

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def score(self, query_texts, doc_texts):
        dataset = ReRankDataset(query_texts, doc_texts, self.tokenizer, model_type=self.type)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_scores = []

        with torch.no_grad():
            for q_batch, d_batch in tqdm(loader, desc=f"Scoring {self.model_name}", leave=False):
                # Tokenize
                q_inputs = self.tokenizer(list(q_batch), padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
                d_inputs = self.tokenizer(list(d_batch), padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)

                if self.type == "colbert":
                    # Custom ColBERT MaxSim implementation
                    q_out = self.model(**q_inputs)[0] 
                    d_out = self.model(**d_inputs)[0]
                    
                    q_out = torch.nn.functional.normalize(q_out, p=2, dim=2)
                    d_out = torch.nn.functional.normalize(d_out, p=2, dim=2)

                    sim_matrix = torch.bmm(q_out, d_out.transpose(1, 2))
                    max_sim_values, _ = torch.max(sim_matrix, dim=2) 
                    scores = torch.sum(max_sim_values, dim=1) 
                    
                else:
                    # Standard Dot Product (Contriever, E5, etc.)
                    q_out = self.model(**q_inputs)[0]
                    d_out = self.model(**d_inputs)[0]
                    
                    q_emb = self.mean_pooling(q_out, q_inputs['attention_mask'])
                    d_emb = self.mean_pooling(d_out, d_inputs['attention_mask'])
                    
                    q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1)
                    d_emb = torch.nn.functional.normalize(d_emb, p=2, dim=1)
                    
                    scores = torch.sum(q_emb * d_emb, dim=1)

                all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores)

def scan_query_variants():
    """
    Scans rewritten-queries and original-queries directories to build a catalog of available variants.
    Returns:
        dict: { "variant_name": { "train": Path(...), "dev1": Path(...) } }
    """
    variants = {}

    # 1. Scan Rewritten Directory
    # Patterns:
    # Rewritten:  {model}_{dataset}_rewritten_queries.jsonl  -> llama_dev-1_rewritten_queries.jsonl
    # Summarized: {dataset}_summarized-{model}.jsonl         -> dev-1_summarized-llama.jsonl
    
    if PATHS["rewritten_dir"].exists():
        for f in PATHS["rewritten_dir"].glob("*.jsonl"):
            name = f.name
            
            # Pattern A: Rewritten
            match_rw = re.match(r"^(?P<model>.*?)_(?P<dataset>.*?)_rewritten_queries\.jsonl$", name)
            if match_rw:
                model = match_rw.group("model")
                raw_ds = match_rw.group("dataset")
                dataset = DATASET_MAP.get(raw_ds)
                
                if dataset:
                    key = f"rewritten-{model}"
                    if key not in variants: variants[key] = {}
                    variants[key][dataset] = f
                continue

            # Pattern B: Summarized (Handling both 'summarized-llama' and 'summarized_llama' styles if they exist)
            match_sum = re.match(r"^(?P<dataset>.*?)_summarized-(?P<model>.*?)\.jsonl$", name)
            if match_sum:
                model = match_sum.group("model")
                raw_ds = match_sum.group("dataset")
                dataset = DATASET_MAP.get(raw_ds)
                
                if dataset:
                    key = f"summarized-{model}"
                    if key not in variants: variants[key] = {}
                    variants[key][dataset] = f
                continue

    # 2. Scan Original Directory
    # Pattern: {dataset}-original.jsonl
    if PATHS["original_dir"].exists():
        for f in PATHS["original_dir"].glob("*-original.jsonl"):
            name = f.name
            match_orig = re.match(r"^(?P<dataset>.*?)-original\.jsonl$", name)
            if match_orig:
                raw_ds = match_orig.group("dataset")
                dataset = DATASET_MAP.get(raw_ds)
                
                if dataset:
                    key = "original"
                    if key not in variants: variants[key] = {}
                    variants[key][dataset] = f

    return variants

def get_user_variant_selection(variants):
    """Interactively ask user to select a variant."""
    print(f"\n{'='*40}")
    print("AVAILABLE QUERY VARIANTS")
    print(f"{'='*40}")
    
    sorted_keys = sorted(variants.keys())
    
    for idx, key in enumerate(sorted_keys, 1):
        files_found = len(variants[key])
        datasets = ", ".join(sorted(variants[key].keys()))
        print(f"{idx}. {key:<25} ({files_found} files: {datasets})")
    
    while True:
        try:
            choice = input("\nSelect variant number to optimize: ")
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_keys):
                selected_key = sorted_keys[idx]
                return selected_key, variants[selected_key]
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a number.")

def _read_query_file(path):
    df = pd.read_json(path, lines=True)
    if 'query_id' in df.columns: df = df.rename(columns={'query_id': 'qid'})
    if 'text' in df.columns: df = df.rename(columns={'text': 'query'})
    df['qid'] = df['qid'].astype(str)
    return df[['qid', 'query']]

def find_input_sparse_run(variant_name, dataset):
    """
    Locates the best sparse run file from the previous stage.
    Looking for: {dataset}_BEST_bm25_..._{variant_name}.run
    """
    if not PATHS["sparse_runs_dir"].exists():
        raise FileNotFoundError(f"Sparse runs directory not found: {PATHS['sparse_runs_dir']}")
        
    # Regex to match: train_BEST_bm25_k1-1.20_b-0.75_rewritten-llama.run
    pattern = re.compile(rf"^{dataset}_BEST_bm25_.*_{re.escape(variant_name)}\.run$")
    
    candidates = []
    for f in PATHS["sparse_runs_dir"].glob("*.run"):
        if pattern.match(f.name):
            candidates.append(f)
            
    if not candidates:
        raise FileNotFoundError(f"No sparse BM25 run found for {dataset} with variant {variant_name}")
        
    # If multiple (unlikely with BEST naming), pick the first one
    return candidates[0]

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
    fused_df["system"] = "RRF_Dense_Fusion"
    return fused_df

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Dense Bi-Encoder Re-Ranking Stage")
    parser.add_argument("--variant", type=str, required=False, 
                        help="Query variant name (e.g., rewritten-llama, original). If omitted, interactive menu is shown.")
    parser.add_argument("--models", nargs="*", default=list(MODELS_CONFIG.keys()),
                        help="Specific models to run (default: all)")
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(PATHS["run_output_dir"], exist_ok=True)
    os.makedirs(PATHS["eval_output_dir"], exist_ok=True)
    os.makedirs(PATHS["backup_5k_dir"], exist_ok=True)

    print(f"Loading Index from: {INDEX_PATH}")
    if not os.path.exists(INDEX_PATH):
        print("CRITICAL: Index not found.")
        return
    index = pt.IndexFactory.of(INDEX_PATH)

    # -----------------------------------------------
    # PHASE 0: SELECT VARIANT
    # -----------------------------------------------
    variants = scan_query_variants()
    if not variants:
        print("CRITICAL: No query files found in ../rewritten-queries or ../original_queries")
        return

    selected_variant_name = ""
    selected_variant_files = {}

    if args.variant:
        if args.variant in variants:
            selected_variant_name = args.variant
            selected_variant_files = variants[args.variant]
        else:
            print(f"Error: Specified variant '{args.variant}' not found.")
            return
    else:
        selected_variant_name, selected_variant_files = get_user_variant_selection(variants)

    # Validate Datasets
    active_sequence = []
    for ds in DATASET_SEQUENCE:
        if ds in selected_variant_files:
            active_sequence.append(ds)
        else:
            print(f"Warning: {selected_variant_name} missing file for {ds}")
    
    if not active_sequence:
        print("Error: No valid datasets found for this variant.")
        return

    # -----------------------------------------------
    # PHASE 1: DENSE SCORING & TUNING
    # -----------------------------------------------
    final_dense_runs = {ds: {} for ds in active_sequence}
    
    # Validate Models
    selected_models = [m for m in args.models if m in MODELS_CONFIG]
    if not selected_models:
        print("No valid models selected.")
        return

    for model_name in selected_models:
        config = MODELS_CONFIG[model_name]
        print(f"\n{'#'*60}")
        print(f"MODEL: {model_name} | VARIANT: {selected_variant_name}")
        print(f"{'#'*60}")
        
        scorer = BiEncoderScorer(model_name, config)
        
        tuning_matrix = {k: [] for k in K_GRID}
        master_dfs = {}
        
        for dataset in active_sequence:
            print(f"\n>>> Processing Dataset: {dataset}")
            
            # 1. Locate Input Run
            try:
                run_path = find_input_sparse_run(selected_variant_name, dataset)
                print(f"  Input Run: {run_path.name}")
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            # 2. Load Queries (Using pre-scanned file path)
            try:
                query_path = selected_variant_files[dataset]
                queries_df = _read_query_file(query_path)
            except Exception as e:
                print(f"  [SKIP] Error reading query file: {e}")
                continue

            # 3. Load Run & Merge
            sparse_run = pt.io.read_results(str(run_path))
            sparse_run['qid'] = sparse_run['qid'].astype(str)
            sparse_run = sparse_run.merge(queries_df, on='qid', how='left')
            
            # 4. Filter Top-K Candidates
            sparse_run = sparse_run.sort_values(["qid", "score"], ascending=[True, False])
            sparse_run["rank"] = sparse_run.groupby("qid").cumcount() + 1
            candidates = sparse_run[sparse_run["rank"] <= MAX_K_INFERENCE].copy()
            
            if candidates.empty:
                print("  [WARN] No candidates found.")
                continue

            # 5. Fetch Text
            print("  Fetching text from index...")
            text_pipeline = pt.text.get_text(index, "text")
            candidates_with_text = text_pipeline.transform(candidates)
            
            # 6. Score
            print(f"  Re-ranking {len(candidates_with_text)} pairs...")
            dense_scores = scorer.score(
                candidates_with_text["query"].tolist(), 
                candidates_with_text["text"].tolist()
            )
            
            # 7. Create Master Dense DataFrame
            master_dense = candidates_with_text.copy()
            master_dense["score"] = dense_scores
            master_dense = master_dense.rename(columns={"rank": "sparse_rank"}) 
            
            # 8. Save Backup
            master_fname = f"{dataset}_{model_name}_{selected_variant_name}_FULL_5k.run"
            master_path = PATHS["backup_5k_dir"] / master_fname
            
            master_save = master_dense.sort_values(["qid", "score"], ascending=[True, False])
            master_save["rank"] = master_save.groupby("qid").cumcount() + 1
            master_save["Q0"] = "Q0"
            master_save["system"] = f"{model_name}_dense"
            pt.io.write_results(master_save, str(master_path))
            
            master_dfs[dataset] = master_dense

            # 9. Virtual Tuning (Tuning sets only)
            if dataset in TUNING_DATASETS:
                qrel_path = QRELS_FILES.get(dataset)
                if qrel_path and qrel_path.exists():
                    qrels = pt.io.read_qrels(str(qrel_path))
                    q_formatted = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
                    
                    print("  Virtual Tuning Sweep...")
                    for k in K_GRID:
                        # Keep docs that were in TOP K of SPARSE run
                        subset = master_dense[master_dense["sparse_rank"] <= k].copy()
                        subset = subset.sort_values(["qid", "score"], ascending=[True, False])
                        r_formatted = subset.rename(columns={"qid":"query_id", "docno":"doc_id"})
                        
                        m = calc_aggregate([R@1000], q_formatted, r_formatted)
                        tuning_matrix[k].append(m.get(R@1000, 0))

        # 10. Robust Selection
        print(f"\n>>> Robust Parameter Selection (K)")
        best_k = 1000 # Default
        best_score = -1.0
        
        for k in K_GRID:
            recalls = tuning_matrix[k]
            if not recalls: continue
            mean_r = np.mean(recalls)
            std_r = np.std(recalls)
            robust_score = mean_r - (ALPHA * std_r)
            
            if robust_score > best_score:
                best_score = robust_score
                best_k = k
                
        print(f"    Selected K={best_k} (Robust Score: {best_score:.4f})")
        
        # 11. Generate Final Files
        for dataset, master_df in master_dfs.items():
            final_subset = master_df[master_df["sparse_rank"] <= best_k].copy()
            final_subset = final_subset.sort_values(["qid", "score"], ascending=[True, False])
            final_subset["rank"] = final_subset.groupby("qid").cumcount() + 1
            final_subset["Q0"] = "Q0"
            final_subset["system"] = f"{model_name}_{selected_variant_name}"
            
            out_name = f"{dataset}_{model_name}_{selected_variant_name}.run"
            out_path = PATHS["run_output_dir"] / out_name
            pt.io.write_results(final_subset, str(out_path))
            
            final_dense_runs[dataset][model_name] = out_path
            print(f"  Final Run Saved: {out_name}")

        # Free Memory
        del scorer
        torch.cuda.empty_cache()
        gc.collect()

    # -----------------------------------------------
    # PHASE 2: FUSION
    # -----------------------------------------------
    print(f"\n{'#'*60}")
    print(f"RRF FUSION | {selected_variant_name}")
    print(f"{'#'*60}")
    
    fusion_log = []
    
    for dataset in active_sequence:
        runs_map = final_dense_runs[dataset]
        if not runs_map: continue
        
        print(f"Fusing {dataset} ({len(runs_map)} systems)...")
        
        dfs = [pt.io.read_results(str(p)) for p in runs_map.values()]
        fused_res = pandas_rrf(dfs, k=60)
        
        fused_name = f"{dataset}_dense_fusion_{selected_variant_name}.run"
        fused_path = PATHS["run_output_dir"] / fused_name
        pt.io.write_results(fused_res, str(fused_path))
        
        # Eval
        qrel_path = QRELS_FILES.get(dataset)
        if qrel_path and qrel_path.exists():
            qrels = pt.io.read_qrels(str(qrel_path))
            q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
            r_std = fused_res.rename(columns={"qid":"query_id", "docno":"doc_id"})
            
            metrics = calc_aggregate(MEASURES, q_std, r_std)
            fusion_log.append({
                "dataset": dataset,
                "nDCG@10": metrics.get(nDCG@10, 0),
                "R@1000": metrics.get(R@1000, 0)
            })
            print(f"  Fusion Score nDCG@10: {metrics.get(nDCG@10, 0):.4f}")

    # Save Log
    if fusion_log:
        log_path = PATHS["eval_output_dir"] / f"dense_fusion_results_{selected_variant_name}.csv"
        pd.DataFrame(fusion_log).to_csv(log_path, index=False)
        print(f"\nResults logged to {log_path}")

if __name__ == "__main__":
    main()