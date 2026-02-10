import pyterrier as pt
import pandas as pd
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import json
import numpy as np
import argparse
import re
import sys
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from ir_measures import calc_aggregate, nDCG, R, P, RR, Success

# ==========================================
# 1. SYSTEM & HARDWARE SETUP
# ==========================================
# High RAM allocation for PyTerrier MetaIndex
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
    # We look for the dense fusion runs in the dense-retrieval-stage sibling directory
    "dense_runs_dir": (SCRIPT_DIR / "../runs/be-rerank-stage-2").resolve(),
    "rewritten_dir": (SCRIPT_DIR / "../rewritten-queries").resolve(),
    "original_dir": (SCRIPT_DIR / "../original_queries").resolve(),
    "qrels_dir": (SCRIPT_DIR / "../qrel").resolve(),
    
    # Outputs
    "run_output_dir": (SCRIPT_DIR / "runs/ce-rerank-stage-3").resolve(),
    "eval_output_dir": (SCRIPT_DIR / "evaluations/ce-rerank-stage-3").resolve()
}

# QREL Mappings
QRELS_FILES = {
    "train": PATHS["qrels_dir"] / "train-2025-qrel.txt",
    "dev1": PATHS["qrels_dir"] / "dev1-2025-qrel.txt",
    "dev2": PATHS["qrels_dir"] / "dev2-2025-qrel.txt",
    "dev3": PATHS["qrels_dir"] / "dev3-2025-qrel.txt",
    "test": PATHS["qrels_dir"] / "test-2025-qrel.txt"
}

# Dataset Key Normalization
DATASET_MAP = {
    "train": "train",
    "dev-1": "dev1", "dev1": "dev1",
    "dev-2": "dev2", "dev2": "dev2", 
    "dev-3": "dev3", "dev3": "dev3",
    "test": "test"
}

# Model Config
MODEL_KEY = "monot5-large-msmarco-10k"
MODEL_CONFIG = {
    "hf_id": "castorini/monot5-large-msmarco-10k",
    "local_path": os.path.join(LOCAL_MODEL_DIR, "monot5-large-msmarco"),
    "batch_size": 32 # T5 Large is VRAM heavy. 32 per GPU.
}

# Tuning Settings
K_GRID = [50, 100, 150, 200, 250, 300]
MAX_K_INFERENCE = 300
TUNING_DATASETS = ["train", "dev1", "dev2"]
TEST_DATASET = "dev3"
DATASET_SEQUENCE = TUNING_DATASETS + [TEST_DATASET]
ALPHA = 0.5 

# Metrics
MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10]

# ==========================================
# 3. HELPER CLASSES & FUNCTIONS
# ==========================================

class MonoT5Dataset(Dataset):
    def __init__(self, queries, docs, tokenizer):
        self.queries = queries
        self.docs = docs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # MonoT5 Format: "Query: {q} Document: {d} Relevant:"
        text = f"Query: {self.queries[idx]} Document: {self.docs[idx]} Relevant:"
        return text

class MonoT5Scorer:
    def __init__(self, config):
        self.batch_size = config['batch_size'] * max(1, NUM_GPUS)
        
        path = config['local_path'] if os.path.exists(config['local_path']) else config['hf_id']
        print(f"Loading MonoT5 from {path}...")
        
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        
        # Token IDs for "true" and "false"
        self.true_token = self.tokenizer.encode("true")[0]
        self.false_token = self.tokenizer.encode("false")[0]
        
        if NUM_GPUS > 1:
            self.model = nn.DataParallel(self.model)
        
        self.model.to(DEVICE)
        self.model.eval()
        
        if DEVICE == "cuda":
            self.model.half() # FP16 for speed/memory

    def score(self, query_texts, doc_texts):
        dataset = MonoT5Dataset(query_texts, doc_texts, self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_scores = []

        with torch.no_grad():
            for batch_text in tqdm(loader, desc="MonoT5 Re-ranking", leave=False):
                inputs = self.tokenizer(
                    list(batch_text), 
                    padding=True, truncation=True, max_length=512, return_tensors='pt'
                ).to(DEVICE)

                # Decoder start token
                decoder_input_ids = torch.full(
                    (inputs.input_ids.size(0), 1),
                    self.model.module.config.decoder_start_token_id if hasattr(self.model, "module") else self.model.config.decoder_start_token_id,
                    device=DEVICE
                )

                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False
                )
                
                # Logits at first token
                logits = outputs.logits[:, 0, :]
                true_logits = logits[:, self.true_token]
                false_logits = logits[:, self.false_token]
                
                # Probability of "true"
                batch_scores = torch.exp(true_logits) / (torch.exp(true_logits) + torch.exp(false_logits))
                all_scores.append(batch_scores.float().cpu().numpy())

        return np.concatenate(all_scores)

def scan_query_variants():
    """Scans for available query variants."""
    variants = {}
    
    # Scan Rewritten
    if PATHS["rewritten_dir"].exists():
        for f in PATHS["rewritten_dir"].glob("*.jsonl"):
            name = f.name
            # Rewritten
            match_rw = re.match(r"^(?P<model>.*?)_(?P<dataset>.*?)_rewritten_queries\.jsonl$", name)
            if match_rw:
                model, raw_ds = match_rw.group("model", "dataset")
                dataset = DATASET_MAP.get(raw_ds)
                if dataset:
                    key = f"rewritten-{model}"
                    variants.setdefault(key, {})[dataset] = f
                continue
            
            # Summarized
            match_sum = re.match(r"^(?P<dataset>.*?)_summarized-(?P<model>.*?)\.jsonl$", name)
            if match_sum:
                model, raw_ds = match_sum.group("model", "dataset")
                dataset = DATASET_MAP.get(raw_ds)
                if dataset:
                    key = f"summarized-{model}"
                    variants.setdefault(key, {})[dataset] = f
                continue

    # Scan Original
    if PATHS["original_dir"].exists():
        for f in PATHS["original_dir"].glob("*-original.jsonl"):
            name = f.name
            match_orig = re.match(r"^(?P<dataset>.*?)-original\.jsonl$", name)
            if match_orig:
                raw_ds = match_orig.group("dataset")
                dataset = DATASET_MAP.get(raw_ds)
                if dataset:
                    variants.setdefault("original", {})[dataset] = f

    return variants

def get_user_variant_selection(variants):
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
                return sorted_keys[idx], variants[sorted_keys[idx]]
        except ValueError: pass

def _read_query_file(path):
    df = pd.read_json(path, lines=True)
    if 'query_id' in df.columns: df = df.rename(columns={'query_id': 'qid'})
    if 'text' in df.columns: df = df.rename(columns={'text': 'query'})
    df['qid'] = df['qid'].astype(str)
    return df[['qid', 'query']]

def find_input_dense_run(variant_name, dataset):
    """
    Locates the dense fusion run file from the previous stage.
    Looking for: {dataset}_dense_fusion_{variant_name}.run
    """
    if not PATHS["dense_runs_dir"].exists():
        raise FileNotFoundError(f"Dense runs directory not found: {PATHS['dense_runs_dir']}")
        
    # Exact match for fusion file pattern
    expected_name = f"{dataset}_dense_fusion_{variant_name}.run"
    candidate = PATHS["dense_runs_dir"] / expected_name
    
    if candidate.exists():
        return candidate
            
    raise FileNotFoundError(f"No dense fusion run found: {expected_name}")

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Cross-Encoder Re-Ranking Stage 2")
    parser.add_argument("--variant", type=str, required=False, 
                        help="Query variant name. If omitted, interactive menu is shown.")
    args = parser.parse_args()

    # Directories
    os.makedirs(PATHS["run_output_dir"], exist_ok=True)
    os.makedirs(PATHS["eval_output_dir"], exist_ok=True)

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
        print("CRITICAL: No query files found.")
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

    # Validate Sequence
    active_sequence = [ds for ds in DATASET_SEQUENCE if ds in selected_variant_files]
    if not active_sequence:
        print("Error: No valid datasets found for this variant.")
        return

    # -----------------------------------------------
    # PHASE 1: RE-RANKING
    # -----------------------------------------------
    print(f"\n{'#'*60}")
    print(f"CROSS-ENCODER | {selected_variant_name} | {MODEL_KEY}")
    print(f"{'#'*60}")

    scorer = MonoT5Scorer(MODEL_CONFIG)
    tuning_matrix = {k: [] for k in K_GRID}
    master_dfs = {}
    
    for dataset in active_sequence:
        print(f"\n>>> Processing Dataset: {dataset}")

        # 1. Locate Input Run
        try:
            run_path = find_input_dense_run(selected_variant_name, dataset)
            print(f"  Input Baseline: {run_path.name}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # 2. Load Queries
        queries_df = _read_query_file(selected_variant_files[dataset])

        # 3. Merge
        dense_run = pt.io.read_results(str(run_path))
        dense_run['qid'] = dense_run['qid'].astype(str)
        dense_run = dense_run.merge(queries_df, on='qid', how='left')

        # 4. Top-K Filter
        dense_run = dense_run.sort_values(["qid", "score"], ascending=[True, False])
        dense_run["rank"] = dense_run.groupby("qid").cumcount() + 1
        candidates = dense_run[dense_run["rank"] <= MAX_K_INFERENCE].copy()

        # 5. Fetch Text
        print("  Fetching text...")
        text_pipeline = pt.text.get_text(index, "text")
        candidates_with_text = text_pipeline.transform(candidates)

        # 6. Score
        print(f"  Re-ranking Top-{MAX_K_INFERENCE}...")
        ce_scores = scorer.score(
            candidates_with_text["query"].tolist(), 
            candidates_with_text["text"].tolist()
        )

        # 7. Master DataFrame
        master_ce = candidates_with_text.copy()
        master_ce["score"] = ce_scores
        master_ce = master_ce.rename(columns={"rank": "dense_rank"})
        master_dfs[dataset] = master_ce

        # 8. Virtual Tuning
        if dataset in TUNING_DATASETS:
            qrel_path = QRELS_FILES.get(dataset)
            if qrel_path and qrel_path.exists():
                qrels = pt.io.read_qrels(str(qrel_path))
                q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
                
                print("  Virtual Sweep (K Tuning)...")
                for k in K_GRID:
                    subset = master_ce[master_ce["dense_rank"] <= k].copy()
                    subset = subset.sort_values(["qid", "score"], ascending=[True, False])
                    r_std = subset.rename(columns={"qid":"query_id", "docno":"doc_id"})
                    
                    m = calc_aggregate([nDCG@10], q_std, r_std)
                    tuning_matrix[k].append(m.get(nDCG@10, 0))

    # -----------------------------------------------
    # PHASE 2: ROBUST K & FINAL RUNS
    # -----------------------------------------------
    print(f"\n>>> Robust K Selection (Target: nDCG@10)")
    best_k = 50
    best_score = -1.0
    selection_log = []
    
    for k in K_GRID:
        scores = tuning_matrix[k]
        if not scores: continue
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        robust_s = mean_s - (ALPHA * std_s)
        selection_log.append({"k": k, "mean": mean_s, "std": std_s, "score": robust_s})
        
        if robust_s > best_score:
            best_score = robust_s
            best_k = k
            
    print(f"    Winner: K={best_k} (Robust Score: {best_score:.4f})")
    
    # Save Tuning Log
    pd.DataFrame(selection_log).to_csv(PATHS["eval_output_dir"] / "k_tuning_robustness.csv", index=False)

    print(f"\n>>> Final Generation & Analysis")
    final_results_log = []
    
    for dataset, master_ce in master_dfs.items():
        # Apply Best K
        final_subset = master_ce[master_ce["dense_rank"] <= best_k].copy()
        final_subset = final_subset.sort_values(["qid", "score"], ascending=[True, False])
        final_subset["rank"] = final_subset.groupby("qid").cumcount() + 1
        final_subset["Q0"] = "Q0"
        final_subset["system"] = f"monoT5_{selected_variant_name}"
        
        # Save
        fname = f"{dataset}_monoT5_{selected_variant_name}.run"
        fpath = PATHS["run_output_dir"] / fname
        pt.io.write_results(final_subset, str(fpath))
        print(f"  Saved: {fname}")
        
        # Calculate Gain
        qrel_path = QRELS_FILES.get(dataset)
        if qrel_path and qrel_path.exists():
            qrels = pt.io.read_qrels(str(qrel_path))
            q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
            r_std = final_subset.rename(columns={"qid":"query_id", "docno":"doc_id"})
            
            # CE Metrics
            metrics = calc_aggregate(MEASURES, q_std, r_std)
            ce_score = metrics.get(nDCG@10, 0)
            
            # Baseline (Dense Fusion) Metrics
            try:
                base_path = find_input_dense_run(selected_variant_name, dataset)
                base_run = pt.io.read_results(str(base_path))
                base_r_std = base_run.rename(columns={"qid":"query_id", "docno":"doc_id"})
                base_m = calc_aggregate([nDCG@10], q_std, base_r_std)
                base_score = base_m.get(nDCG@10, 0)
            except:
                base_score = 0.0

            gain = ((ce_score - base_score) / base_score * 100) if base_score > 0 else 0
            
            record = {
                "dataset": dataset,
                "Baseline_nDCG": base_score,
                "CE_nDCG": ce_score,
                "Gain_Pct": gain
            }
            # Add all metrics
            for m in MEASURES:
                record[str(m)] = metrics.get(m, 0)
                
            final_results_log.append(record)
            print(f"  {dataset}: nDCG@10 {ce_score:.4f} (Gain: {gain:+.2f}%)")

    # Final Output
    if final_results_log:
        df_res = pd.DataFrame(final_results_log)
        csv_path = PATHS["eval_output_dir"] / f"ce_gains_{selected_variant_name}.csv"
        df_res.to_csv(csv_path, index=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_res, x="dataset", y="Gain_Pct", palette="RdYlGn")
        plt.title(f"MonoT5 Gain over Dense Fusion ({selected_variant_name})")
        plt.ylabel("Improvement in nDCG@10 (%)")
        plt.axhline(0, color='black')
        plt.savefig(PATHS["eval_output_dir"] / f"ce_gain_chart_{selected_variant_name}.png")

    print("\nCross-Encoder Stage Complete.")

if __name__ == "__main__":
    main()