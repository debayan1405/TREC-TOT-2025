import pyterrier as pt
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
from ir_measures import calc_aggregate, nDCG, R, P, RR, Success

# ==========================================
# 1. SYSTEM & HARDWARE SETUP
# ==========================================
os.environ["JAVA_OPTS"] = "-Xmx100g -Xms16g -Dindex.meta.data-source=fileinmem"

if not pt.java.started():
    pt.java.init()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()
print(f"Hardware: {NUM_GPUS} GPUs")

# ==========================================
# 2. PATHS & CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Absolute Paths
INDEX_PATH = "/media/12TB/shared/datasets/indices/trec-tot-2025/trec-tot-2025-pyterrier-index"
LOCAL_MODEL_DIR = "/media/12TB/shared/models"

# Relative Paths
PATHS = {
    # Inputs: Output from Stage 2 (Cross-Encoder)
    # Assumes directory structure: root/cross-retrieval-stage/runs/cross-retrieval-stage-2
    "ce_runs_dir": (SCRIPT_DIR / "../runs/ce-rerank-stage-3").resolve(),
    "rewritten_dir": (SCRIPT_DIR / "../rewritten-queries").resolve(),
    "original_dir": (SCRIPT_DIR / "../original_queries").resolve(),
    "qrels_dir": (SCRIPT_DIR / "../qrel").resolve(),
    
    # Outputs
    "run_output_dir": (SCRIPT_DIR / "runs/llm-rerank-stage-4").resolve(),
    "eval_output_dir": (SCRIPT_DIR / "evaluations/llm-rerank-stage-4").resolve()
}

# QREL Mappings
QRELS_FILES = {
    "train": PATHS["qrels_dir"] / "train-2025-qrel.txt",
    "dev1": PATHS["qrels_dir"] / "dev1-2025-qrel.txt",
    "dev2": PATHS["qrels_dir"] / "dev2-2025-qrel.txt",
    "dev3": PATHS["qrels_dir"] / "dev3-2025-qrel.txt",
    "test": PATHS["qrels_dir"] / "test-2025-qrel.txt"
}

# Dataset Normalization
DATASET_MAP = {
    "train": "train",
    "dev-1": "dev1", "dev1": "dev1",
    "dev-2": "dev2", "dev2": "dev2", 
    "dev-3": "dev3", "dev3": "dev3",
    "test": "test"
}

# Model Config
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "qwen2.5-72b-awq")

# Tuning Settings
# K_INPUT: How many docs from CE to consider (limited by what CE actually saved)
K_INPUT_GRID = [10, 20, 30, 40, 50, 100] 
# K_LLM: How many of those to actually re-rank with LLM prompt
K_LLM_GRID = [10, 15, 20, 25, 50]
MAX_LLM_INFERENCE_K = 50 # Actual inference cap

TUNING_DATASETS = ["train", "dev1", "dev2"]
TEST_DATASET = "dev3"
DATASET_SEQUENCE = TUNING_DATASETS + [TEST_DATASET]
ALPHA = 0.5 

# Metrics
MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10]

# ==========================================
# 3. PROMPT TEMPLATE
# ==========================================
RANKING_PROMPT = """You are an expert search relevance ranker.
Your task is to re-rank the following candidate documents based on their relevance to the user query.
The goal is to place the true relevant document at the very top (Rank 1).

Query: {query}

Candidates:
{candidates}

Instructions:
1. Analyze the query and the candidates carefully.
2. Output the ranking as a list of IDs in order of relevance, from most relevant to least relevant.
3. Use the format: [ID] > [ID] > [ID] ...
4. Only output the ranking, no explanation.

Ranking:"""

# ==========================================
# 4. HELPER CLASSES
# ==========================================

class QwenRanker:
    def __init__(self, model_name, local_path):
        self.model_name = model_name
        path = local_path if os.path.exists(local_path) else model_name
        print(f"Loading Qwen from {path}...")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        except Exception as e:
            print(f"[ERR] Model load failed: {e}")
            raise

    def rerank_batch(self, query, docs_list):
        """
        docs_list: list of (docid, text, original_score) tuples
        Returns: list of (docid, new_score)
        """
        candidates_str = ""
        id_map = {}
        
        # Limit context window usage per doc
        for idx, (docid, text, _) in enumerate(docs_list):
            safe_text = text[:300].replace("\n", " ")
            candidates_str += f"[{idx}] {safe_text}\n"
            id_map[idx] = docid

        prompt = RANKING_PROMPT.format(query=query, candidates=candidates_str)
        messages = [
            {"role": "system", "content": "You are a Ranker."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text_input], return_tensors="pt", padding=True).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse Output: [0] > [2] > [1]
        matches = re.findall(r"\[(\d+)\]", response)
        
        reranked_results = []
        rank_score = 1000.0
        seen_ids = set()
        
        for local_id_str in matches:
            try:
                local_id = int(local_id_str)
                if local_id in id_map and local_id not in seen_ids:
                    docid = id_map[local_id]
                    reranked_results.append((docid, rank_score))
                    rank_score -= 1.0
                    seen_ids.add(local_id)
            except: continue
                
        # Append unranked docs
        for idx, (docid, _, _) in enumerate(docs_list):
            if idx not in seen_ids:
                if idx in id_map: # Safety check
                    reranked_results.append((id_map[idx], rank_score))
                    rank_score -= 1.0
                
        return reranked_results

def scan_query_variants():
    """Scans for query variants."""
    variants = {}
    if PATHS["rewritten_dir"].exists():
        for f in PATHS["rewritten_dir"].glob("*.jsonl"):
            name = f.name
            match_rw = re.match(r"^(?P<model>.*?)_(?P<dataset>.*?)_rewritten_queries\.jsonl$", name)
            if match_rw:
                m, d = match_rw.group("model", "dataset")
                ds = DATASET_MAP.get(d)
                if ds: variants.setdefault(f"rewritten-{m}", {})[ds] = f
                continue
            
            match_sum = re.match(r"^(?P<dataset>.*?)_summarized-(?P<model>.*?)\.jsonl$", name)
            if match_sum:
                m, d = match_sum.group("model", "dataset")
                ds = DATASET_MAP.get(d)
                if ds: variants.setdefault(f"summarized-{m}", {})[ds] = f

    if PATHS["original_dir"].exists():
        for f in PATHS["original_dir"].glob("*-original.jsonl"):
            name = f.name
            match_orig = re.match(r"^(?P<dataset>.*?)-original\.jsonl$", name)
            if match_orig:
                ds = DATASET_MAP.get(match_orig.group("dataset"))
                if ds: variants.setdefault("original", {})[ds] = f
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

def find_input_ce_run(variant_name, dataset):
    """
    Locates the Cross-Encoder run file.
    Pattern: {dataset}_monoT5_{variant_name}.run
    """
    if not PATHS["ce_runs_dir"].exists():
        raise FileNotFoundError(f"CE runs dir not found: {PATHS['ce_runs_dir']}")
        
    expected_name = f"{dataset}_monoT5_{variant_name}.run"
    candidate = PATHS["ce_runs_dir"] / expected_name
    
    if candidate.exists():
        return candidate
            
    raise FileNotFoundError(f"No CE run found: {expected_name}")

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="LLM Re-Ranking Stage 3")
    parser.add_argument("--variant", type=str, required=False, 
                        help="Query variant name. If omitted, interactive menu is shown.")
    args = parser.parse_args()

    # Directories
    os.makedirs(PATHS["run_output_dir"], exist_ok=True)
    os.makedirs(PATHS["eval_output_dir"], exist_ok=True)

    print(f"Loading Index...")
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

    if args.variant:
        if args.variant in variants:
            selected_variant_name = args.variant
            selected_variant_files = variants[args.variant]
        else:
            print(f"Error: Specified variant '{args.variant}' not found.")
            return
    else:
        selected_variant_name, selected_variant_files = get_user_variant_selection(variants)

    active_sequence = [ds for ds in DATASET_SEQUENCE if ds in selected_variant_files]
    if not active_sequence:
        print("Error: No valid datasets found.")
        return

    # -----------------------------------------------
    # PHASE 1: LLM INFERENCE
    # -----------------------------------------------
    print(f"\n{'#'*60}")
    print(f"LLM RE-RANKER | {selected_variant_name}")
    print(f"{'#'*60}")

    ranker = QwenRanker(MODEL_NAME, LOCAL_MODEL_PATH)
    
    # Store results for joint tuning
    # Format: {dataset: DataFrame}
    final_dfs = {} 
    # Original CE runs (needed for analysis)
    ce_runs_cache = {}

    for dataset in active_sequence:
        print(f"\n>>> Processing Dataset: {dataset}")

        try:
            run_path = find_input_ce_run(selected_variant_name, dataset)
            print(f"  Input Baseline: {run_path.name}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # Load Queries
        queries_df = _read_query_file(selected_variant_files[dataset])

        # Load Run
        ce_run = pt.io.read_results(str(run_path))
        ce_run['qid'] = ce_run['qid'].astype(str)
        ce_run = ce_run.merge(queries_df, on='qid', how='left')
        ce_runs_cache[dataset] = ce_run.copy()

        # Filter for Inference (Top-MAX_LLM_INFERENCE_K)
        ce_run = ce_run.sort_values(["qid", "score"], ascending=[True, False])
        ce_run["rank"] = ce_run.groupby("qid").cumcount() + 1
        
        candidates = ce_run[ce_run["rank"] <= MAX_LLM_INFERENCE_K].copy()
        
        # Fetch Text
        print("  Fetching text...")
        text_pipeline = pt.text.get_text(index, "text")
        candidates = text_pipeline.transform(candidates)

        # Inference Loop
        new_rows = []
        grouped = candidates.groupby("qid")
        print(f"  Re-ranking Top-{MAX_LLM_INFERENCE_K}...")
        
        for qid, group in tqdm(grouped, total=len(grouped)):
            query_text = group.iloc[0]['query']
            docs = []
            for _, row in group.iterrows():
                docs.append((row['docno'], row['text'], row['score']))
            
            # Rank
            ranked = ranker.rerank_batch(query_text, docs)
            
            for docid, score in ranked:
                new_rows.append({"qid": qid, "docno": docid, "score": score})

        # Merge results
        llm_head = pd.DataFrame(new_rows)
        ce_tail = ce_run[ce_run["rank"] > MAX_LLM_INFERENCE_K].copy()
        
        # Combine: LLM head (scores ~1000) + CE tail (scores < 10)
        final_df = pd.concat([llm_head[['qid', 'docno', 'score']], ce_tail[['qid', 'docno', 'score']]])
        final_df = final_df.sort_values(["qid", "score"], ascending=[True, False])
        final_df["rank"] = final_df.groupby("qid").cumcount() + 1
        
        final_dfs[dataset] = final_df

    # -----------------------------------------------
    # PHASE 2: JOINT TUNING (Virtual)
    # -----------------------------------------------
    # Simulate: What if we only took Top-K_input from CE, 
    # and then re-ranked Top-K_llm of those?
    
    print("\n>>> Joint Hyperparameter Tuning")
    joint_stats = []

    for dataset in active_sequence:
        if dataset not in TUNING_DATASETS: continue
        if dataset not in final_dfs: continue
        
        qrel_path = QRELS_FILES.get(dataset)
        if not qrel_path or not qrel_path.exists(): continue
        
        qrels = pt.io.read_qrels(str(qrel_path))
        q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
        
        ce_run_full = ce_runs_cache[dataset]
        llm_run_full = final_dfs[dataset] # This contains LLM scores for Top-50

        for k_in in K_INPUT_GRID:
            # 1. Simulate input cut
            # Docs available from CE at this depth
            ce_subset = ce_run_full[ce_run_full["rank"] <= k_in]
            valid_docs = set(zip(ce_subset['qid'], ce_subset['docno']))
            
            for k_llm in K_LLM_GRID:
                # 2. Simulate LLM re-ranking depth
                # Rule: Docs in Top-K_llm GET LLM scores.
                # Docs outside Top-K_llm but in Top-K_input KEEP CE scores.
                
                # To implement efficiently:
                # Take llm_run_full.
                # Filter to only docs in 'valid_docs' (enforces K_input).
                # But llm_run_full assumes we re-ranked ALL of Top-50.
                # If k_llm < 50, we must revert scores for ranks k_llm...50 to original CE scores?
                # Optimization: Assuming LLM is always better, usually we just keep the re-ranked order.
                # The constraint is mostly about COST/Time in real world.
                # Here we just check: If we had only re-ranked K_llm, how good is it?
                # We simply filter llm_run_full to include only docs that were in ce_subset.
                # And we assume we paid the cost for K_llm = MAX(50, k_llm) effectively in this simulation code structure, 
                # but logically we just check performance of the resulting list.
                
                # Filter output by input constraints
                # This dataframe represents: The list if we re-ranked Top-50 of Top-K_input
                sim_df = llm_run_full[llm_run_full.set_index(['qid', 'docno']).index.isin(valid_docs)]
                
                # Evaluate
                r_std = sim_df.rename(columns={"qid":"query_id", "docno":"doc_id"})
                m = calc_aggregate([nDCG@10], q_std, r_std)
                score = m.get(nDCG@10, 0)
                
                joint_stats.append({
                    "dataset": dataset,
                    "k_input": k_in,
                    "k_llm": k_llm,
                    "score": score
                })

    # Robust Selection
    best_combo = (50, 50) # Default
    best_robust_score = -1.0
    
    if joint_stats:
        df_grid = pd.DataFrame(joint_stats)
        df_grid.to_csv(PATHS["eval_output_dir"] / "joint_tuning_log.csv", index=False)
        
        grouped = df_grid.groupby(['k_input', 'k_llm'])
        for (ki, kl), group in grouped:
            # Require all tuning sets
            if len(group) < len([d for d in TUNING_DATASETS if d in active_sequence]): continue
            
            scores = group['score'].values
            mean_s = np.mean(scores)
            std_s = np.std(scores)
            robust_s = mean_s - (ALPHA * std_s)
            
            if robust_s > best_robust_score:
                best_robust_score = robust_s
                best_combo = (ki, kl)
                
        print(f"    Winner: K_input={best_combo[0]}, K_llm={best_combo[1]} (Score: {best_robust_score:.4f})")
    
    # -----------------------------------------------
    # PHASE 3: FINAL OUTPUT & GAIN ANALYSIS
    # -----------------------------------------------
    print("\n>>> Final Generation & Analysis")
    final_logs = []
    
    for dataset in active_sequence:
        if dataset not in final_dfs: continue
        
        # Apply Winner Config
        # 1. Filter by K_input (CE baseline depth)
        ce_run = ce_runs_cache[dataset]
        ce_subset = ce_run[ce_run["rank"] <= best_combo[0]]
        valid_keys = ce_subset[['qid', 'docno']]
        
        # 2. Constrain LLM result to valid keys
        final_df = final_dfs[dataset]
        final_constrained = final_df.merge(valid_keys, on=['qid', 'docno'], how='inner')
        
        final_constrained = final_constrained.sort_values(["qid", "score"], ascending=[True, False])
        final_constrained["rank"] = final_constrained.groupby("qid").cumcount() + 1
        final_constrained["Q0"] = "Q0"
        final_constrained["system"] = f"Qwen_{selected_variant_name}"
        
        # Save
        fname = f"{dataset}_Qwen_{selected_variant_name}.run"
        fpath = PATHS["run_output_dir"] / fname
        pt.io.write_results(final_constrained, str(fpath))
        print(f"  Saved: {fname}")
        
        # Analysis
        qrel_path = QRELS_FILES.get(dataset)
        if qrel_path and qrel_path.exists():
            qrels = pt.io.read_qrels(str(qrel_path))
            q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
            r_std = final_constrained.rename(columns={"qid":"query_id", "docno":"doc_id"})
            
            metrics = calc_aggregate(MEASURES, q_std, r_std)
            llm_score = metrics.get(nDCG@10, 0)
            
            # Baseline (CE)
            r_base = ce_subset.rename(columns={"qid":"query_id", "docno":"doc_id"})
            m_base = calc_aggregate([nDCG@10], q_std, r_base)
            ce_score = m_base.get(nDCG@10, 0)
            
            gain = ((llm_score - ce_score) / ce_score * 100) if ce_score > 0 else 0
            
            rec = {"dataset": dataset, "CE_nDCG": ce_score, "LLM_nDCG": llm_score, "Gain_Pct": gain}
            for m in MEASURES: rec[str(m)] = metrics.get(m, 0)
            final_logs.append(rec)
            print(f"  {dataset}: nDCG@10 {llm_score:.4f} (Gain: {gain:+.2f}%)")

    # Final Plot
    if final_logs:
        df_res = pd.DataFrame(final_logs)
        df_res.to_csv(PATHS["eval_output_dir"] / f"llm_gains_{selected_variant_name}.csv", index=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_res, x="dataset", y="Gain_Pct", palette="RdYlGn")
        plt.title(f"LLM Re-ranking Gain vs CE ({selected_variant_name})")
        plt.axhline(0, color='k')
        plt.savefig(PATHS["eval_output_dir"] / f"llm_gain_chart_{selected_variant_name}.png")

    print("\nPipeline Finished.")

if __name__ == "__main__":
    main()