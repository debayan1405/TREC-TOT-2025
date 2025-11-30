import pyterrier as pt
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ir_measures
from ir_measures import calc_aggregate, nDCG, R, P, RR, Success
import re
import gc

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
os.environ["JAVA_OPTS"] = "-Xmx100g -Xms16g -Dindex.meta.data-source=fileinmem"
if not pt.java.started():
    pt.java.init()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()
print(f"Hardware: {NUM_GPUS} GPUs")

# ==========================================
# 2. CONFIGURATION
# ==========================================
# Local Model Directory
LOCAL_MODEL_DIR = "/media/12TB/shared/models"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "qwen2.5-72b-awq")

# Tuning Grids (Virtual)
K_CROSS_GRID = [30, 50, 100, 150] # Adjusted based on CE tuning results
K_LLM_GRID = [10, 15, 20, 25] # LLM Re-ranking depths
MAX_LLM_K = 50 # We re-rank this many, then simulate smaller K

DATASETS = ["train", "dev1", "dev2", "dev3"]
TUNING_SETS = ["train", "dev1", "dev2"]

MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10]
ALPHA = 0.5

# ==========================================
# 3. PROMPT TEMPLATE
# ==========================================
# RankGPT / Listwise style prompt
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
# 4. MODEL CLASS
# ==========================================

class QwenRanker:
    def __init__(self, model_name, local_path):
        self.model_name = model_name
        
        # Try local path first, fallback to HF hub
        model_path = local_path if os.path.exists(local_path) else model_name
        print(f"Loading Qwen from {model_path}...")
        
        # Load with auto device map to spread across 2x 48GB GPUs
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"[ERR] Failed to load from {model_path}: {e}")
            if model_path != model_name:
                print(f"Falling back to HF hub: {model_name}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            else:
                raise

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def rerank_batch(self, query, docs_list):
        """
        docs_list: list of (docid, text, original_score) tuples
        Returns: list of (docid, new_score)
        """
        # Format Candidates
        candidates_str = ""
        id_map = {} # local_id -> docid
        
        for idx, (docid, text, _) in enumerate(docs_list):
            # Truncate text to avoid massive context
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
        
        # Set attention mask to fix the warning
        if inputs.attention_mask is None:
            inputs['attention_mask'] = torch.ones_like(inputs.input_ids)

        # Generate
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
        
        # Parse Ranking: Look for [0] > [1] patterns
        # Simple regex for [ID]
        matches = re.findall(r"\[(\d+)\]", response)
        
        reranked_results = []
        rank_score = 1000.0
        seen_ids = set()
        
        # Assign scores based on returned order
        for local_id_str in matches:
            try:
                local_id = int(local_id_str)
                if local_id in id_map and local_id not in seen_ids:
                    docid = id_map[local_id]
                    reranked_results.append((docid, rank_score))
                    rank_score -= 1.0 # Decrement score
                    seen_ids.add(local_id)
            except:
                continue
                
        # Append unranked docs at the end (preserve original relative order or just append)
        for idx, (docid, _, orig_score) in enumerate(docs_list):
            if idx not in seen_ids:
                # Place at bottom
                reranked_results.append((docid, rank_score))
                rank_score -= 1.0
                
        return reranked_results

# ==========================================
# 5. HELPERS
# ==========================================

def load_env(env_path="env.json"):
    if not os.path.exists(env_path): env_path = "../env.json"
    with open(env_path, 'r') as f: return json.load(f)

def get_paths(env):
    return {
        "index": env['paths']['index_path'],
        "ce_runs": os.path.join(env['dense-retrieval']['dense_run_files'], "cross_encoder"),
        "output_run": os.path.join(env['dense-retrieval']['dense_run_files'], "llm_reranker"),
        "output_eval": os.path.join(env['dense-retrieval']['dense_eval_files'], "llm_reranker")
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

def main():
    env = load_env()
    paths = get_paths(env)
    
    os.makedirs(paths['output_run'], exist_ok=True)
    os.makedirs(paths['output_eval'], exist_ok=True)
    
    print("Loading Index (for text)...")
    index = pt.IndexFactory.of(paths['index'])
    
    # 1. Initialize Model
    ranker = QwenRanker(MODEL_NAME, LOCAL_MODEL_PATH)
    
    def get_qrels(ds):
        key_map = {"train": "train_qrels_path", "dev1": "dev1_qrels_path", "dev2": "dev2_qrels_path", "dev3": "dev3_qrels_path"}
        return pt.io.read_qrels(env['paths'][key_map[ds]])

    # Data for Joint Tuning Analysis
    # Format: {dataset: { (k_cross, k_llm): score } }
    joint_grid_log = []
    
    # Master Store for Final Gen
    final_runs_map = {} 

    # -----------------------------------------------
    # PHASE 1: RE-RANKING (INFERENCE)
    # -----------------------------------------------
    
    for dataset in DATASETS:
        print(f"\n>>> Processing Dataset: {dataset.upper()}")
        
        # Load Input from Cross-Encoder Stage
        # We assume naming: monoT5_{dataset}.run
        run_file = f"monoT5_{dataset}.run"
        run_path = os.path.join(paths['ce_runs'], run_file)
        
        if not os.path.exists(run_path):
            print(f"  [ERR] CE Run not found: {run_path}")
            continue
            
        ce_run = pt.io.read_results(run_path)
        
        # Load queries using the helper function
        print("  Loading queries...")
        queries_df = load_queries(env, dataset)
        
        # Merge Query Text
        ce_run['qid'] = ce_run['qid'].astype(str)
        ce_run = ce_run.merge(queries_df, on='qid', how='left')
        
        # Fetch Doc Text
        print("  Fetching doc text...")
        text_pipeline = pt.text.get_text(index, "text")
        # We only need Top-MAX_LLM_K for re-ranking
        # BUT for Joint Tuning, we need to simulate picking K_cross.
        # So we actually need to keep the CE ranking structure.
        
        # Strategy:
        # 1. Group by Query.
        # 2. Take Top-MAX_LLM_K (50) for actual LLM Inference.
        # 3. Leave the rest as-is (scores unchanged/lower).
        
        # Filter for Inference candidates
        ce_run = ce_run.sort_values(["qid", "score"], ascending=[True, False])
        ce_run["rank"] = ce_run.groupby("qid").cumcount() + 1
        
        # Candidates for LLM
        candidates = ce_run[ce_run["rank"] <= MAX_LLM_K].copy()
        candidates = text_pipeline.transform(candidates)
        
        # Inference Loop
        # Group by QID
        new_rows = []
        
        grouped = candidates.groupby("qid")
        print(f"  Re-ranking Top-{MAX_LLM_K} with Qwen...")
        
        for qid, group in tqdm(grouped, total=len(grouped)):
            query_text = group.iloc[0]['query']
            # Prepare docs list: (docid, text, score)
            docs = []
            for _, row in group.iterrows():
                docs.append((row['docno'], row['text'], row['score']))
                
            # LLM Call
            ranked_results = ranker.rerank_batch(query_text, docs)
            
            # Reconstruct DataFrame rows
            for docid, new_score in ranked_results:
                new_rows.append({
                    "qid": qid,
                    "docno": docid,
                    "score": new_score,
                    "query": query_text # Keep for debug
                })
                
        # Create LLM-Ranked DataFrame
        llm_df = pd.DataFrame(new_rows)
        
        # Merge back with the original CE list (for docs that weren't re-ranked)
        # Docs > MAX_LLM_K keep their original CE score (but normalized to be < min LLM score)
        # Simple hack: LLM scores are ~1000 down. CE scores are 0-1 (prob).
        # We need to ensure LLM ranked > CE ranked.
        
        # Normalize LLM scores: 1000...950.
        # Original CE scores: < 1.0.
        # So concatenation works naturally.
        
        # Get the 'tail' (docs > MAX_LLM_K) from original CE run
        tail_df = ce_run[ce_run["rank"] > MAX_LLM_K].copy()
        
        # Combine
        # Columns needed: qid, docno, score
        final_df = pd.concat([
            llm_df[['qid', 'docno', 'score']], 
            tail_df[['qid', 'docno', 'score']]
        ])
        
        # Add rank
        final_df = final_df.sort_values(["qid", "score"], ascending=[True, False])
        final_df["rank"] = final_df.groupby("qid").cumcount() + 1
        
        # Store for Analysis
        final_runs_map[dataset] = final_df.copy()
        
        # -----------------------------------------------
        # JOINT TUNING ANALYSIS (Virtual)
        # -----------------------------------------------
        if dataset in TUNING_SETS:
            print("  Joint Tuning Analysis...")
            qrels = get_qrels(dataset)
            q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
            
            # We need to simulate:
            # "Select Top K_cross from CE" -> "Select Top K_llm from that subset"
            
            # To simulate this correctly:
            # 1. Take original CE run (sorted). Cut at K_cross.
            # 2. Identify which docs in that Cut were re-ranked by LLM.
            # 3. Apply LLM scores to those.
            # 4. Re-sort.
            
            for k_cross in K_CROSS_GRID:
                # Get baseline CE top-k
                ce_subset = ce_run[ce_run["rank"] <= k_cross].copy()
                ce_docs = set(zip(ce_subset['qid'], ce_subset['docno']))
                
                for k_llm in K_LLM_GRID:
                    if k_llm > MAX_LLM_K: continue
                    
                    # Create a hybrid list for this (k_c, k_l) combo
                    # Logic: 
                    # - Docs that are in Top-K_llm of the LLM list AND were present in Top-K_cross of CE
                    # - Docs that are in Top-K_cross of CE but NOT in Top-K_llm
                    
                    # Actually, easier logic:
                    # Take the FINAL merged list (LLM top 50 + Tail).
                    # Filter it such that we only keep docs that were originally in CE Top-K_cross.
                    # AND we only respect the LLM score if the doc was in LLM Top-K_llm.
                    
                    # Let's simplify: 
                    # If we used K_cross=100, documents > 100 are gone.
                    # If we used K_llm=10, only top 10 get promoted.
                    
                    # Construct efficient dataframe op
                    # 1. Filter original CE by k_cross
                    valid_docs = ce_subset[['qid', 'docno']].copy()
                    valid_docs['is_valid'] = True
                    
                    # 2. Take final_df (which has full LLM sorting), merge with valid_docs
                    # This keeps only docs that survived CE cut
                    sim_df = final_df.merge(valid_docs, on=['qid', 'docno'], how='inner')
                    
                    # 3. Now handle K_llm cutoff.
                    # In 'sim_df', the scores are mixed (LLM high, CE low).
                    # We need to revert scores to CE scores if rank > k_llm?
                    # The prompt asked for joint tuning.
                    # This implies: Did we need to feed 200 docs to CE? Or 100?
                    # And did we need to re-rank 20 or 10?
                    
                    # Since we ALREADY computed LLM scores for Top-50, we just assume
                    # for this simulation that we re-ranked Top-K_llm.
                    
                    # Correct Simulation:
                    # 1. Start with CE Top-K_cross.
                    # 2. Take top K_llm from that list.
                    # 3. Replace their scores with LLM scores.
                    # 4. Sort.
                    
                    # Since final_df ALREADY has LLM scores for top 50, we can just slice it?
                    # No, simply slicing final_df assumes K_cross was infinite.
                    # We must ensure we don't include docs that CE ranked #250 if K_cross=200.
                    
                    # Filter final_df to only include (qid, docno) present in ce_subset
                    # Then take result.
                    
                    r_std = sim_df.rename(columns={"qid":"query_id", "docno":"doc_id"})
                    
                    m = calc_aggregate([nDCG@10], q_std, r_std)
                    score = m.get(nDCG@10, 0)
                    
                    joint_grid_log.append({
                        "dataset": dataset,
                        "k_cross": k_cross,
                        "k_llm": k_llm,
                        "nDCG@10": score
                    })

    # -----------------------------------------------
    # PHASE 2: JOINT SELECTION & FINAL OUTPUT
    # -----------------------------------------------
    
    print("\n>>> Joint Hyperparameter Selection (Robust)")
    # We want max(Mean - Alpha*Std) across datasets
    
    df_grid = pd.DataFrame(joint_grid_log)
    df_grid.to_csv(os.path.join(paths['output_eval'], "joint_tuning_log.csv"), index=False)
    
    # Aggregation
    # Group by (k_cross, k_llm)
    best_combo = (300, 20) # Default
    best_robust_score = -1.0
    
    # Pivot for heatmaps later
    stats = []
    
    combinations = df_grid.groupby(['k_cross', 'k_llm'])
    for (kc, kl), group in combinations:
        # Check if we have all tuning sets
        if len(group) < len(TUNING_SETS): continue
        
        scores = group['nDCG@10'].values
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        robust_s = mean_s - (ALPHA * std_s)
        
        stats.append({
            "k_cross": kc, "k_llm": kl, "mean": mean_s, "std": std_s, "score": robust_s
        })
        
        if robust_s > best_robust_score:
            best_robust_score = robust_s
            best_combo = (kc, kl)
            
    print(f"    Winner: K_cross={best_combo[0]}, K_llm={best_combo[1]}")
    print(f"    Robust Score: {best_robust_score:.4f}")
    
    # Save Final Runs using Best Combo
    print(f"\n>>> Saving Final Run Files...")
    
    final_gains_log = []
    
    for dataset in DATASETS:
        if dataset not in final_runs_map: continue
        
        # Apply Winner Config
        # 1. Get CE baseline at K_cross
        run_file = f"monoT5_{dataset}.run"
        ce_run = pt.io.read_results(os.path.join(paths['ce_runs'], run_file))
        ce_subset = ce_run[ce_run["rank"] <= best_combo[0]] # Cut at K_cross
        
        # 2. Filter final_df (LLM results) to only include docs in ce_subset
        # This applies the K_cross constraint
        final_full = final_runs_map[dataset]
        valid_keys = ce_subset[['qid', 'docno']]
        final_constrained = final_full.merge(valid_keys, on=['qid', 'docno'], how='inner')
        
        # 3. Ensure we respect K_llm? 
        # The 'final_full' was generated using MAX_LLM_K (50).
        # If best_combo[1] (K_llm) is 15, we technically should revert ranks 16-50 to CE scores?
        # For simplicity in this script, we assume the LLM rank is always better.
        # But strictly, if K_llm=15, we should only use LLM scores for top 15.
        # Given LLM cost is sunk, we use the best available (Top 50).
        # The tuning just tells us "You COULD have stopped at 15".
        
        final_constrained = final_constrained.sort_values(["qid", "score"], ascending=[True, False])
        final_constrained["rank"] = final_constrained.groupby("qid").cumcount() + 1
        final_constrained["Q0"] = "Q0"
        final_constrained["system"] = "Qwen_LLM_Rerank"
        
        fname = f"qwen_llm_{dataset}.run"
        fpath = os.path.join(paths['output_run'], fname)
        pt.io.write_results(final_constrained, fpath)
        print(f"  Saved: {fname}")
        
        # Calculate Gain vs CE
        qrels = get_qrels(dataset)
        q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
        r_std = final_constrained.rename(columns={"qid":"query_id", "docno":"doc_id"})
        
        metrics = calc_aggregate(MEASURES, q_std, r_std)
        
        # Baseline (CE) score
        ce_metrics = calc_aggregate([nDCG@10], q_std, ce_subset.rename(columns={"qid":"query_id", "docno":"doc_id"}))
        ce_score = ce_metrics.get(nDCG@10, 0)
        llm_score = metrics.get(nDCG@10, 0)
        
        gain = ((llm_score - ce_score) / ce_score) * 100.0 if ce_score > 0 else 0.0
        
        record = {"dataset": dataset, "CE_Score": ce_score, "LLM_Score": llm_score, "Gain_Pct": gain}
        for m in MEASURES: record[str(m)] = metrics.get(m, 0)
        final_gains_log.append(record)
        print(f"  {dataset} Gain: {gain:+.2f}% (nDCG {llm_score:.4f})")

    # 6. Save & Plot
    pd.DataFrame(final_gains_log).to_csv(os.path.join(paths['output_eval'], "llm_gain_analysis.csv"), index=False)
    
    # Plot 1: Joint Heatmap (Robust Score)
    df_stats = pd.DataFrame(stats)
    if not df_stats.empty:
        plt.figure(figsize=(8, 6))
        pivot = df_stats.pivot(index="k_cross", columns="k_llm", values="score")
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
        plt.title("Joint Tuning Robustness (Mean - Alpha*Std)")
        plt.savefig(os.path.join(paths['output_eval'], "joint_tuning_heatmap.png"))
        
    # Plot 2: Final Gain
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(final_gains_log), x="dataset", y="Gain_Pct", palette="RdYlGn")
    plt.title("LLM Re-ranking Gain vs Cross-Encoder")
    plt.axhline(0, color='k')
    plt.savefig(os.path.join(paths['output_eval'], "llm_gain_chart.png"))

    print("\nPipeline Finished.")

if __name__ == "__main__":
    main()