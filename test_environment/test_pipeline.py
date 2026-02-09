import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import pyterrier as pt
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
import os
import json
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
import ir_measures
from ir_measures import Qrel, ScoredDoc, calc_aggregate, nDCG, AP, RR, Success, R, P
from tqdm import tqdm
import gc

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
os.environ["JAVA_OPTS"] = "-Xmx100g -Xms16g -Dindex.meta.data-source=fileinmem"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if not pt.java.started():
    pt.java.init()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()
print(f"Hardware: {NUM_GPUS} GPUs detected.")

# --- PARAMETERS ---
# Local Model Directory
LOCAL_MODEL_DIR = "/media/12TB/shared/models"

# Stage 1: Sparse
BM25_PARAMS = {'c': 0.01, 'bm25.b': 0.55}

# Stage 2: Dense
DENSE_MODELS = {
    "colbertv2.0": {
        "hf_id": "colbert-ir/colbertv2.0",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "colbertv2.0"),
        "type": "colbert",
        "batch": 256
    },
    "contriever": {
        "hf_id": "facebook/contriever",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "contriever"),
        "type": "dot",
        "batch": 512
    },
    "e5-large": {
        "hf_id": "intfloat/e5-large",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "intfloat", "e5-large"),
        "type": "e5",
        "batch": 1024
    }
}
K_DENSE_INPUT = 1000 # Retrieve top 1000 from sparse

# Stage 3: Cross-Encoder
CE_MODEL = "castorini/monot5-large-msmarco-10k"
CE_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "monot5-large-msmarco")
CE_BATCH = 256
K_CE_POOL_SPARSE = 100 # Take top 100 from BM25
K_CE_POOL_DENSE = 100  # Take top 100 from Dense Fusion
# Total pool approx 200

# Stage 4: LLM
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
LLM_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "qwen", "qwen2.5-72b-awq")
K_LLM_INPUT = 30 # Input to LLM
LLM_CONTEXT_CHARS = 500 # Increased context window
LLM_VLLM_BATCH = 32 # Number of queries to batch in vLLM

MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10, AP@10]

# ==========================================
# 2. HELPER CLASSES
# ==========================================

class ReRankDataset(Dataset):
    def __init__(self, queries, docs, tokenizer, model_type="dot"):
        self.queries = queries
        self.docs = docs
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __len__(self): return len(self.queries)

    def __getitem__(self, idx):
        q, d = self.queries[idx], self.docs[idx]
        if self.model_type == "e5":
            q, d = f"query: {q}", f"passage: {d}"
        return q, d

class DenseScorer:
    def __init__(self, config):
        self.type = config['type']
        self.batch_size = config['batch'] * max(1, NUM_GPUS)

        # Try local path first, fallback to HF hub
        model_path = config.get('local_path', config['hf_id'])
        if not os.path.exists(model_path):
            model_path = config['hf_id']

        print(f"Loading {config['hf_id']} from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16).to(DEVICE).eval()
        if NUM_GPUS > 1: self.model = nn.DataParallel(self.model)

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def score(self, queries, docs):
        dataset = ReRankDataset(queries, docs, self.tokenizer, self.type)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)
        all_scores = []

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for q_batch, d_batch in tqdm(loader, desc=f"Dense Scoring ({self.type})", leave=False):
                q_in = self.tokenizer(list(q_batch), padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
                d_in = self.tokenizer(list(d_batch), padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)

                if self.type == "colbert":
                    q_out = torch.nn.functional.normalize(self.model(**q_in)[0], p=2, dim=2)
                    d_out = torch.nn.functional.normalize(self.model(**d_in)[0], p=2, dim=2)
                    sim = torch.bmm(q_out, d_out.transpose(1, 2))
                    scores = torch.sum(torch.max(sim, dim=2)[0], dim=1)
                else:
                    q_emb = torch.nn.functional.normalize(self.mean_pooling(self.model(**q_in)[0], q_in['attention_mask']), p=2, dim=1)
                    d_emb = torch.nn.functional.normalize(self.mean_pooling(self.model(**d_in)[0], d_in['attention_mask']), p=2, dim=1)
                    scores = torch.sum(q_emb * d_emb, dim=1)
                all_scores.append(scores.float().cpu().numpy())
        return np.concatenate(all_scores)

class MonoT5Scorer:
    def __init__(self, model_name, local_path):
        # Try local path first, fallback to HF hub
        model_path = local_path if os.path.exists(local_path) else model_name
        print(f"Loading MonoT5 from {model_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(DEVICE).eval()
        if NUM_GPUS > 1: self.model = nn.DataParallel(self.model)
        self.true_tok = self.tokenizer.encode("true")[0]
        self.false_tok = self.tokenizer.encode("false")[0]
        self.dec_start_id = self.model.module.config.decoder_start_token_id if hasattr(self.model, "module") else self.model.config.decoder_start_token_id

    def score(self, pairs_text):
        batch_size = CE_BATCH * max(1, NUM_GPUS)
        all_scores = []

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for i in tqdm(range(0, len(pairs_text), batch_size), desc="MonoT5 Scoring", leave=False):
                batch = pairs_text[i:i+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
                dec_in = torch.full((len(batch), 1), self.dec_start_id, device=DEVICE)

                out = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, decoder_input_ids=dec_in, use_cache=False)
                logits = out.logits[:, 0, [self.true_tok, self.false_tok]]
                scores = torch.softmax(logits, dim=-1)[:, 0]
                all_scores.append(scores.float().cpu().numpy())
        return np.concatenate(all_scores)

class QwenRanker:
    def __init__(self, model_name, local_path):
        model_path = local_path if os.path.exists(local_path) else model_name
        print(f"Loading Qwen from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        from vllm import LLM, SamplingParams
        self.vllm_engine = LLM(
            model=model_path,
            tensor_parallel_size=NUM_GPUS,
            quantization="awq_marlin",
            dtype="float16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
        )
        print(f"vLLM engine ready (tensor_parallel={NUM_GPUS}).")

    def _build_prompt(self, query, docs_list):
        cands = ""
        id_map = {}
        for idx, (did, txt) in enumerate(docs_list):
            safe_text = txt[:LLM_CONTEXT_CHARS].replace("\n", " ")
            cands += f"[{idx}] {safe_text}\n"
            id_map[idx] = did

        prompt = f"""You are an expert search relevance ranker.
Your task is to re-rank the following candidate documents based on their relevance to the user query.
The goal is to place the true relevant document at the very top (Rank 1).

Query: {query}

Candidates:
{cands}
Instructions:
1. Analyze the query and the candidates carefully.
2. Output the ranking as a list of IDs in order of relevance, from most relevant to least relevant.
3. Use the format: [ID] > [ID] > [ID] ...
4. Only output the ranking, no explanation.

Ranking:"""

        msgs = [{"role": "system", "content": "You are a Ranker."}, {"role": "user", "content": prompt}]
        text_in = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text_in, id_map

    def _parse_response(self, resp, id_map, docs_list):
        matches = re.findall(r"\[(\d+)\]", resp)
        reranked = []
        seen = set()
        score = 1000.0

        for mid in matches:
            try:
                mid = int(mid)
                if mid in id_map and mid not in seen:
                    reranked.append({'docno': id_map[mid], 'score': score})
                    score -= 1.0
                    seen.add(mid)
            except: continue

        for idx, (did, _) in enumerate(docs_list):
            if idx not in seen:
                reranked.append({'docno': did, 'score': score})
                score -= 1.0
        return reranked

    def rerank_batch(self, query_groups):
        """Batch rerank: query_groups is list of (qid, query_text, [(docno, text), ...])"""
        prompts = []
        id_maps = []
        docs_lists = []

        for qid, q_text, docs_list in query_groups:
            text_in, id_map = self._build_prompt(q_text, docs_list)
            prompts.append(text_in)
            id_maps.append(id_map)
            docs_lists.append(docs_list)

        outputs = self.vllm_engine.generate(prompts, self.sampling_params)

        results = []
        for i, output in enumerate(outputs):
            resp = output.outputs[0].text
            reranked = self._parse_response(resp, id_maps[i], docs_lists[i])
            results.append(reranked)
        return results

# ==========================================
# 3. UTILS
# ==========================================
def load_env(path="env.json"):
    if not os.path.exists(path): path = "../env.json"
    with open(path) as f: return json.load(f)

def pandas_rrf(runs, k=60):
    """
    Performs RRF and returns a DataFrame with 'rank' column suitable for write_results.
    """
    parts = []
    for r in runs:
        r = r.sort_values(["qid", "score"], ascending=[True, False]) # Sort by QID, then Score Desc
        # Recalculate rank just in case
        r["rank"] = r.groupby("qid").cumcount() + 1
        r["rrf"] = 1.0 / (k + r["rank"])
        parts.append(r[["qid", "docno", "rrf"]])

    combined = pd.concat(parts)
    fused = combined.groupby(["qid", "docno"], as_index=False)["rrf"].sum()
    fused = fused.rename(columns={"rrf": "score"})

    # CRITICAL FIX: Add Rank, Q0, System columns for TREC format
    fused = fused.sort_values(["qid", "score"], ascending=[True, False])
    fused["rank"] = fused.groupby("qid").cumcount() + 1
    fused["Q0"] = "Q0"
    fused["system"] = "RRF_fusion"

    return fused

# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================
def main():
    env = load_env()

    # Paths
    run_dir = env['paths']['test_run_directory']
    eval_dir = env['paths']['test_eval_directory']
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"{'='*60}\nSTARTING ISOLATED TEST PIPELINE (RESUMABLE)\n{'='*60}")

    # 1. Load Data
    print("Loading Index & Test Data...")
    index = pt.IndexFactory.of(env['paths']['index_path'])

    q_path = "./original-queries/trec-tot_Llama-3.3-70B-Instruct_hybrid_cot_test-2025-queries.jsonl"
    print(f"Queries (Mistral): {q_path}")

    # Robust JSONL loader: read line-by-line and handle common minor malformations
    records = []
    with open(q_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Some lines contain a trailing comma after the JSON object — strip it
            if line.endswith(','):
                line = line[:-1]
            # Fix a stray comma before the closing brace: '...,}' -> '...}'
            line = re.sub(r',\s*}$', '}', line)
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                # Last-resort: remove non-JSON-friendly control chars and try again
                cleaned = line.replace('\x00', '')
                cleaned = re.sub(r'\s+$', '', cleaned)
                obj = json.loads(cleaned)
                records.append(obj)

    queries = pd.DataFrame.from_records(records)
    if 'text' in queries.columns:
        queries = queries.rename(columns={'text': 'query'})
    if 'query_id' in queries.columns:
        queries = queries.rename(columns={'query_id': 'qid'})
    queries['qid'] = queries['qid'].astype(str)
    # Tokenize for Terrier
    queries_tok = pt.rewrite.tokenise()(queries)

    # ---------------------------------------------------------
    # STAGE 1: SPARSE (BM25)
    # ---------------------------------------------------------
    print("\n>>> STAGE 1: BM25 (Sparse)")
    stage1_file = os.path.join(run_dir, "test_stage1_bm25_original.run")

    if os.path.exists(stage1_file):
        print(f"  [CHECKPOINT] Found existing run: {stage1_file}")
        sparse_run = pt.io.read_results(stage1_file)
    else:
        # Updated API Call: Use pt.terrier.Retriever instead of BatchRetrieve
        bm25 = pt.terrier.Retriever(index, wmodel="BM25", controls=BM25_PARAMS, verbose=True)
        sparse_run = bm25.transform(queries_tok)
        pt.io.write_results(sparse_run, stage1_file)
        print(f"  Retrieved {len(sparse_run)} docs.")

    # ---------------------------------------------------------
    # STAGE 2: DENSE (Ensemble)
    # ---------------------------------------------------------
    print("\n>>> STAGE 2: Dense Retrieval (Fusion)")
    stage2_file = os.path.join(run_dir, "test_stage2_dense_fusion_original.run")

    # Optimization: If fusion exists, we can skip everything
    if os.path.exists(stage2_file):
        print(f"  [CHECKPOINT] Found existing Fusion run: {stage2_file}")
        dense_fused = pt.io.read_results(stage2_file)
    else:
        # Filter Sparse to Top-1000 for input
        sparse_run = sparse_run.sort_values(["qid", "score"], ascending=[True, False])
        sparse_run["rank"] = sparse_run.groupby("qid").cumcount() + 1
        candidates_dense = sparse_run[sparse_run["rank"] <= K_DENSE_INPUT].copy()

        # Fetch Text + Query
        text_pipe = pt.text.get_text(index, "text")
        candidates_text = text_pipe.transform(candidates_dense)
        candidates_text = candidates_text.merge(queries[['qid', 'query']], on='qid', how='left')

        dense_results = []

        # Iterate through models with checkpointing for each
        for name, cfg in DENSE_MODELS.items():
            model_run_file = os.path.join(run_dir, f"test_stage2_{name}.run")

            if os.path.exists(model_run_file):
                print(f"  [CHECKPOINT] Found existing run for {name}")
                df = pt.io.read_results(model_run_file)
            else:
                print(f"  Running {name}...")
                scorer = DenseScorer(cfg)
                scores = scorer.score(candidates_text["query"].tolist(), candidates_text["text"].tolist())

                df = candidates_text.copy()
                df["score"] = scores
                df["system"] = name
                df = df.sort_values(["qid", "score"], ascending=[True, False])
                df["rank"] = df.groupby("qid").cumcount() + 1
                df["Q0"] = "Q0"

                # Save intermediate model run
                pt.io.write_results(df, model_run_file)

                # Clear VRAM
                del scorer
                torch.cuda.empty_cache()
                gc.collect()

            dense_results.append(df)

        # RRF Fusion
        print("  Fusing Dense Runs...")
        dense_fused = pandas_rrf(dense_results, k=60)
        pt.io.write_results(dense_fused, stage2_file)

    # ---------------------------------------------------------
    # STAGE 3: CROSS-ENCODER (Hybrid Pool)
    # ---------------------------------------------------------
    print("\n>>> STAGE 3: Cross-Encoder (Hybrid Pool)")
    stage3_file = os.path.join(run_dir, "test_stage3_ce_original.run")

    if os.path.exists(stage3_file):
        print(f"  [CHECKPOINT] Found existing CE run: {stage3_file}")
        pool = pt.io.read_results(stage3_file)

        # We need query/text columns for next stage if they aren't in standard run file
        # Standard run file usually drops 'text' and 'query'. We need to re-fetch if loaded from disk.
        # But wait, next stage needs text. Let's re-fetch text to be safe.
        pool = pool.merge(queries[['qid', 'query']], on='qid')
        text_pipe = pt.text.get_text(index, "text")
        pool = text_pipe.transform(pool)
    else:
        # Pool Construction: Union of Top-100 BM25 + Top-100 Dense
        # 1. BM25 Top 100
        bm25_top = sparse_run[sparse_run["rank"] <= K_CE_POOL_SPARSE][['qid', 'docno']]

        # 2. Dense Top 100
        dense_fused = dense_fused.sort_values(["qid", "score"], ascending=[True, False])
        dense_fused["rank"] = dense_fused.groupby("qid").cumcount() + 1
        dense_top = dense_fused[dense_fused["rank"] <= K_CE_POOL_DENSE][['qid', 'docno']]

        # 3. Union
        pool = pd.concat([bm25_top, dense_top]).drop_duplicates(subset=['qid', 'docno'])
        print(f"  Hybrid Pool Size: {len(pool)} (Approx {len(pool)/len(queries):.1f} per query)")

        # Fetch Text for Pool
        pool = pool.merge(queries[['qid', 'query']], on='qid')
        text_pipe = pt.text.get_text(index, "text")
        pool = text_pipe.transform(pool)

        # Prepare text for MonoT5
        pairs = [f"Query: {r.query} Document: {r.text} Relevant:" for r in pool.itertuples()]

        # Score
        ce_scorer = MonoT5Scorer(CE_MODEL, CE_LOCAL_PATH)
        ce_scores = ce_scorer.score(pairs)
        del ce_scorer
        torch.cuda.empty_cache()
        gc.collect()

        # FIX: Explicit reconstruction to ensure cleanliness and avoid KeyError
        pool["score"] = ce_scores
        pool = pool.sort_values(["qid", "score"], ascending=[True, False])
        # Reset Index to prevent ambiguity
        pool = pool.reset_index(drop=True)
        pool["rank"] = pool.groupby("qid").cumcount() + 1
        pool["Q0"] = "Q0"
        pool["system"] = "monoT5"

        # Force select columns to verify they exist before passing to PyTerrier
        final_pool_df = pool[["qid", "docno", "rank", "score", "Q0", "system"]]

        pt.io.write_results(final_pool_df, stage3_file)

    # ---------------------------------------------------------
    # STAGE 4: LLM RE-RANKING (vLLM batched)
    # ---------------------------------------------------------
    print("\n>>> STAGE 4: LLM Re-ranking (Qwen-72B via vLLM)")
    stage4_file = os.path.join(run_dir, "test_final_qwen_original.run")

    if os.path.exists(stage4_file):
        print(f"  [CHECKPOINT] Found existing LLM run: {stage4_file}")
        final_run = pt.io.read_results(stage4_file)
    else:
        # Free GPU memory before loading LLM
        torch.cuda.empty_cache()
        gc.collect()

        # Filter CE results to Top-30 (Input for LLM)
        pool = pool.sort_values(["qid", "score"], ascending=[True, False])
        pool["rank"] = pool.groupby("qid").cumcount() + 1

        llm_candidates = pool[pool["rank"] <= K_LLM_INPUT].copy()

        # Initialize LLM
        llm = QwenRanker(LLM_MODEL, LLM_LOCAL_PATH)

        # Build all query groups
        grouped = llm_candidates.groupby("qid")
        query_groups = []
        for qid, group in grouped:
            q_text = group.iloc[0]['query']
            docs = [(row.docno, row.text) for row in group.itertuples()]
            query_groups.append((qid, q_text, docs))

        print(f"  Re-ranking Top-{K_LLM_INPUT} for {len(query_groups)} queries (batch_size={LLM_VLLM_BATCH})...")

        final_rows = []
        for batch_start in tqdm(range(0, len(query_groups), LLM_VLLM_BATCH), desc="vLLM Batches"):
            batch = query_groups[batch_start:batch_start + LLM_VLLM_BATCH]
            batch_results = llm.rerank_batch(batch)

            for (qid, _, _), reranked in zip(batch, batch_results):
                for r in reranked:
                    final_rows.append({"qid": qid, "docno": r['docno'], "score": r['score']})

        # Merge with Tail (CE ranks 31+)
        llm_df = pd.DataFrame(final_rows)
        ce_tail = pool[pool["rank"] > K_LLM_INPUT][['qid', 'docno', 'score']]

        final_run = pd.concat([llm_df, ce_tail])

        # Final Format ensure
        final_run = final_run.sort_values(["qid", "score"], ascending=[True, False])
        final_run = final_run.reset_index(drop=True) # Checkpoint safety
        final_run["rank"] = final_run.groupby("qid").cumcount() + 1
        final_run["Q0"] = "Q0"
        final_run["system"] = "Qwen_72B"

        pt.io.write_results(final_run, stage4_file)

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------
    print("\n>>> FINAL EVALUATION")
    qrels = pt.io.read_qrels(env['paths']['test_qrels_path'])

    # Standardize column names for ir_measures
    q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
    r_std = final_run.rename(columns={"qid":"query_id", "docno":"doc_id"})

    metrics = calc_aggregate(MEASURES, q_std, r_std)

    # Save Metrics
    with open(os.path.join(eval_dir, "test_metrics.json"), "w") as f:
        # Convert metric objects to string keys
        json.dump({str(k): v for k, v in metrics.items()}, f, indent=2)

    print("\nFinal Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nPipeline Finished. Results saved to {eval_dir}")

if __name__ == "__main__":
    main()
