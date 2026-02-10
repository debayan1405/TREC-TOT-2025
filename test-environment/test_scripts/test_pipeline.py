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
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import ir_measures
from ir_measures import Qrel, ScoredDoc, calc_aggregate, nDCG, AP, RR, Success, R, P
from tqdm import tqdm
import gc
import sys

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

# --- PATHS ---
SCRIPT_DIR = Path(__file__).resolve().parent
# Assume project root is two levels up from test_scripts
PROJECT_ROOT = SCRIPT_DIR.parent.parent 

PATHS = {
    "rewritten_dir": (PROJECT_ROOT / "rewritten-queries").resolve(),
    "original_dir": (PROJECT_ROOT / "original_queries").resolve(),
    "index": "/media/12TB/shared/datasets/indices/trec-tot-2025/trec-tot-2025-pyterrier-index",
    "test_qrels": (PROJECT_ROOT / "qrel") / "test-2025-qrel.txt",
    "run_dir": (SCRIPT_DIR.parent / "test_run_files").resolve(),
    "eval_dir": (SCRIPT_DIR.parent / "test_eval_files").resolve()
}

# --- PARAMETERS ---
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
K_DENSE_INPUT = 1000 

# Stage 3: Cross-Encoder
CE_MODEL = "castorini/monot5-large-msmarco-10k"
CE_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "monot5-large-msmarco")
CE_BATCH = 256
K_CE_POOL_SPARSE = 100 
K_CE_POOL_DENSE = 100  

# Stage 4: LLM
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
LLM_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, "qwen", "qwen2.5-72b-awq")
K_LLM_INPUT = 30 
LLM_CONTEXT_CHARS = 500 
LLM_VLLM_BATCH = 32 

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
        model_path = config.get('local_path', config['hf_id'])
        if not os.path.exists(model_path): model_path = config['hf_id']

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
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
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
# 3. UTILS & FILE SCANNING
# ==========================================
def pandas_rrf(runs, k=60):
    parts = []
    for r in runs:
        r = r.sort_values(["qid", "score"], ascending=[True, False])
        r["rank"] = r.groupby("qid").cumcount() + 1
        r["rrf"] = 1.0 / (k + r["rank"])
        parts.append(r[["qid", "docno", "rrf"]])
    combined = pd.concat(parts)
    fused = combined.groupby(["qid", "docno"], as_index=False)["rrf"].sum()
    fused = fused.rename(columns={"rrf": "score"})
    fused = fused.sort_values(["qid", "score"], ascending=[True, False])
    fused["rank"] = fused.groupby("qid").cumcount() + 1
    fused["Q0"] = "Q0"
    fused["system"] = "RRF_fusion"
    return fused

def scan_for_test_files():
    """Scans rewritten and original directories for test-set files."""
    variants = {}
    
    # 1. Rewritten: {model}_test_rewritten_queries.jsonl
    if PATHS['rewritten_dir'].exists():
        for f in PATHS['rewritten_dir'].glob("*test_rewritten_queries.jsonl"):
            # e.g. llama_test_rewritten_queries.jsonl -> rewritten-llama
            match = re.match(r"^(?P<model>.*?)_test_rewritten_queries\.jsonl$", f.name)
            if match:
                variant = f"rewritten-{match.group('model')}"
                variants[variant] = f

    # 2. Summarized: test_summarized-{model}.jsonl
    if PATHS['rewritten_dir'].exists():
        for f in PATHS['rewritten_dir'].glob("test_summarized-*.jsonl"):
            # e.g. test_summarized-llama.jsonl -> summarized-llama
            match = re.match(r"^test_summarized-(?P<model>.*?)\.jsonl$", f.name)
            if match:
                variant = f"summarized-{match.group('model')}"
                variants[variant] = f

    # 3. Original: test-original.jsonl
    if PATHS['original_dir'].exists():
        f = PATHS['original_dir'] / "test-original.jsonl"
        if f.exists():
            variants["original"] = f
            
    return variants

def get_variant_selection(variants):
    print(f"\n{'='*40}")
    print("AVAILABLE TEST SET VARIANTS")
    print(f"{'='*40}")
    sorted_keys = sorted(variants.keys())
    for idx, key in enumerate(sorted_keys, 1):
        print(f"{idx}. {key}")
    
    while True:
        try:
            choice = input("\nSelect variant number: ")
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_keys):
                return sorted_keys[idx], variants[sorted_keys[idx]]
        except ValueError: pass

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, help="Query variant (e.g., rewritten-llama, original)")
    args = parser.parse_args()

    # Setup directories
    os.makedirs(PATHS["run_dir"], exist_ok=True)
    os.makedirs(PATHS["eval_dir"], exist_ok=True)
    
    # Select Variant
    variants = scan_for_test_files()
    if not variants:
        print("CRITICAL: No test query files found.")
        sys.exit(1)
        
    if args.variant:
        if args.variant not in variants:
            print(f"Error: Variant '{args.variant}' not found.")
            sys.exit(1)
        selected_variant = args.variant
        query_path = variants[args.variant]
    else:
        selected_variant, query_path = get_variant_selection(variants)
        
    print(f"{'='*60}\nSTARTING TEST PIPELINE | VARIANT: {selected_variant}\n{'='*60}")
    print(f"Query File: {query_path}")

    # Load Index & Queries
    print("Loading Index...")
    index = pt.IndexFactory.of(PATHS['index'])

    print("Loading Queries...")
    # Robust loading
    records = []
    with open(query_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            if line.endswith(','): line = line[:-1]
            line = re.sub(r',\s*}$', '}', line)
            try:
                obj = json.loads(line)
                records.append(obj)
            except: 
                # Last resort cleanup
                cleaned = line.replace('\x00', '')
                try: records.append(json.loads(cleaned))
                except: pass

    queries = pd.DataFrame.from_records(records)
    if 'text' in queries.columns: queries = queries.rename(columns={'text': 'query'})
    if 'query_id' in queries.columns: queries = queries.rename(columns={'query_id': 'qid'})
    queries['qid'] = queries['qid'].astype(str)
    queries_tok = pt.rewrite.tokenise()(queries)

    # Variant-Specific Output Directory (Optional, currently flat in test_run_files)
    # We will namespace filenames instead: test_stageX_{model}_{variant}.run

    # ---------------------------------------------------------
    # STAGE 1: SPARSE (BM25)
    # ---------------------------------------------------------
    print("\n>>> STAGE 1: BM25 (Sparse)")
    stage1_file = PATHS["run_dir"] / f"test_stage1_bm25_{selected_variant}.run"

    if stage1_file.exists():
        print(f"  [CHECKPOINT] Found: {stage1_file.name}")
        sparse_run = pt.io.read_results(str(stage1_file))
    else:
        bm25 = pt.terrier.Retriever(index, wmodel="BM25", controls=BM25_PARAMS, verbose=True)
        sparse_run = bm25.transform(queries_tok)
        pt.io.write_results(sparse_run, str(stage1_file))
        print(f"  Retrieved {len(sparse_run)} docs.")

    # ---------------------------------------------------------
    # STAGE 2: DENSE (Ensemble)
    # ---------------------------------------------------------
    print("\n>>> STAGE 2: Dense Retrieval (Fusion)")
    stage2_file = PATHS["run_dir"] / f"test_stage2_dense_fusion_{selected_variant}.run"

    if stage2_file.exists():
        print(f"  [CHECKPOINT] Found: {stage2_file.name}")
        dense_fused = pt.io.read_results(str(stage2_file))
    else:
        # Filter Sparse
        sparse_run = sparse_run.sort_values(["qid", "score"], ascending=[True, False])
        sparse_run["rank"] = sparse_run.groupby("qid").cumcount() + 1
        candidates_dense = sparse_run[sparse_run["rank"] <= K_DENSE_INPUT].copy()

        text_pipe = pt.text.get_text(index, "text")
        candidates_text = text_pipe.transform(candidates_dense)
        candidates_text = candidates_text.merge(queries[['qid', 'query']], on='qid', how='left')

        dense_results = []
        for name, cfg in DENSE_MODELS.items():
            model_run_file = PATHS["run_dir"] / f"test_stage2_{name}_{selected_variant}.run"

            if model_run_file.exists():
                print(f"  [CHECKPOINT] Found run for {name}")
                df = pt.io.read_results(str(model_run_file))
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
                pt.io.write_results(df, str(model_run_file))
                
                del scorer
                torch.cuda.empty_cache()
                gc.collect()

            dense_results.append(df)

        print("  Fusing Dense Runs...")
        dense_fused = pandas_rrf(dense_results, k=60)
        pt.io.write_results(dense_fused, str(stage2_file))

    # ---------------------------------------------------------
    # STAGE 3: CROSS-ENCODER
    # ---------------------------------------------------------
    print("\n>>> STAGE 3: Cross-Encoder (Hybrid Pool)")
    stage3_file = PATHS["run_dir"] / f"test_stage3_ce_{selected_variant}.run"

    if stage3_file.exists():
        print(f"  [CHECKPOINT] Found: {stage3_file.name}")
        pool = pt.io.read_results(str(stage3_file))
        pool = pool.merge(queries[['qid', 'query']], on='qid')
        pool = pt.text.get_text(index, "text").transform(pool)
    else:
        # Hybrid Pool
        bm25_top = sparse_run[sparse_run["rank"] <= K_CE_POOL_SPARSE][['qid', 'docno']]
        dense_fused = dense_fused.sort_values(["qid", "score"], ascending=[True, False])
        dense_fused["rank"] = dense_fused.groupby("qid").cumcount() + 1
        dense_top = dense_fused[dense_fused["rank"] <= K_CE_POOL_DENSE][['qid', 'docno']]

        pool = pd.concat([bm25_top, dense_top]).drop_duplicates(subset=['qid', 'docno'])
        
        # Add text
        pool = pool.merge(queries[['qid', 'query']], on='qid')
        pool = pt.text.get_text(index, "text").transform(pool)

        # Score
        pairs = [f"Query: {r.query} Document: {r.text} Relevant:" for r in pool.itertuples()]
        ce_scorer = MonoT5Scorer(CE_MODEL, CE_LOCAL_PATH)
        ce_scores = ce_scorer.score(pairs)
        del ce_scorer
        torch.cuda.empty_cache()
        gc.collect()

        pool["score"] = ce_scores
        pool = pool.sort_values(["qid", "score"], ascending=[True, False])
        pool = pool.reset_index(drop=True)
        pool["rank"] = pool.groupby("qid").cumcount() + 1
        pool["Q0"] = "Q0"
        pool["system"] = "monoT5"

        final_pool_df = pool[["qid", "docno", "rank", "score", "Q0", "system"]]
        pt.io.write_results(final_pool_df, str(stage3_file))

    # ---------------------------------------------------------
    # STAGE 4: LLM RE-RANKING
    # ---------------------------------------------------------
    print("\n>>> STAGE 4: LLM Re-ranking")
    stage4_file = PATHS["run_dir"] / f"test_final_qwen_{selected_variant}.run"

    if stage4_file.exists():
        print(f"  [CHECKPOINT] Found: {stage4_file.name}")
        final_run = pt.io.read_results(str(stage4_file))
    else:
        torch.cuda.empty_cache()
        gc.collect()

        pool = pool.sort_values(["qid", "score"], ascending=[True, False])
        pool["rank"] = pool.groupby("qid").cumcount() + 1
        llm_candidates = pool[pool["rank"] <= K_LLM_INPUT].copy()

        llm = QwenRanker(LLM_MODEL, LLM_LOCAL_PATH)
        grouped = llm_candidates.groupby("qid")
        query_groups = []
        for qid, group in grouped:
            q_text = group.iloc[0]['query']
            docs = [(row.docno, row.text) for row in group.itertuples()]
            query_groups.append((qid, q_text, docs))

        print(f"  Re-ranking Top-{K_LLM_INPUT}...")
        final_rows = []
        for batch_start in tqdm(range(0, len(query_groups), LLM_VLLM_BATCH), desc="vLLM Batches"):
            batch = query_groups[batch_start:batch_start + LLM_VLLM_BATCH]
            batch_results = llm.rerank_batch(batch)
            for (qid, _, _), reranked in zip(batch, batch_results):
                for r in reranked:
                    final_rows.append({"qid": qid, "docno": r['docno'], "score": r['score']})

        llm_df = pd.DataFrame(final_rows)
        ce_tail = pool[pool["rank"] > K_LLM_INPUT][['qid', 'docno', 'score']]
        final_run = pd.concat([llm_df, ce_tail])
        
        final_run = final_run.sort_values(["qid", "score"], ascending=[True, False])
        final_run["rank"] = final_run.groupby("qid").cumcount() + 1
        final_run["Q0"] = "Q0"
        final_run["system"] = "Qwen_72B"
        pt.io.write_results(final_run, str(stage4_file))

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------
    print("\n>>> FINAL EVALUATION")
    qrels = pt.io.read_qrels(PATHS['test_qrels'])
    q_std = qrels.rename(columns={"qid":"query_id", "docno":"doc_id", "label":"relevance"})
    r_std = final_run.rename(columns={"qid":"query_id", "docno":"doc_id"})

    metrics = calc_aggregate(MEASURES, q_std, r_std)
    
    out_metric_file = PATHS["eval_dir"] / f"test_metrics_{selected_variant}.json"
    with open(out_metric_file, "w") as f:
        json.dump({str(k): v for k, v in metrics.items()}, f, indent=2)

    print("\nFinal Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nPipeline Finished. Metrics saved to {out_metric_file.name}")

if __name__ == "__main__":
    main()