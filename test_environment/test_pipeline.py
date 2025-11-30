import pyterrier as pt
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
import os
import json
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
import ir_measures
from ir_measures import calc_aggregate, nDCG, R, P, RR, Success
from tqdm import tqdm
import gc

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
os.environ["JAVA_OPTS"] = "-Xmx100g -Xms16g -Dindex.meta.data-source=fileinmem"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not pt.java.started():
    pt.java.init()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()
print(f"Hardware: {NUM_GPUS} GPUs detected.")

# --- PARAMETERS ---
# Stage 1: Sparse
BM25_PARAMS = {'c': 0.01, 'bm25.b': 0.55}

# Stage 2: Dense
DENSE_MODELS = {
    "colbertv2.0": {"hf_id": "colbert-ir/colbertv2.0", "type": "colbert", "batch": 128},
    "contriever": {"hf_id": "facebook/contriever", "type": "dot", "batch": 256},
    "e5-large": {"hf_id": "intfloat/e5-large", "type": "e5", "batch": 128}
}
K_DENSE_INPUT = 1000 # Retrieve top 1000 from sparse

# Stage 3: Cross-Encoder
CE_MODEL = "castorini/monot5-large-msmarco-10k"
CE_BATCH = 64
K_CE_POOL_SPARSE = 100 # Take top 100 from BM25
K_CE_POOL_DENSE = 100  # Take top 100 from Dense Fusion
# Total pool approx 200

# Stage 4: LLM
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
K_LLM_INPUT = 30 # Input to LLM
LLM_CONTEXT_CHARS = 500 # Increased context window

MEASURES = [nDCG@10, nDCG@100, R@1000, P@10, RR, Success@10]

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
        print(f"Loading {config['hf_id']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config['hf_id'])
        self.model = AutoModel.from_pretrained(config['hf_id']).to(DEVICE).eval()
        if NUM_GPUS > 1: self.model = nn.DataParallel(self.model)

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def score(self, queries, docs):
        dataset = ReRankDataset(queries, docs, self.tokenizer, self.type)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)
        all_scores = []
        
        with torch.no_grad():
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
                all_scores.append(scores.cpu().numpy())
        return np.concatenate(all_scores)

class MonoT5Scorer:
    def __init__(self, model_name):
        print(f"Loading {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE).eval()
        if DEVICE=="cuda": self.model.half()
        if NUM_GPUS > 1: self.model = nn.DataParallel(self.model)
        self.true_tok = self.tokenizer.encode("true")[0]
        self.false_tok = self.tokenizer.encode("false")[0]

    def score(self, pairs_text):
        # pairs_text: list of "Query: ... Document: ... Relevant:"
        # Simple list batching manually since it's just text
        batch_size = CE_BATCH * max(1, NUM_GPUS)
        all_scores = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(pairs_text), batch_size), desc="MonoT5 Scoring", leave=False):
                batch = pairs_text[i:i+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
                
                # T5 decoder start
                dec_in = torch.full((len(batch), 1), self.model.module.config.decoder_start_token_id if hasattr(self.model, "module") else self.model.config.decoder_start_token_id, device=DEVICE)
                
                out = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, decoder_input_ids=dec_in)
                logits = out.logits[:, 0, :]
                true_log = logits[:, self.true_tok]
                false_log = logits[:, self.false_tok]
                scores = torch.exp(true_log) / (torch.exp(true_log) + torch.exp(false_log))
                all_scores.append(scores.float().cpu().numpy())
        return np.concatenate(all_scores)

class QwenRanker:
    def __init__(self, model_name):
        print(f"Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def rerank(self, query, docs_list):
        # docs_list: [(docid, text), ...]
        cands = ""
        id_map = {}
        for idx, (did, txt) in enumerate(docs_list):
            safe_text = txt[:LLM_CONTEXT_CHARS].replace("\n", " ") # 500 chars
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
        inputs = self.tokenizer([text_in], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            gen_ids = self.model.generate(inputs.input_ids, max_new_tokens=128, temperature=0.01, do_sample=False)
        
        resp = self.tokenizer.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        matches = re.findall(r"\[(\d+)\]", resp)
        reranked = []
        seen = set()
        score = 1000.0
        
        # Add ranked
        for mid in matches:
            try:
                mid = int(mid)
                if mid in id_map and mid not in seen:
                    reranked.append({'docno': id_map[mid], 'score': score})
                    score -= 1.0
                    seen.add(mid)
            except: continue
            
        # Add rest
        for idx, (did, _) in enumerate(docs_list):
            if idx not in seen:
                reranked.append({'docno': did, 'score': score})
                score -= 1.0
        return reranked

# ==========================================
# 3. UTILS
# ==========================================
def load_env(path="env.json"):
    if not os.path.exists(path): path = "/content/env.json"
    with open(path) as f: return json.load(f)

def pandas_rrf(runs, k=60):
    parts = []
    for r in runs:
        r = r.sort_values(["qid", "score"], ascending=[False, False]) # Sort desc score
        r["rank"] = r.groupby("qid").cumcount() + 1
        r["rrf"] = 1.0 / (k + r["rank"])
        parts.append(r[["qid", "docno", "rrf"]])
    
    combined = pd.concat(parts)
    fused = combined.groupby(["qid", "docno"], as_index=False)["rrf"].sum()
    fused = fused.rename(columns={"rrf": "score"})
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
    
    print(f"{'='*60}\nSTARTING ISOLATED TEST PIPELINE\n{'='*60}")
    
    # 1. Load Data
    print("Loading Index & Test Data...")
    index = pt.IndexFactory.of(env['paths']['index_path'])
    
    q_path = env['paths']['query_variations']['mistral']['test']
    print(f"Queries (Mistral): {q_path}")
    queries = pd.read_json(q_path, lines=True)
    if 'text' in queries.columns: queries = queries.rename(columns={'text': 'query'})
    if 'query_id' in queries.columns: queries = queries.rename(columns={'query_id': 'qid'})
    queries['qid'] = queries['qid'].astype(str)
    # Tokenize for Terrier
    queries_tok = pt.rewrite.tokenise()(queries)
    
    # ---------------------------------------------------------
    # STAGE 1: SPARSE (BM25)
    # ---------------------------------------------------------
    print("\n>>> STAGE 1: BM25 (Sparse)")
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls=BM25_PARAMS)
    sparse_run = bm25.transform(queries_tok)
    
    # Save
    pt.io.write_results(sparse_run, os.path.join(run_dir, "test_stage1_bm25.run"))
    print(f"  Retrieved {len(sparse_run)} docs.")

    # ---------------------------------------------------------
    # STAGE 2: DENSE (Ensemble)
    # ---------------------------------------------------------
    print("\n>>> STAGE 2: Dense Retrieval (Fusion)")
    
    # Filter Sparse to Top-1000 for input
    sparse_run = sparse_run.sort_values(["qid", "score"], ascending=[True, False])
    sparse_run["rank"] = sparse_run.groupby("qid").cumcount() + 1
    candidates_dense = sparse_run[sparse_run["rank"] <= K_DENSE_INPUT].copy()
    
    # Fetch Text Once
    text_pipe = pt.text.get_text(index, "text")
    candidates_text = text_pipe.transform(candidates_dense)
    
    dense_results = []
    
    for name, cfg in DENSE_MODELS.items():
        print(f"  Running {name}...")
        scorer = DenseScorer(cfg)
        scores = scorer.score(candidates_text["query"].tolist(), candidates_text["text"].tolist())
        
        df = candidates_text.copy()
        df["score"] = scores
        df["system"] = name
        dense_results.append(df)
        
        # Clear VRAM
        del scorer
        torch.cuda.empty_cache()
    
    # RRF Fusion
    print("  Fusing Dense Runs...")
    dense_fused = pandas_rrf(dense_results, k=60)
    pt.io.write_results(dense_fused, os.path.join(run_dir, "test_stage2_dense_fusion.run"))

    # ---------------------------------------------------------
    # STAGE 3: CROSS-ENCODER (Hybrid Pool)
    # ---------------------------------------------------------
    print("\n>>> STAGE 3: Cross-Encoder (Hybrid Pool)")
    
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
    # We need query text too
    pool = pool.merge(queries[['qid', 'query']], on='qid')
    pool = text_pipe.transform(pool)
    
    # Prepare text for MonoT5
    # "Query: ... Document: ... Relevant:"
    pairs = [f"Query: {r.query} Document: {r.text} Relevant:" for r in pool.itertuples()]
    
    # Score
    ce_scorer = MonoT5Scorer(CE_MODEL)
    ce_scores = ce_scorer.score(pairs)
    del ce_scorer
    torch.cuda.empty_cache()
    
    pool["score"] = ce_scores
    pt.io.write_results(pool, os.path.join(run_dir, "test_stage3_ce.run"))

    # ---------------------------------------------------------
    # STAGE 4: LLM RE-RANKING
    # ---------------------------------------------------------
    print("\n>>> STAGE 4: LLM Re-ranking (Qwen-72B)")
    
    # Filter CE results to Top-30 (Input for LLM)
    pool = pool.sort_values(["qid", "score"], ascending=[True, False])
    pool["rank"] = pool.groupby("qid").cumcount() + 1
    
    llm_candidates = pool[pool["rank"] <= K_LLM_INPUT].copy()
    
    # Initialize LLM
    llm = QwenRanker(LLM_MODEL)
    
    final_rows = []
    
    grouped = llm_candidates.groupby("qid")
    print(f"  Re-ranking Top-{K_LLM_INPUT}...")
    
    for qid, group in tqdm(grouped, total=len(grouped)):
        q_text = group.iloc[0]['query']
        docs = [(row.docno, row.text) for row in group.itertuples()]
        
        reranked = llm.rerank(q_text, docs)
        
        for r in reranked:
            final_rows.append({"qid": qid, "docno": r['docno'], "score": r['score']})
            
    # Merge with Tail (CE ranks 31+)
    # LLM scores are ~1000. CE scores are 0-1.
    llm_df = pd.DataFrame(final_rows)
    ce_tail = pool[pool["rank"] > K_LLM_INPUT][['qid', 'docno', 'score']]
    
    final_run = pd.concat([llm_df, ce_tail])
    pt.io.write_results(final_run, os.path.join(run_dir, "test_final_qwen.run"))
    
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