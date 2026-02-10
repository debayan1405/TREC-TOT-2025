# rewriter.py
import json
import os
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------
# Hardcoded Configurations
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded generation parameters
GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.0,
    "top_p": 0.95,
    "do_sample": False
}

# Logging settings
PREVIEW_COUNT = 5

# Prompts
SYSTEM_PROMPT_RW = (
    """
    You are a query rewriter for a search engine. Your task is to rewrite a complex, verbose, tip-of-the-tongue description into a simple, keyword-focused search query.
    Guidelines:
    1. Identify the core entity type if implied (e.g., could be a movie, book, song, product, place, person, software tool, concept, etc. - the domain is open-ended). Perform cautious entity expansion by adding closely related aliases or canonical forms only if they are directly supported by the input text as well as by incorporating domain specific vocabulary (e.g., movies: "cinematography'', "anthology film''; books: "epistolary novel'', "bildungsroman''; science: "biochemical pathway'', "quantum phenomenon''; products: "form factor'', "backwards compatibility''). Never guess, invent, or infer entities that are not explicitly mentioned or unambiguously implied. If unsure, do not expand.

    2. Extract key details in a domain-agnostic way: concrete attributes such as events, functions, features, relationships, names, dates, locations, behaviors, or other unique identifiers explicitly stated in the input (e.g., plot points for media, specifications for products, symptoms for medical queries, APIs for software, etc.). Do not add new facts.

    3. Remove conversational filler ("I think it was...", "It might be...", "I remember seeing...").

    4. Remove negative constraints or uncertainty unless crucial ("Not sure if...").

    5. Strict grounding rule: Do NOT introduce any information, entities, attributes, or assumptions that are not present in the input query. Every token in the rewritten query must be traceable to the original text or to safe lexical transformations (e.g., synonyms). No external knowledge.

    6. Formulate a concise query that a standard search engine (like Google or BM25) would understand. Output ONLY the rewritten query text. Do not output any explanations.
    """
)
USER_PROMPT_RW = "Original query: {QUERY}\nRewritten query:"

SYSTEM_PROMPT_SUMMARIZE = (
    """
    You are a query summarizer for tip-of-the-tongue (TOT) information retrieval.
    You will receive up to three different rewritten queries for the same original input.
    Analyze each of the Queries given below and try to preserve the maximum, non-redundant information in the summarized version, which can be used as a substitute for the three separate queries without any information loss.
    """
)


# -----------------------
# Utility Functions
# -----------------------
def find_env_path():
    """Find env.json file in current or parent directories."""
    current_dir = Path(__file__).parent
    checks = [
        current_dir / "env.json",
        current_dir.parent / "env.json",
        current_dir.parent.parent / "env.json"
    ]
    for path in checks:
        if path.exists():
            return str(path)
    raise FileNotFoundError("env.json not found in hierarchy.")

def validate_env(env: dict):
    """Validate the cleaned environment structure."""
    if "llm-models" not in env:
        raise ValueError("env.json must contain 'llm-models'")
    if "paths" not in env:
        raise ValueError("env.json must contain 'paths'")
    
    required_paths = ["rewritten_queries_directory", "query_variations", "qrels"]
    for p in required_paths:
        if p not in env["paths"]:
            raise ValueError(f"env.json.paths must contain key '{p}'")

    if "original" not in env["paths"]["query_variations"]:
        raise ValueError("env.json.paths.query_variations must contain 'original'")

    logger.info("Environment configuration validated.")

def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def save_jsonl(data: List[Dict], path: str):
    """Save list of dicts to JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# -----------------------
# Model Setup
# -----------------------
def setup_model(model_hf_id: str, hf_token: str = None):
    """
    Setup model without quantization (Full Precision/BFloat16).
    """
    logger.info(f"Loading model {model_hf_id} in bfloat16...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_hf_id,
            use_fast=True,
            token=hf_token,
            trust_remote_code=True,
            padding_side='left'
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_hf_id,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            torch_dtype=torch.bfloat16, # Max precision for A6000s
            low_cpu_mem_usage=True
        )
        model.eval()
        
        # Optional compile
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except:
            pass
            
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model {model_hf_id}: {e}")
        raise


# -----------------------
# Generation Logic
# -----------------------
def run_batch_generation(tokenizer, model, prompts: List[str], batch_size: int = 8) -> List[str]:
    """Generic batch generation function."""
    results = []
    
    # Update config for generation call
    gen_kwargs = {
        "max_new_tokens": GENERATION_CONFIG["max_new_tokens"],
        "do_sample": GENERATION_CONFIG["do_sample"],
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    
    # Only add sampling params if sampling is enabled
    if GENERATION_CONFIG["do_sample"]:
        gen_kwargs["temperature"] = GENERATION_CONFIG["temperature"]
        gen_kwargs["top_p"] = GENERATION_CONFIG["top_p"]

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        try:
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )
            
            for j, output in enumerate(outputs):
                input_len = len(inputs["input_ids"][j])
                generated_tokens = output[input_len:]
                text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                # Cleanup output
                text = text.strip().split("\n")[0].strip()
                results.append(text if text else "")
                
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            results.extend([""] * len(batch_prompts))
            
    return results


# -----------------------
# Task: Rewriting
# -----------------------
def run_rewriting(env, models_to_run, datasets_to_run, hf_token):
    """
    Execute rewriting task.
    """
    output_dir = env["paths"]["rewritten_queries_directory"]
    
    for model_name in models_to_run:
        logger.info(f"=== Starting Rewriting with Model: {model_name} ===")
        
        model_cfg = env["llm-models"][model_name]
        tokenizer, model = setup_model(model_cfg["hf_id"], hf_token)

        for dataset in datasets_to_run:
            logger.info(f"Processing dataset: {dataset}")
            
            # Check Output Existence
            output_filename = f"{model_name}_{dataset}_rewritten_queries.jsonl"
            outpath = os.path.join(output_dir, output_filename)
            
            if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                logger.info(f"Output already exists for {model_name}/{dataset}. Skipping.")
                continue

            # Load Topics
            topic_path = env["paths"]["query_variations"]["original"].get(dataset)
            if not topic_path:
                logger.error(f"No original path found for dataset {dataset}")
                continue
                
            topics = load_jsonl(topic_path)
            # Normalize ID field
            for t in topics:
                if "q_id" in t: t["query_id"] = str(t.pop("q_id"))
                if "query_id" in t: t["query_id"] = str(t["query_id"])

            # Prepare Prompts
            prompts = [
                f"{SYSTEM_PROMPT_RW}\n\n{USER_PROMPT_RW.format(QUERY=t['query'])}" 
                for t in topics
            ]
            
            # Run Inference
            rewrites = []
            logger.info(f"Rewriting {len(prompts)} queries...")
            for i in tqdm(range(0, len(prompts), 16), desc=f"{model_name}-{dataset}"):
                batch_p = prompts[i:i+16]
                batch_r = run_batch_generation(tokenizer, model, batch_p, batch_size=16)
                rewrites.extend(batch_r)

            # Save Results
            results = []
            for t, rw in zip(topics, rewrites):
                results.append({
                    "query_id": t["query_id"],
                    "query": rw if rw else t["query"] # Fallback to original if empty
                })
            
            save_jsonl(results, outpath)
            logger.info(f"Saved to {outpath}")
            
            # Preview
            print(f"\nPreview ({model_name} - {dataset}):")
            for r in results[:PREVIEW_COUNT]:
                print(f"ID: {r['query_id']} | RW: {r['query']}")

        # Cleanup model to free VRAM for next model
        del model, tokenizer
        torch.cuda.empty_cache()


# -----------------------
# Task: Summarization
# -----------------------
def run_summarization(env, summarizer_model_key, datasets_to_run, hf_token):
    """
    Execute summarization task.
    """
    rewritten_dir = Path(env["paths"]["rewritten_queries_directory"])
    if summarizer_model_key not in env["llm-models"]:
        raise ValueError(f"Summarizer model '{summarizer_model_key}' not found in env.json models.")

    model_hf_id = env["llm-models"][summarizer_model_key]["hf_id"]
    tokenizer, model = setup_model(model_hf_id, hf_token)
    
    logger.info(f"=== Starting Summarization using {summarizer_model_key} ===")

    for dataset in datasets_to_run:
        # Check Output Existence
        # Output format: {dataset}_summarized_{model_name}.jsonl
        output_filename = f"{dataset}_summarized_{summarizer_model_key}.jsonl"
        output_path = rewritten_dir / output_filename
        
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Summary file exists for {dataset}. Skipping.")
            continue

        # Regex/Glob approach to find input files
        # Looking for {model}_{dataset}_rewritten_queries.jsonl
        # Exclude existing summary files to avoid recursion
        pattern = f"*_{dataset}_rewritten_queries.jsonl"
        found_files = list(rewritten_dir.glob(pattern))
        
        if not found_files:
            logger.warning(f"No rewritten files found for dataset '{dataset}' matching pattern '{pattern}'")
            continue
            
        logger.info(f"Found {len(found_files)} source files for {dataset}: {[f.name for f in found_files]}")
        
        # Load and Group Queries
        grouped_queries = {} # query_id -> [list of rewrites]
        
        for fpath in found_files:
            data = load_jsonl(str(fpath))
            for item in data:
                qid = str(item["query_id"])
                if qid not in grouped_queries:
                    grouped_queries[qid] = []
                grouped_queries[qid].append(item["query"])

        # Prepare Prompts
        query_ids = list(grouped_queries.keys())
        prompts = []
        
        for qid in query_ids:
            variants = grouped_queries[qid]
            # Pad or trim to exactly 3 for prompt consistency if needed, 
            # though prompt handles "up to three".
            # Just listing them:
            prompt_text = SYSTEM_PROMPT_SUMMARIZE
            for idx, v in enumerate(variants[:3]):
                prompt_text += f"Query {idx+1}: {v}\n"
            prompt_text += "\nSummarized Query:"
            prompts.append(prompt_text)

        # Run Inference
        logger.info(f"Summarizing {len(prompts)} query sets...")
        summaries = []
        for i in tqdm(range(0, len(prompts), 16), desc=f"Summ-{dataset}"):
            batch_p = prompts[i:i+16]
            batch_r = run_batch_generation(tokenizer, model, batch_p, batch_size=16)
            summaries.extend(batch_r)

        # Save Results
        results = []
        for qid, summ, original_variants in zip(query_ids, summaries, grouped_queries.values()):
            final_query = summ if summ else original_variants[0] # Fallback
            results.append({
                "query_id": qid,
                "query": final_query
            })

        save_jsonl(results, str(output_path))
        logger.info(f"Saved summaries to {output_path}")

        # Preview
        print(f"\nPreview (Summary - {dataset}):")
        for r in results[:PREVIEW_COUNT]:
            print(f"ID: {r['query_id']} | Summ: {r['query']}")

    del model, tokenizer
    torch.cuda.empty_cache()


# -----------------------
# Main Entry Point
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="LLM Query Rewriter & Summarizer")
    
    # Task Selection
    parser.add_argument("--task", choices=["rewrite", "summarize"], required=True, 
                        help="Choose whether to rewrite original queries or summarize existing rewrites.")
    
    # Configuration
    parser.add_argument("--env", default=None, help="Path to env.json")
    
    # Scope Selection
    parser.add_argument("--datasets", nargs="*", 
                        choices=["train", "dev-1", "dev-2", "dev-3", "test"],
                        help="Specific datasets to process (default: all)")
    
    # Model Selection
    parser.add_argument("--models", nargs="*", 
                        help="For rewriting: specific models to run (default: all in env.json).")
    
    parser.add_argument("--summarizer", 
                        help="For summarization: which model key from env.json to use as the summarizer.")

    args = parser.parse_args()

    # Load Env
    env_path = args.env if args.env else find_env_path()
    logger.info(f"Using environment: {env_path}")
    with open(env_path) as f:
        env = json.load(f)
    validate_env(env)
    
    hf_token = env.get("hf_token") or None
    
    # Determine Datasets
    available_datasets = list(env["paths"]["query_variations"]["original"].keys())
    if args.datasets:
        datasets_to_run = args.datasets
    else:
        datasets_to_run = available_datasets

    # Execute Logic
    if args.task == "rewrite":
        # Determine Models
        available_models = list(env["llm-models"].keys())
        if args.models:
            models_to_run = [m for m in args.models if m in available_models]
            if len(models_to_run) != len(args.models):
                logger.warning("Some requested models were not found in env.json")
        else:
            models_to_run = available_models
            
        run_rewriting(env, models_to_run, datasets_to_run, hf_token)

    elif args.task == "summarize":
        if not args.summarizer:
            raise ValueError("--summarizer argument is required when task is 'summarize'")
        
        run_summarization(env, args.summarizer, datasets_to_run, hf_token)

if __name__ == "__main__":
    main()