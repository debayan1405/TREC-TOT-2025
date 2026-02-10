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
# Configuration & Constants
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Hardcoded Paths (Resolved relative to script location for safety)
# using ../ notation as requested, but wrapping in resolve()
PATHS = {
    "rewritten_output_dir": (SCRIPT_DIR / "../rewritten-queries").resolve(),
    "original_queries": {
        "train": (SCRIPT_DIR / "../original_queries/train-original.jsonl").resolve(),
        "dev1": (SCRIPT_DIR / "../original_queries/dev1-original.jsonl").resolve(),
        "dev2": (SCRIPT_DIR / "../original_queries/dev2-original.jsonl").resolve(),
        "dev3": (SCRIPT_DIR / "../original_queries/dev3-original.jsonl").resolve(),
        "test": (SCRIPT_DIR / "../original_queries/test-original.jsonl").resolve()
    }
}

# Generation Config
GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.0,
    "top_p": 0.95,
    "do_sample": False
}

PREVIEW_COUNT = 5

# -----------------------
# Prompts
# -----------------------
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
def load_llm_config() -> Dict:
    """Load LLM configurations from local JSON file."""
    config_path = SCRIPT_DIR / "llm-config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"llm-config.json not found at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file safely."""
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return []
    return data

def save_jsonl(data: List[Dict], path: Path):
    """Save list of dicts to JSONL."""
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# -----------------------
# Model Setup
# -----------------------
def setup_model(model_hf_id: str):
    """
    Setup model without quantization (Full Precision/BFloat16).
    """
    logger.info(f"Loading model {model_hf_id} in bfloat16...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_hf_id,
            use_fast=True,
            trust_remote_code=True,
            padding_side='left'
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_hf_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, # Max precision for A6000s
            low_cpu_mem_usage=True
        )
        model.eval()
        
        # Optional compile for speedup
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
    
    gen_kwargs = {
        "max_new_tokens": GENERATION_CONFIG["max_new_tokens"],
        "do_sample": GENERATION_CONFIG["do_sample"],
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    
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
                # Cleanup output: take first line, strip whitespace
                text = text.strip().split("\n")[0].strip()
                results.append(text if text else "")
                
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            results.extend([""] * len(batch_prompts))
            
    return results


# -----------------------
# Task: Rewriting
# -----------------------
def run_rewriting(llm_config, models_to_run, datasets_to_run, batch_size):
    """Execute rewriting task."""
    output_dir = PATHS["rewritten_output_dir"]
    
    for model_name in models_to_run:
        logger.info(f"=== Starting Rewriting with Model: {model_name} ===")
        
        model_cfg = llm_config[model_name]
        tokenizer, model = setup_model(model_cfg["hf_id"])

        for dataset in datasets_to_run:
            logger.info(f"Processing dataset: {dataset}")
            
            # Check Output Existence
            output_filename = f"{model_name}_{dataset}_rewritten_queries.jsonl"
            outpath = output_dir / output_filename
            
            if outpath.exists() and outpath.stat().st_size > 0:
                logger.info(f"Output already exists for {model_name}/{dataset}. Skipping.")
                continue

            # Load Topics
            topic_path = PATHS["original_queries"].get(dataset)
            if not topic_path:
                logger.error(f"No original path configured for dataset '{dataset}'")
                continue
                
            topics = load_jsonl(topic_path)
            if not topics:
                logger.warning(f"No topics loaded from {topic_path}")
                continue

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
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name}-{dataset}"):
                batch_p = prompts[i:i+batch_size]
                batch_r = run_batch_generation(tokenizer, model, batch_p, batch_size=batch_size)
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
def run_summarization(llm_config, summarizer_model_key, datasets_to_run, batch_size):
    """Execute summarization task."""
    rewritten_dir = PATHS["rewritten_output_dir"]
    
    if summarizer_model_key not in llm_config:
        raise ValueError(f"Summarizer model '{summarizer_model_key}' not found in llm-config.json")

    model_hf_id = llm_config[summarizer_model_key]["hf_id"]
    tokenizer, model = setup_model(model_hf_id)
    
    logger.info(f"=== Starting Summarization using {summarizer_model_key} ===")

    for dataset in datasets_to_run:
        # Check Output Existence
        output_filename = f"{dataset}_summarized_{summarizer_model_key}.jsonl"
        output_path = rewritten_dir / output_filename
        
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Summary file exists for {dataset}. Skipping.")
            continue

        # Regex/Glob approach to find input files
        # Pattern: {model}_{dataset}_rewritten_queries.jsonl
        # Must exclude existing summary files to avoid recursion
        pattern = f"*_{dataset}_rewritten_queries.jsonl"
        found_files = list(rewritten_dir.glob(pattern))
        
        # Filter out self-reference or other summary files just in case
        source_files = [f for f in found_files if "summarized" not in f.name]
        
        if not source_files:
            logger.warning(f"No rewritten source files found for dataset '{dataset}' matching '{pattern}'")
            continue
            
        logger.info(f"Found {len(source_files)} source files for {dataset}: {[f.name for f in source_files]}")
        
        # Load and Group Queries
        grouped_queries = {} # query_id -> [list of rewrites]
        
        for fpath in source_files:
            data = load_jsonl(fpath)
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
            # Construct prompt with up to 3 variants
            prompt_text = SYSTEM_PROMPT_SUMMARIZE + "\n\n"
            for idx, v in enumerate(variants[:3]):
                prompt_text += f"Query {idx+1}: {v}\n"
            prompt_text += "\nSummarized Query:"
            prompts.append(prompt_text)

        # Run Inference
        logger.info(f"Summarizing {len(prompts)} query sets...")
        summaries = []
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Summ-{dataset}"):
            batch_p = prompts[i:i+batch_size]
            batch_r = run_batch_generation(tokenizer, model, batch_p, batch_size=batch_size)
            summaries.extend(batch_r)

        # Save Results
        results = []
        for qid, summ, original_variants in zip(query_ids, summaries, grouped_queries.values()):
            final_query = summ if summ else original_variants[0] # Fallback
            results.append({
                "query_id": qid,
                "query": final_query
            })

        save_jsonl(results, output_path)
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
    
    # Scope Selection
    parser.add_argument("--datasets", nargs="*", 
                        choices=["train", "dev-1", "dev-2", "dev-3", "test"],
                        help="Specific datasets to process (default: all)")
    
    # Model Selection
    parser.add_argument("--models", nargs="*", 
                        help="For rewriting: specific models to run (default: all available).")
    
    parser.add_argument("--summarizer", 
                        help="For summarization: which model key from llm-config.json to use.")

    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for generation (default: 8).")

    args = parser.parse_args()

    # Load Config
    llm_config = load_llm_config()
    available_models = list(llm_config.keys())
    logger.info(f"Loaded LLM configuration. Available models: {available_models}")
    
    # Map CLI dataset names to PATHS keys
    dataset_map = {
        "train": "train",
        "dev-1": "dev1",
        "dev-2": "dev2",
        "dev-3": "dev3",
        "test": "test"
    }
    
    if args.datasets:
        datasets_to_run = [dataset_map[d] for d in args.datasets if d in dataset_map]
    else:
        datasets_to_run = list(dataset_map.values())

    # Execute Logic
    if args.task == "rewrite":
        if args.models:
            models_to_run = [m for m in args.models if m in available_models]
            invalid_models = [m for m in args.models if m not in available_models]
            if invalid_models:
                logger.warning(f"Skipping invalid models requested: {invalid_models}")
                logger.warning(f"Available models: {available_models}")
            if not models_to_run:
                raise ValueError("No valid models selected for rewriting.")
        else:
            models_to_run = available_models
            
        run_rewriting(llm_config, models_to_run, datasets_to_run, args.batch_size)

    elif args.task == "summarize":
        if not args.summarizer:
            raise ValueError("--summarizer argument is required when task is 'summarize'")
        
        run_summarization(llm_config, args.summarizer, datasets_to_run, args.batch_size)

if __name__ == "__main__":
    main()