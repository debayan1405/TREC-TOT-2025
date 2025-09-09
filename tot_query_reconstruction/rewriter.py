# rewriter.py
import json
import os
from typing import List, Dict
import logging
import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------
# Topic loader
# -----------------------
def load_topics(path: str) -> List[Dict]:
    topics = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in topics file {path}: {e}")
                raise
            if "query_id" not in obj or "query" not in obj:
                raise ValueError(
                    f"Each topic must contain 'query_id' and 'query'. Offending line: {line}")
            topics.append(obj)
    logger.info(f"Loaded {len(topics)} topics from {path}")
    return topics


# -----------------------
# Env validation
# -----------------------
def validate_env(env: dict):
    required_paths = [
        "train_topics_path", "dev_1_topics_path", "dev_2_topics_path",
        "dev_3_topics_path", "test_topics_path", "rewritten_queries_directory"
    ]
    if "models" not in env:
        raise ValueError("env.json must contain top-level 'models' key.")
    for p in required_paths:
        if p not in env["paths"]:
            raise ValueError(f"env.json.paths must contain key '{p}'")
    if "hf_token" not in env:
        raise ValueError(
            "env.json must contain top-level 'hf_token' (can be empty string).")
    logger.info("env.json validated.")


# -----------------------
# Model loader with 8-bit
# -----------------------
def setup_model(model_hf_id: str, bnb_conf: dict, hf_token: str = None):
    """
    Returns tokenizer, model.
    Supports ONLY 8-bit quantization via bitsandbytes.
    Uses HF token if provided (needed for gated models like Llama 3.1).
    """
    logger.info(f"Setting up model {model_hf_id} with 8-bit quantization.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_hf_id,
            use_fast=True,
            token=hf_token if hf_token else None
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_hf_id,
            device_map="auto",
            load_in_8bit=bnb_conf.get("load_in_8bit", True),
            trust_remote_code=True,
            token=hf_token if hf_token else None
        )

    except Exception as e:
        logger.exception(f"Failed to load model {model_hf_id}. Exception: {e}")
        raise

    logger.info("Model loaded successfully.")
    return tokenizer, model


# -----------------------
# Prompt templates
# -----------------------
SYSTEM_PROMPT_RW = (
    "You are an expert in information retrieval query rewriting. \n"
    "Your task is to take a verbose, artifact-rich query where the user "
    "is trying to describe an item they cannot name, and rewrite it "
    "into a clearer, concise, search-ready query.\n\n"
    "Rules:\n"
    "- Use ONLY information explicitly present in the input query.\n"
    "- DO NOT add any facts, assumptions, or invented entities.\n"
    "- DO NOT output commentary, confirmations, or step-by-step text.\n"
    "- Output ONLY the rewritten query as a single paragraph of free text.\n"
    "- Preserve as many identifying clues as possible, while keeping the output compact.\n"
)

USER_PROMPT_RW = "Original query: {QUERY}\nRewritten query:"


# -----------------------
# Single-query rewrite
# -----------------------
def rewrite_single(tokenizer, model, original_query: str, generation_conf: dict) -> str:
    import torch

    prompt = f"{SYSTEM_PROMPT_RW}\n\n{USER_PROMPT_RW.format(QUERY=original_query)}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=generation_conf.get("max_new_tokens", 128),
        do_sample=generation_conf.get("do_sample", False),
        temperature=generation_conf.get("temperature", 0.0),
        top_p=generation_conf.get("top_p", 0.95),
        eos_token_id=tokenizer.eos_token_id
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Rewritten query:" in decoded:
        rewritten = decoded.split("Rewritten query:")[-1].strip()
    else:
        prompt_decoded = tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True)
        rewritten = decoded[len(prompt_decoded):].strip()

    rewritten = rewritten.split("\n")[0].strip()
    return rewritten


# -----------------------
# Batch rewrite loop
# -----------------------
def rewrite_topic_set(model_name: str, model_hf_id: str, bnb_conf: dict,
                      topic_file: str, output_dir: str, generation_conf: dict,
                      preview_count: int = 5, hf_token: str = None):

    tokenizer, model = setup_model(
        model_hf_id=model_hf_id,
        bnb_conf=bnb_conf,
        hf_token=hf_token
    )

    topics = load_topics(topic_file)
    results = []
    logger.info(
        f"Beginning rewrite loop for {len(topics)} topics using model {model_name}"
    )

    for t in tqdm.tqdm(topics):
        qid = t["query_id"]
        qtext = t["query"]
        try:
            rewritten = rewrite_single(
                tokenizer, model, qtext, generation_conf)
        except Exception as e:
            logger.exception(f"Rewrite failed for query_id {qid}: {e}")
            rewritten = ""
        results.append({"query_id": qid, "query": rewritten})

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{model_name}_reconstruction.jsonl")
    with open(outpath, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(results)} rewrites to {outpath}")

    logger.info(
        "Preview of first %d rewrites (original -> rewrite):", preview_count)
    for i, (orig, rew) in enumerate(zip(topics[:preview_count], results[:preview_count])):
        print(f"\n--- Example {i+1} ---")
        print("ORIGINAL:")
        print(orig["query"])
        print("\nREWRITTEN:")
        print(rew["query"])

    return results
