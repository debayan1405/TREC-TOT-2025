# rewriter.py
import json
import os
from typing import List, Dict
from pathlib import Path
import sys
import logging
import itertools
import tqdm
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Topic loader
# -----------------------


def load_topics(path: str) -> List[Dict]:
    """
    Expects a JSONL file where each line is a JSON object containing at least:
      - query_id
      - query
    Returns a list of dicts.
    """
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
    logger.info("env.json validated.")

# -----------------------
# Model loader with 8-bit bnb config
# -----------------------


def setup_model(model_hf_id: str, quantization: str, bnb_conf: dict, generation_conf: dict):
    """
    Returns tokenizer, model.
    Supports both 4-bit and 8-bit quantization via bitsandbytes.
    """
    logger.info(
        f"Setting up model {model_hf_id} with {quantization} quantization.")

    model = None
    tokenizer = AutoTokenizer.from_pretrained(model_hf_id, use_fast=True)

    try:
        if quantization == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=bnb_conf.get(
                    "bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=getattr(torch, bnb_conf.get(
                    "bnb_4bit_compute_dtype", "float16")),
                bnb_4bit_quant_type=bnb_conf.get("bnb_4bit_quant_type", "nf4"),
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_hf_id,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True
            )

        elif quantization == "8bit":
            # For 8-bit we don’t need the full BitsAndBytesConfig, just load_in_8bit=True
            model = AutoModelForCausalLM.from_pretrained(
                model_hf_id,
                device_map="auto",
                load_in_8bit=bnb_conf.get("load_in_8bit", True),
                trust_remote_code=True
            )

        else:
            raise ValueError(f"Unsupported quantization type: {quantization}")

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
# Single-query rewrite function
# -----------------------


def rewrite_single(tokenizer, model, original_query: str, generation_conf: dict) -> str:
    prompt = f"{SYSTEM_PROMPT_RW}\n\n{USER_PROMPT_RW.format(QUERY=original_query)}"
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move to model device automatically (tokenizer outputs on cpu; model uses device_map="auto")
    # Using model.generate directly. Use generation params from generation_conf.
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
    # The model output will contain the whole prompt + generated portion on some HF setups; try extract after "Rewritten query:"
    if "Rewritten query:" in decoded:
        rewritten = decoded.split("Rewritten query:")[-1].strip()
    else:
        # Fallback: remove prompt prefix
        rewritten = decoded[len(tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True)):].strip()
    # Keep only first line/paragraph
    rewritten = rewritten.split("\n")[0].strip()
    return rewritten

# -----------------------
# Batch rewrite loop and save
# -----------------------


def rewrite_topic_set(model_name: str, model_hf_id: str, bnb_conf: dict, topic_file: str, output_dir: str, generation_conf: dict, preview_count: int = 5):
    import torch  # import here to avoid global dependency unless used

    tokenizer, model = setup_model(
        model_hf_id=model_cfg["hf_id"],
        quantization=model_cfg["quantization"],
        bnb_conf=model_cfg["bitsandbytes"],
        generation_conf=env["generation"]
    )

    topics = load_topics(topic_file)
    results = []
    logger.info(
        f"Beginning rewrite loop for {len(topics)} topics using model {model_name}")
    for t in tqdm.tqdm(topics):
        qid = t["query_id"]
        qtext = t["query"]
        try:
            rewritten = rewrite_single(
                tokenizer, model, qtext, generation_conf)
        except Exception as e:
            logger.exception(f"Rewrite failed for query_id {qid}: {e}")
            rewritten = ""  # safe fallback
        results.append({"query_id": qid, "query": rewritten})

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{model_name}_reconstruction.jsonl")
    with open(outpath, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(results)} rewrites to {outpath}")

    # log previews
    logger.info(
        "Preview of first %d rewrites (original -> rewrite):", preview_count)
    for i, (orig, rew) in enumerate(zip(topics[:preview_count], results[:preview_count])):
        print(f"\n--- Example {i+1} ---")
        print("ORIGINAL:")
        print(orig["query"])
        print("\nREWRITTEN:")
        print(rew["query"])

    # return list for programmatic access
    return results
