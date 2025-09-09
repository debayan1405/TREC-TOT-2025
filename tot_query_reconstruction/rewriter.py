# rewriter.py
import json
import os
from typing import List, Dict
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------
# Topic loader - Fixed field mapping
# -----------------------
def load_topics(path: str) -> List[Dict]:
    """Load topics from JSONL file, handling both q_id and query_id fields."""
    topics = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Invalid JSON in topics file {path} at line {line_num}: {e}")
                raise

            # Handle both q_id and query_id field names
            if "q_id" in obj:
                query_id = obj["q_id"]
            elif "query_id" in obj:
                query_id = obj["query_id"]
            else:
                raise ValueError(
                    f"Each topic must contain 'q_id' or 'query_id'. Missing in line {line_num}: {line}")

            if "query" not in obj:
                raise ValueError(
                    f"Each topic must contain 'query'. Missing in line {line_num}: {line}")

            # Standardize to query_id
            topics.append({
                "query_id": str(query_id),
                "query": obj["query"]
            })

    logger.info(f"Loaded {len(topics)} topics from {path}")
    return topics


# -----------------------
# Env validation - Fixed for new structure
# -----------------------
def validate_env(env: dict):
    """Validate environment configuration structure."""
    if "models" not in env:
        raise ValueError("env.json must contain top-level 'models' key.")

    if "paths" not in env:
        raise ValueError("env.json must contain top-level 'paths' key.")

    required_paths = [
        "train_topics_path", "dev_1_topics_path", "dev_2_topics_path",
        "dev_3_topics_path", "test_topics_path", "rewritten_queries_directory"
    ]

    for p in required_paths:
        if p not in env["paths"]:
            raise ValueError(f"env.json.paths must contain key '{p}'")

    if "hf_token" not in env:
        raise ValueError(
            "env.json must contain top-level 'hf_token' (can be empty string).")

    # Validate model configurations
    for model_name, model_config in env["models"].items():
        if "hf_id" not in model_config:
            raise ValueError(f"Model '{model_name}' must have 'hf_id' key")
        if "bitsandbytes" not in model_config:
            raise ValueError(
                f"Model '{model_name}' must have 'bitsandbytes' config")

    logger.info("env.json validated successfully.")


# -----------------------
# Model loader with updated HuggingFace practices
# -----------------------
def setup_model(model_hf_id: str, bnb_conf: dict, hf_token: str = None):
    """
    Setup model with 8-bit quantization using modern HuggingFace practices.

    Args:
        model_hf_id: HuggingFace model identifier
        bnb_conf: BitsAndBytesConfig dictionary
        hf_token: HuggingFace token for gated models

    Returns:
        tuple: (tokenizer, model)
    """
    logger.info(f"Setting up model {model_hf_id} with 8-bit quantization.")

    try:
        # Setup tokenizer with proper token handling
        tokenizer = AutoTokenizer.from_pretrained(
            model_hf_id,
            use_fast=True,
            token=hf_token if hf_token else None,
            trust_remote_code=True
        )

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Setup quantization config using BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=bnb_conf.get("load_in_8bit", True),
            load_in_4bit=False,  # Explicitly disable 4-bit when using 8-bit
        )

        # Load model with proper configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_hf_id,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
            token=hf_token if hf_token else None,
            torch_dtype=torch.float16,  # Use fp16 for better performance with quantization
        )

        # Ensure model is in eval mode
        model.eval()

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
# Single-query rewrite with better error handling
# -----------------------
def rewrite_single(tokenizer, model, original_query: str, generation_conf: dict) -> str:
    """
    Rewrite a single query using the loaded model.

    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        original_query: Original query text
        generation_conf: Generation configuration dictionary

    Returns:
        str: Rewritten query
    """
    try:
        prompt = f"{SYSTEM_PROMPT_RW}\n\n{USER_PROMPT_RW.format(QUERY=original_query)}"

        # Tokenize with proper attention mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": generation_conf.get("max_new_tokens", 128),
            "do_sample": generation_conf.get("do_sample", False),
            "temperature": generation_conf.get("temperature", 0.0),
            "top_p": generation_conf.get("top_p", 0.95),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }

        # Add sampling parameters only if do_sample is True
        if not gen_kwargs["do_sample"]:
            gen_kwargs.pop("temperature")
            gen_kwargs.pop("top_p")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )

        # Decode only the generated part
        generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
        rewritten = tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        # Clean up the output
        rewritten = rewritten.strip()
        if "\n" in rewritten:
            rewritten = rewritten.split("\n")[0].strip()

        return rewritten

    except Exception as e:
        logger.error(f"Error in rewrite_single: {e}")
        return ""


# -----------------------
# Batch rewrite loop with dynamic naming
# -----------------------
def rewrite_topic_set(model_name: str, model_hf_id: str, bnb_conf: dict,
                      topic_file: str, output_dir: str, generation_conf: dict,
                      dataset_version: str, preview_count: int = 5,
                      hf_token: str = None):
    """
    Rewrite all topics in a topic set using dynamic file naming.

    Args:
        model_name: Name of the model (for file naming)
        model_hf_id: HuggingFace model ID
        bnb_conf: BitsAndBytes configuration
        topic_file: Path to input topic file
        output_dir: Output directory
        generation_conf: Generation configuration
        dataset_version: Dataset version (train, dev-1, etc.)
        preview_count: Number of examples to preview
        hf_token: HuggingFace token

    Returns:
        list: List of rewritten topics
    """
    tokenizer, model = setup_model(
        model_hf_id=model_hf_id,
        bnb_conf=bnb_conf,
        hf_token=hf_token
    )

    topics = load_topics(topic_file)
    results = []
    logger.info(
        f"Beginning rewrite loop for {len(topics)} topics using model {model_name} "
        f"on dataset {dataset_version}"
    )

    for t in tqdm(topics, desc=f"Rewriting with {model_name}"):
        qid = t["query_id"]
        qtext = t["query"]
        try:
            rewritten = rewrite_single(
                tokenizer, model, qtext, generation_conf)
            if not rewritten:  # Fallback to original if rewrite fails
                logger.warning(
                    f"Empty rewrite for query_id {qid}, using original")
                rewritten = qtext
        except Exception as e:
            logger.exception(f"Rewrite failed for query_id {qid}: {e}")
            rewritten = qtext  # Fallback to original query

        results.append({"query_id": qid, "query": rewritten})

    # Dynamic file naming: model_name_dataset_version_rewritten_queries.jsonl
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(
        output_dir, f"{model_name}_{dataset_version}_rewritten_queries.jsonl")

    with open(outpath, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(results)} rewrites to {outpath}")

    # Preview results
    logger.info(
        f"Preview of first {preview_count} rewrites (original -> rewrite):")
    for i, (orig, rew) in enumerate(zip(topics[:preview_count], results[:preview_count])):
        print(f"\n--- Example {i+1} ---")
        print("ORIGINAL:")
        print(orig["query"])
        print("\nREWRITTEN:")
        print(rew["query"])

    # Clean up GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return results
