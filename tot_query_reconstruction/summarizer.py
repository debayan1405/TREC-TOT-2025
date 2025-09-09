# summarizer.py
import json
from pathlib import Path
from typing import Dict, List
import logging
import torch
from tqdm import tqdm
from rewriter import setup_model  # reuse the same setup_model from rewriter.py

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_rewritten_queries(files: List[Path], dataset_version: str = "train") -> Dict[str, List[str]]:
    """
    Load rewritten queries from JSONL files into dict keyed by query_id.

    Args:
        files: List of file paths to load from
        dataset_version: Dataset version to filter files by

    Returns:
        Dict mapping query_id to list of rewritten queries
    """
    all_queries = {}
    files_loaded = 0

    for fpath in files:
        # Skip files that don't match the dataset version
        if dataset_version not in fpath.name:
            continue

        if fpath.exists():
            logger.info(f"Loading rewritten queries from: {fpath}")
            with open(fpath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        qid = record["query_id"]
                        query = record["query"]
                        all_queries.setdefault(qid, []).append(query)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(
                            f"Invalid record in {fpath} at line {line_num}: {e}")
                        continue
            files_loaded += 1
        else:
            logger.warning(f"File not found: {fpath}")

    logger.info(
        f"Loaded queries from {files_loaded} files for dataset {dataset_version}")
    logger.info(f"Total unique query IDs: {len(all_queries)}")

    return all_queries


def summarize_queries(env: dict, model_cfg: dict, rewritten_dir: Path,
                      dataset_version: str, hf_token: str = None) -> str:
    """
    Summarize multiple rewrites into a single query per query_id.

    Args:
        env: Environment configuration
        model_cfg: Summarizer model configuration
        rewritten_dir: Directory containing rewritten query files
        dataset_version: Dataset version to process
        hf_token: HuggingFace token

    Returns:
        str: Path to output summary file
    """
    logger.info(f"Loading summarizer model: {model_cfg['hf_id']}")

    tokenizer, model = setup_model(
        model_hf_id=model_cfg["hf_id"],
        bnb_conf=model_cfg.get("bitsandbytes", {}),
        hf_token=hf_token
    )

    # Find all rewritten query files for this dataset version
    pattern = f"*_{dataset_version}_rewritten_queries.jsonl"
    rewritten_files = list(rewritten_dir.glob(pattern))

    if not rewritten_files:
        raise FileNotFoundError(
            f"No rewritten query files found for dataset {dataset_version} in {rewritten_dir}")

    logger.info(f"Found rewritten files: {[f.name for f in rewritten_files]}")

    queries = load_rewritten_queries(rewritten_files, dataset_version)

    if not queries:
        raise ValueError(f"No queries loaded for dataset {dataset_version}")

    logger.info(f"Loaded {len(queries)} query sets for summarization.")

    # Create output file with dataset version in name
    output_filename = f"{dataset_version}_{model_cfg['output_filename']}"
    output_path = rewritten_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generation_conf = env.get("generation", {})

    try:
        with open(output_path, "w", encoding="utf-8") as fout:
            for idx, (qid, variants) in enumerate(tqdm(queries.items(), desc="Summarizing queries")):
                # Handle cases with fewer than 3 variants by padding with empty strings
                padded_variants = variants + [""] * (3 - len(variants))

                # Take only first 3 variants if more than 3 exist
                padded_variants = padded_variants[:3]

                prompt = (
                    "You are a query summarizer for tip-of-the-tongue (TOT) information retrieval.\n"
                    "You will receive up to three different rewritten queries for the same original input.\n\n"
                    "Analyze each of the Queries given below and try to preserve the maximum, non-redundant information in the summarized version, which can be used as a substitute for the three separate queries without any information loss.\n\n"
                    f"Query 1: {padded_variants[0]}\n"
                    f"Query 2: {padded_variants[1]}\n"
                    f"Query 3: {padded_variants[2]}\n\n"
                    "Summarized Query:"
                )

                try:
                    # Tokenize with proper handling
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
                    summarized_query = tokenizer.decode(
                        generated_tokens, skip_special_tokens=True).strip()

                    # Clean up output
                    if "\n" in summarized_query:
                        summarized_query = summarized_query.split("\n")[
                            0].strip()

                    # Fallback to first variant if summarization fails
                    if not summarized_query and padded_variants[0]:
                        summarized_query = padded_variants[0]
                        logger.warning(
                            f"Empty summarization for query_id {qid}, using first variant")

                except Exception as e:
                    logger.error(f"Error summarizing query_id {qid}: {e}")
                    # Fallback to first non-empty variant
                    summarized_query = next(
                        (v for v in padded_variants if v), "")

                out_record = {"query_id": qid, "query": summarized_query}
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

                # Log first 5 examples
                if idx < 5:
                    logger.info(f"[{qid}] → {summarized_query}")

        logger.info(f"Summarized queries written to {output_path}")

        # Clean up GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

        return str(output_path)

    except Exception as e:
        logger.error(f"Error in summarize_queries: {e}")
        # Clean up on error
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        raise


def summarize_all_datasets(env: dict, rewritten_dir: Path, datasets_to_run: List[str] = None,
                           hf_token: str = None) -> Dict[str, str]:
    """
    Summarize queries for all specified datasets.

    Args:
        env: Environment configuration
        rewritten_dir: Directory containing rewritten query files  
        datasets_to_run: List of dataset versions to process (all if None)
        hf_token: HuggingFace token

    Returns:
        Dict mapping dataset version to output file path
    """
    if datasets_to_run is None:
        # Auto-detect available datasets from rewritten files
        available_patterns = [
            "train", "dev-1", "dev-2", "dev-3", "test"
        ]
        datasets_to_run = []
        for pattern in available_patterns:
            if list(rewritten_dir.glob(f"*_{pattern}_rewritten_queries.jsonl")):
                datasets_to_run.append(pattern)

        if not datasets_to_run:
            raise ValueError(
                f"No rewritten query files found in {rewritten_dir}")

    summarizer_cfg = env["summarizer"]
    results = {}

    for dataset_version in datasets_to_run:
        logger.info(f"Processing dataset: {dataset_version}")
        try:
            output_path = summarize_queries(
                env=env,
                model_cfg=summarizer_cfg,
                rewritten_dir=rewritten_dir,
                dataset_version=dataset_version,
                hf_token=hf_token
            )
            results[dataset_version] = output_path
            logger.info(f"✓ Successfully summarized {dataset_version}")
        except Exception as e:
            logger.error(f"✗ Error summarizing {dataset_version}: {e}")
            continue

    return results
