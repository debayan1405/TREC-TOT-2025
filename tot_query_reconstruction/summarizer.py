import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from loguru import logger
from .model_loader import setup_model  # reuse the same setup_model function


def load_rewritten_queries(files):
    """Load rewritten queries from JSONL files into dict keyed by query_id."""
    all_queries = {}
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                qid = record["query_id"]
                query = record["query"]
                all_queries.setdefault(qid, []).append(query)
    return all_queries


def summarize_queries(env, model_cfg, input_files, output_path):
    """Summarize multiple rewrites into a single query per query_id."""
    logger.info(f"Loading summarizer model: {model_cfg['hf_id']}")
    tokenizer, model = setup_model(
        model_hf_id=model_cfg["hf_id"],
        quantization=model_cfg["quantization"],
        bnb_conf=model_cfg["bitsandbytes"],
        generation_conf=env["generation"]
    )

    summarizer = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    queries = load_rewritten_queries(input_files)
    logger.info(f"Loaded {len(queries)} query sets for summarization.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, (qid, variants) in enumerate(queries.items()):
            # Build summarization input
            prompt = (
                "You are a query summarizer for tip-of-the-tongue (TOT) information retrieval.\n"
                "You will receive three different rewritten queries for the same original input.\n\n"
                f"Query 1: {variants[0]}\n"
                f"Query 2: {variants[1]}\n"
                f"Query 3: {variants[2]}\n\n"
                "Summarized Query:"
            )

            resp = summarizer(
                prompt,
                max_new_tokens=env["generation"]["max_new_tokens"],
                temperature=env["generation"]["temperature"],
                top_p=env["generation"]["top_p"],
                do_sample=env["generation"]["do_sample"]
            )[0]["generated_text"]

            # Extract only the summarized query
            summarized_query = resp.split("Summarized Query:")[-1].strip()

            out_record = {"query_id": qid, "query": summarized_query}
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            if idx < 5:  # log first 5
                logger.info(f"[{qid}] → {summarized_query}")

    logger.info(f"Summarized queries written to {output_path}")
