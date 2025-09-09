# summarizer.py
import json
from pathlib import Path
from transformers import pipeline
from loguru import logger
from rewriter import setup_model  # reuse the same setup_model from rewriter.py


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


def summarize_queries(env, model_cfg, input_files, output_path, hf_token: str = None):
    """Summarize multiple rewrites into a single query per query_id."""
    logger.info(f"Loading summarizer model: {model_cfg['hf_id']}")

    tokenizer, model = setup_model(
        model_hf_id=model_cfg["hf_id"],
        bnb_conf=model_cfg.get("bitsandbytes", {}),
        hf_token=hf_token
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
            # Handle cases with fewer than 3 variants
            padded_variants = variants + [""] * (3 - len(variants))

            prompt = (
                "You are a query summarizer for tip-of-the-tongue (TOT) information retrieval.\n"
                "You will receive up to three different rewritten queries for the same original input.\n\n"
                "Analyze each of the Queries given below and try to preserve the maximum, non-redundant information in the summarized version, which can be used as a substitute for the three separate queries without any information loss.\n\n"
                f"Query 1: {padded_variants[0]}\n"
                f"Query 2: {padded_variants[1]}\n"
                f"Query 3: {padded_variants[2]}\n\n"
                "Summarized Query:"
            )

            resp = summarizer(
                prompt,
                max_new_tokens=env["generation"]["max_new_tokens"],
                temperature=env["generation"]["temperature"],
                top_p=env["generation"]["top_p"],
                do_sample=env["generation"]["do_sample"]
            )[0]["generated_text"]

            summarized_query = resp.split("Summarized Query:")[-1].strip()
            out_record = {"query_id": qid, "query": summarized_query}
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            if idx < 5:  # log first 5
                logger.info(f"[{qid}] → {summarized_query}")

    logger.info(f"Summarized queries written to {output_path}")
