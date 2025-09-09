# main.py
import json
import os
from pathlib import Path
import argparse
from rewriter import validate_env, rewrite_topic_set


def main(env_path: str, models_to_run=None):
    with open(env_path, "r", encoding="utf-8") as f:
        env = json.load(f)
    validate_env(env)

    paths = env["paths"]
    generation_conf = env.get("generation", {})
    preview_count = env.get("logging", {}).get("preview_count", 5)

    version_map = {
        "train": paths["train_topics_path"],
        "dev-1": paths["dev_1_topics_path"],
        "dev-2": paths["dev_2_topics_path"],
        "dev-3": paths["dev_3_topics_path"],
        "test": paths["test_topics_path"]
    }

    if models_to_run is None:
        models_to_run = list(env["models"].keys())

    for model_name in models_to_run:
        print(f"\n=== Running rewrites for model: {model_name} ===")
        model_cfg = env["models"][model_name]
        hf_id = model_cfg["hf_id"]
        bnb_conf = model_cfg.get("bitsandbytes", {})

        # Example: run for each dataset version (you can customize)
        for ver, topic_path in version_map.items():
            print(f"\n--- Dataset version: {ver} ---")
            output_dir = paths["rewritten_queries_directory"]
            # Create output dir if missing
            os.makedirs(output_dir, exist_ok=True)
            # call rewrite function
            rewrite_topic_set(
                model_name=model_name,
                model_hf_id=hf_id,
                bnb_conf=bnb_conf,
                topic_file=topic_path,
                output_dir=output_dir,
                generation_conf=generation_conf,
                preview_count=preview_count
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="env.json", help="Path to env.json")
    parser.add_argument("--models", nargs="*",
                        help="Optional subset of model keys to run (e.g. llama mistral)")
    args = parser.parse_args()
    main(args.env, args.models)
