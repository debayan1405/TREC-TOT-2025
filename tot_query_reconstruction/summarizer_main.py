import json
from pathlib import Path
from loguru import logger
from summarizer import summarize_queries


def main():
    with open("env.json", "r", encoding="utf-8") as f:
        env = json.load(f)

    # Paths to rewritten queries (3 models)
    rewritten_dir = Path(env["rewritten_queries_directory"])
    input_files = [
        rewritten_dir / "llama_reconstruction.jsonl",
        rewritten_dir / "mistral_reconstruction.jsonl",
        rewritten_dir / "qwen_reconstruction.jsonl"
    ]

    summarizer_cfg = env["summarizer"]
    output_file = rewritten_dir / summarizer_cfg["output_filename"]

    summarize_queries(env, summarizer_cfg, input_files, output_file)


if __name__ == "__main__":
    main()
