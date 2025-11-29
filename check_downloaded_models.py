import os
import subprocess
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_DIR = "/media/12TB/shared/models"

models_to_check = {
    "ColBERT": {"path": os.path.join(MODEL_DIR, "colbertv2.0"), "type": "encoder"},
    "Contriever": {"path": os.path.join(MODEL_DIR, "contriever"), "type": "encoder"},
    "E5-large": {"path": os.path.join(MODEL_DIR, "e5-large"), "type": "encoder"},
    "MonoT5-large": {"path": os.path.join(MODEL_DIR, "monot5-large-msmarco"), "type": "t5"},
    "Llama3-70B-Instruct": {"path": os.path.join(MODEL_DIR, "llama3-70b-instruct"), "type": "llama"},
    "Qwen2.5-72B-AWQ": {"path": os.path.join(MODEL_DIR, "qwen2.5-72b-awq"), "type": "qwen_awq"},
}


def check_and_pull_lfs(folder):
    """
    Checks if Git LFS files are present; pulls missing ones automatically.
    Returns True if LFS objects are now present.
    """
    if not os.path.exists(os.path.join(folder, ".gitattributes")):
        return True  # no LFS needed

    try:
        # Run git lfs ls-files
        result = subprocess.run(
            ["git", "lfs", "ls-files"],
            cwd=folder,
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        missing = False
        for line in output.split("\n"):
            if line.startswith("oid sha256"):
                missing = True
                break

        if missing:
            print(f"LFS files missing in {folder}, pulling...")
            subprocess.run(["git", "lfs", "install", "--local", "--force"], cwd=folder)
            subprocess.run(["git", "lfs", "pull"], cwd=folder, check=True)
        return True
    except Exception as e:
        print(f"Error checking LFS in {folder}: {e}")
        return False


summary = {}

for name, info in models_to_check.items():
    folder = info["path"]
    model_type = info["type"]

    if not os.path.exists(folder):
        summary[name] = "Folder missing"
        continue

    # Ensure LFS files are pulled
    lfs_ok = check_and_pull_lfs(folder)
    if not lfs_ok:
        summary[name] = "LFS pull failed"
        continue

    # Try loading the model
    try:
        if model_type == "encoder":
            tok = AutoTokenizer.from_pretrained(folder)
            model = AutoModel.from_pretrained(folder)
            summary[name] = "OK"
        elif model_type == "t5":
            tok = AutoTokenizer.from_pretrained(folder, legacy=False)
            model = T5ForConditionalGeneration.from_pretrained(folder)
            summary[name] = "OK"
        elif model_type == "llama":
            tok = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                folder,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            devices = set([p.device for p in model.parameters()])
            summary[name] = f"OK (loaded on {', '.join([str(d) for d in devices])})"
        elif model_type == "qwen_awq":
            # Qwen AWQ already 4-bit quantized
            tok = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                folder,
                device_map="auto",
                trust_remote_code=True,
            )
            devices = set([p.device for p in model.parameters()])
            summary[name] = f"OK (loaded on {', '.join([str(d) for d in devices])})"
        else:
            summary[name] = "Unknown type"
    except Exception as e:
        summary[name] = f"Failed to load: {e}"

# Print summary
print("\n=== Model Verification & LFS Fix Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")
