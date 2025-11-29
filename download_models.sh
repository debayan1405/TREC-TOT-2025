#!/bin/bash
set -e

# -----------------------------
# Directories
# -----------------------------
BASE="$HOME/local"
SRC="$BASE/src"
MODEL_DIR="/media/12TB/shared/models"

mkdir -p "$SRC" "$MODEL_DIR"

# -----------------------------
# Install Git LFS locally
# -----------------------------
echo "=== Installing Git LFS locally ==="
cd "$SRC"
if [ ! -d "git-lfs-3.5.1" ]; then
    wget -q https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
    tar -xzf git-lfs-linux-amd64-v3.5.1.tar.gz
fi

GIT_LFS_BIN="$SRC/git-lfs-3.5.1"
export PATH="$GIT_LFS_BIN:$PATH"

# Make PATH permanent
if ! grep -q "$GIT_LFS_BIN" ~/.bashrc; then
    echo "export PATH=\"$GIT_LFS_BIN:\$PATH\"" >> ~/.bashrc
fi

# Initialize Git LFS (force local install so repos recognize it)
git lfs install --local --force || true
echo "Git LFS version:"
git lfs version || true

# -----------------------------
# Helper function to clone and pull LFS
# -----------------------------
clone_and_pull() {
    local url=$1
    local folder=$2
    if [ ! -d "$folder" ]; then
        echo "Cloning $folder..."
        git clone "$url" "$folder"
    else
        echo "$folder already exists, skipping clone."
    fi

    # Ensure Git LFS hooks are active inside the repo
    cd "$folder"
    git lfs install --local --force || true

    # Pull all LFS objects
    if [ -f ".gitattributes" ]; then
        echo "Pulling LFS files for $folder..."
        git lfs pull || true
    fi
    cd - > /dev/null
}

# -----------------------------
# Clone all models
# -----------------------------
cd "$MODEL_DIR"

# IR / retrieval models
clone_and_pull https://huggingface.co/colbert-ir/colbertv2.0 colbertv2.0
clone_and_pull https://huggingface.co/facebook/contriever contriever
clone_and_pull https://huggingface.co/intfloat/e5-large e5-large
clone_and_pull https://huggingface.co/castorini/monot5-large-msmarco monot5-large-msmarco

# LLMs
clone_and_pull https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct llama3-70b-instruct
clone_and_pull https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ qwen2.5-72b-awq

# -----------------------------
# Finished
# -----------------------------
echo ""
echo "=== All models are cloned and LFS weights pulled ==="
ls -1 "$MODEL_DIR"
echo ""
echo "You can now use your models in Python."
