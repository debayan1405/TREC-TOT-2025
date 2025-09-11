#!/usr/bin/env python3
"""
Fix PyTorch and dependency conflicts for ColBERT
"""

import subprocess
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_pytorch_dependencies():
    """Fix PyTorch version conflicts"""
    try:
        logger.info("🔧 Fixing PyTorch dependencies...")
        
        # Uninstall conflicting packages
        packages_to_uninstall = [
            "torch", "torchvision", "torchaudio", 
            "colbert-ai", "accelerate"
        ]
        
        for package in packages_to_uninstall:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "uninstall", package, "-y"
                ], check=False, capture_output=True)
                logger.info(f"Uninstalled {package}")
            except:
                pass
        
        # Install compatible versions
        logger.info("Installing compatible PyTorch...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch>=2.0.0,<3.0.0", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ], check=True)
        
        # Install compatible accelerate
        logger.info("Installing compatible accelerate...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "accelerate>=0.20.0", "--upgrade"
        ], check=True)
        
        # Install ColBERT without torch dependencies
        logger.info("Installing ColBERT...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "colbert-ai", "--no-deps"
        ], check=True)
        
        # Install missing dependencies manually
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "faiss-gpu", "bitarray", "ninja", "ujson"
        ], check=True)
        
        logger.info("✅ Dependencies fixed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to fix dependencies: {e}")
        return False

if __name__ == "__main__":
    success = fix_pytorch_dependencies()
    if success:
        print("Dependencies fixed! You can now run the ColBERT pipeline.")
    else:
        print("Failed to fix dependencies. Please check the logs.")
