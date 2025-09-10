#!/usr/bin/env python3
"""
Fix Java environment for PyTerrier by setting up proper JAVA_HOME and paths.
"""
import os
import sys
from pathlib import Path

def setup_java_environment():
    """Set up Java environment for PyTerrier."""
    
    # Find the correct Java installation
    possible_java_paths = [
        "/home/ugdf8/anaconda3/envs/trec-rag/lib/jvm",
        "/home/ugdf8/anaconda3/pkgs/openjdk-17.0.15-h5ddf6bc_0/lib/jvm",
        "/home/ugdf8/anaconda3/pkgs/openjdk-21.0.6-h38aa4c6_0/lib",
        "/home/ugdf8/.vscode/extensions/redhat.java-1.45.0-linux-x64/jre/21.0.8-linux-x86_64",
    ]
    
    java_home = None
    for path in possible_java_paths:
        libjvm_path = Path(path) / "lib" / "server" / "libjvm.so"
        if libjvm_path.exists():
            java_home = path
            print(f"Found Java at: {java_home}")
            break
    
    if not java_home:
        print("ERROR: Could not find a working Java installation")
        sys.exit(1)
    
    # Set environment variables
    os.environ['JAVA_HOME'] = java_home
    os.environ['LD_LIBRARY_PATH'] = f"{java_home}/lib/server:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ['JVM_PATH'] = f"{java_home}/lib/server/libjvm.so"
    
    print(f"Set JAVA_HOME to: {java_home}")
    print(f"Set LD_LIBRARY_PATH to include: {java_home}/lib/server")
    
    return java_home

if __name__ == "__main__":
    setup_java_environment()
