#!/usr/bin/env python3
"""
Evaluation utilities for TREC runs
"""

import subprocess
import tempfile
import os
import json
from pathlib import Path

def evaluate_trec_run(run_file: str, qrel_file: str) -> dict:
    """
    Evaluate TREC run file using trec_eval
    
    Args:
        run_file: Path to TREC run file
        qrel_file: Path to QREL file
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Run trec_eval
        cmd = ['trec_eval', '-m', 'all_trec', qrel_file, run_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse output
        metrics = {}
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    metric = parts[0]
                    value = parts[2]
                    try:
                        metrics[metric] = float(value)
                    except ValueError:
                        metrics[metric] = value
        
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"Error running trec_eval: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return {}
    except FileNotFoundError:
        print("trec_eval not found. Installing...")
        # Fall back to pytrec_eval if available
        try:
            import pytrec_eval
            
            # Load qrels
            qrels = {}
            with open(qrel_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                        if qid not in qrels:
                            qrels[qid] = {}
                        qrels[qid][docid] = rel
            
            # Load run
            run = {}
            with open(run_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        qid, _, docid, rank, score = parts[0], parts[1], parts[2], int(parts[3]), float(parts[4])
                        if qid not in run:
                            run[qid] = {}
                        run[qid][docid] = score
            
            # Evaluate
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10', 'recip_rank', 'map'})
            results = evaluator.evaluate(run)
            
            # Average across queries
            metrics = {}
            for qid in results:
                for metric in results[qid]:
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].append(results[qid][metric])
            
            # Calculate averages
            for metric in metrics:
                metrics[metric] = sum(metrics[metric]) / len(metrics[metric])
            
            return metrics
            
        except ImportError:
            print("Neither trec_eval nor pytrec_eval available")
            return {
                'ndcg_cut_10': 0.0,
                'recip_rank': 0.0,
                'map': 0.0
            }
