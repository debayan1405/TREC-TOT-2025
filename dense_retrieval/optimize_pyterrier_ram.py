#!/usr/bin/env python3
"""
PyTerrier RAM Optimization Script
================================

Forces PyTerrier meta index to load entirely into RAM (7.9GB)
for maximum performance during dense retrieval.
"""

import os
import sys
import shutil
import time
import psutil
from pathlib import Path


def backup_properties_file(properties_file: str):
    """Create backup of original properties file"""
    backup_file = f"{properties_file}.backup"
    if not os.path.exists(backup_file):
        shutil.copy2(properties_file, backup_file)
        print(f"✅ Backup created: {backup_file}")
    return backup_file


def optimize_properties_file(properties_file: str):
    """Optimize PyTerrier properties for maximum RAM usage"""
    
    print(f"🔧 Optimizing {properties_file}")
    
    # Create backup first
    backup_properties_file(properties_file)
    
    # Read existing properties
    with open(properties_file, 'r') as f:
        lines = f.readlines()
    
    # Track what optimizations we apply
    optimizations_applied = []
    
    # Find and update existing properties
    found_properties = set()
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Force meta index to memory
        if line_stripped.startswith('index.meta.data-source='):
            lines[i] = 'index.meta.data-source=fileinmem\n'
            found_properties.add('meta-memory')
            optimizations_applied.append("Meta index → RAM (7.9GB)")
        
        # Force lexicon termids to memory
        elif line_stripped.startswith('index.lexicon.termids='):
            lines[i] = 'index.lexicon.termids=fileinmem\n'
            found_properties.add('lexicon-memory')
            optimizations_applied.append("Lexicon term IDs → RAM")
        
        # Optimize compression for speed
        elif line_stripped.startswith('index.meta.compression.configuration='):
            lines[i] = 'index.meta.compression.configuration=NONE\n'
            found_properties.add('compression-speed')
            optimizations_applied.append("Compression → None (faster)")
    
    # Add missing optimizations
    new_properties = []
    
    if 'meta-memory' not in found_properties:
        new_properties.append('index.meta.data-source=fileinmem\n')
        optimizations_applied.append("Meta index → RAM (7.9GB)")
    
    if 'lexicon-memory' not in found_properties:
        new_properties.append('index.lexicon.termids=fileinmem\n')
        optimizations_applied.append("Lexicon term IDs → RAM")
    
    # Additional performance optimizations
    additional_opts = [
        ('index.meta.reverse.compression.configuration=NONE\n', "Reverse meta compression → None"),
        ('index.direct.compression.configuration=NONE\n', "Direct index compression → None"),
        ('index.inverted.compression.configuration=DEFAULT\n', "Inverted index → Default compression"),
        ('termpipelines.skip=true\n', "Term pipelines → Disabled for speed"),
        ('querying.postfilters.controls=\n', "Post filters → Disabled"),
        ('querying.postfilters.order=\n', "Post filter order → Disabled"),
    ]
    
    for prop, description in additional_opts:
        prop_key = prop.split('=')[0]
        if not any(prop_key in line for line in lines):
            new_properties.append(prop)
            optimizations_applied.append(description)
    
    # Add new properties
    if new_properties:
        lines.extend(['\n# RAM Optimizations\n'] + new_properties)
    
    # Write optimized properties
    with open(properties_file, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Applied {len(optimizations_applied)} optimizations:")
    for opt in optimizations_applied:
        print(f"   - {opt}")
    
    return optimizations_applied


def estimate_ram_usage():
    """Estimate RAM usage after optimization"""
    
    ram = psutil.virtual_memory()
    available_gb = ram.available / (1024**3)
    
    # Estimated usage by components
    components = {
        "PyTerrier meta index": 7.9,
        "Document cache": 50.0,
        "Model embeddings": 2.0,
        "System overhead": 5.0,
        "CUDA context": 3.0
    }
    
    total_estimated = sum(components.values())
    
    print(f"\n📊 ESTIMATED RAM USAGE:")
    print(f"   Available RAM: {available_gb:.1f}GB")
    print(f"   Total estimated usage: {total_estimated:.1f}GB")
    print(f"   RAM components:")
    
    for component, usage in components.items():
        print(f"     - {component}: {usage:.1f}GB")
    
    if total_estimated > available_gb:
        print(f"   ⚠️  WARNING: Estimated usage exceeds available RAM!")
        print(f"   💡 Consider reducing document cache size")
    else:
        remaining = available_gb - total_estimated
        print(f"   ✅ Remaining RAM: {remaining:.1f}GB")
    
    return components


def test_optimization(index_path: str):
    """Test if the optimization works"""
    
    print(f"\n🧪 TESTING OPTIMIZATION:")
    print(f"   Index path: {index_path}")
    
    try:
        import pyterrier as pt
        
        # Initialize PyTerrier
        if not pt.started():
            pt.init()
        
        # Load index
        start_time = time.time()
        indexref = pt.IndexRef.of(index_path)
        index = pt.IndexFactory.of(indexref)
        
        # Get index stats
        stats = index.getCollectionStatistics()
        num_docs = stats.getNumberOfDocuments()
        load_time = time.time() - start_time
        
        print(f"   ✅ Index loaded successfully")
        print(f"   📊 Documents: {num_docs:,}")
        print(f"   ⏱️  Load time: {load_time:.2f}s")
        
        # Test document retrieval
        print(f"   🧪 Testing document retrieval...")
        
        import pandas as pd
        import pyterrier.text
        
        test_docnos = ['442647', '1892675', '1246394']
        docs_df = pd.DataFrame({
            'qid': ['test'] * len(test_docnos),
            'docno': test_docnos,
            'score': [1.0] * len(test_docnos),
            'rank': [1, 2, 3]
        })
        
        text_pipeline = pyterrier.text.get_text(indexref)
        
        start_time = time.time()
        result_df = text_pipeline(docs_df)
        retrieval_time = time.time() - start_time
        
        print(f"   ✅ Document retrieval successful")
        print(f"   ⏱️  Retrieval time: {retrieval_time:.3f}s for {len(test_docnos)} docs")
        
        # Check if optimization took effect (no warnings)
        if retrieval_time < 0.1:
            print(f"   🚀 FAST RETRIEVAL - Optimization likely successful!")
        else:
            print(f"   ⚠️  Slower retrieval - Check for disk I/O warnings")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def main():
    """Main optimization function"""
    
    print("🚀 PYTERRIER RAM OPTIMIZATION")
    print("=" * 50)
    
    # Get index path
    index_path = "./trec-tot-2025-pyterrier-index"
    
    if not os.path.exists(index_path):
        print(f"❌ Index not found: {index_path}")
        return 1
    
    properties_file = os.path.join(index_path, "data.properties")
    
    if not os.path.exists(properties_file):
        print(f"❌ Properties file not found: {properties_file}")
        return 1
    
    print(f"📁 Index path: {index_path}")
    print(f"📄 Properties file: {properties_file}")
    
    # Check current RAM
    ram = psutil.virtual_memory()
    print(f"💾 Available RAM: {ram.available / (1024**3):.1f}GB / {ram.total / (1024**3):.1f}GB")
    
    # Estimate RAM usage
    estimate_ram_usage()
    
    # Apply optimizations
    print(f"\n🔧 APPLYING OPTIMIZATIONS:")
    optimizations = optimize_properties_file(properties_file)
    
    print(f"\n✅ Optimization complete!")
    print(f"📝 {len(optimizations)} optimizations applied")
    
    # Test the optimization
    test_optimization(index_path)
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Run the optimized dense retrieval pipeline")
    print(f"   2. Monitor for PyTerrier warnings (should be minimal)")
    print(f"   3. Expect 7.9GB RAM usage for meta index")
    print(f"   4. Enjoy faster document retrieval! 🚀")
    
    return 0


if __name__ == "__main__":
    exit(main())
