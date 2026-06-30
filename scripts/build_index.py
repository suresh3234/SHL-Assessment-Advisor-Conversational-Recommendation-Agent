import os
import sys

# Add the project root to the python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.catalog.loader import load_catalog
from app.retrieval.index import HybridIndex

def main():
    print("Loading catalog...")
    items = load_catalog()
    print(f"Loaded {len(items)} items.")
    
    print("Building hybrid index (BM25 + FAISS)...")
    index = HybridIndex()
    index.build(items)
    
    output_prefix = os.path.join(project_root, "app", "catalog", "index")
    print(f"Saving index to {output_prefix}.faiss and {output_prefix}.pkl...")
    index.save(output_prefix)
    print("Index built and saved successfully!")

if __name__ == "__main__":
    main()
