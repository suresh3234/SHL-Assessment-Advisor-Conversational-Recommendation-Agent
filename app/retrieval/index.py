import os
import re
import pickle
from typing import List, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from app.catalog.loader import CatalogItem

def tokenize(text: str) -> List[str]:
    """
    Simple lowercase word tokenization.
    """
    return re.findall(r"\w+", text.lower())

class HybridIndex:
    """
    A hybrid retrieval index combining BM25 (lexical) and FAISS (semantic) search,
    using Reciprocal Rank Fusion (RRF) for re-ranking.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.catalog_items: List[CatalogItem] = []
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.model_name = model_name
        self._embedder: Optional[SentenceTransformer] = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            # Lazy load the embedder model
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def build(self, items: List[CatalogItem]):
        """
        Builds the BM25 and FAISS index from the provided catalog items.
        """
        self.catalog_items = items
        if not items:
            return

        # 1. Build BM25
        corpus = [f"{item.name} {item.description or ''}" for item in items]
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Build FAISS
        embeddings = self.embedder.encode(corpus, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        
        # Normalize embeddings for cosine similarity (inner product of normalized vectors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-9)
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(normalized_embeddings)

    def save(self, path_prefix: str):
        """
        Saves the index state to disk.
        """
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        
        # Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, f"{path_prefix}.faiss")
            
        # Save metadata and BM25 using pickle
        metadata = {
            "catalog_items": self.catalog_items,
            "bm25": self.bm25,
            "model_name": self.model_name
        }
        with open(f"{path_prefix}.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load(self, path_prefix: str):
        """
        Loads the index state from disk.
        """
        pkl_path = f"{path_prefix}.pkl"
        faiss_path = f"{path_prefix}.faiss"
        
        if not os.path.exists(pkl_path) or not os.path.exists(faiss_path):
            raise FileNotFoundError(f"Index files not found at prefix {path_prefix}")
            
        with open(pkl_path, "rb") as f:
            metadata = pickle.load(f)
            
        self.catalog_items = metadata["catalog_items"]
        self.bm25 = metadata["bm25"]
        self.model_name = metadata.get("model_name", "all-MiniLM-L6-v2")
        self.faiss_index = faiss.read_index(faiss_path)

    def search(
        self,
        query: str,
        top_k: int = 5,
        job_levels: Optional[List[str]] = None,
        languages: Optional[List[str]] = None
    ) -> List[CatalogItem]:
        """
        Performs a hybrid search using Reciprocal Rank Fusion (RRF) and filters by metadata.
        """
        if not self.catalog_items or not self.bm25 or not self.faiss_index:
            return []

        # 1. BM25 Search
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranking = np.argsort(bm25_scores)[::-1]  # indices sorted by score descending
        
        # 2. FAISS Search
        query_embedding = self.embedder.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype("float32")
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        normalized_query = query_embedding / (query_norm + 1e-9)
        
        # Search all items to get full ranking
        similarities, faiss_indices = self.faiss_index.search(normalized_query, len(self.catalog_items))
        faiss_ranking = faiss_indices[0]  # indices sorted by similarity descending

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF score: 1 / (60 + r_bm25) + 1 / (60 + r_faiss)
        rrf_scores = np.zeros(len(self.catalog_items))
        
        # Create rank lookup tables
        bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_ranking)}
        faiss_ranks = {idx: rank for rank, idx in enumerate(faiss_ranking)}
        
        for idx in range(len(self.catalog_items)):
            r_bm25 = bm25_ranks.get(idx, len(self.catalog_items))
            r_faiss = faiss_ranks.get(idx, len(self.catalog_items))
            rrf_scores[idx] = 1.0 / (60.0 + r_bm25) + 1.0 / (60.0 + r_faiss)

        # Sort indices by RRF score descending
        final_ranking = np.argsort(rrf_scores)[::-1]

        # 4. Filter results
        results = []
        for idx in final_ranking:
            item = self.catalog_items[idx]
            keep = True
            
            # Job levels filter (case-insensitive overlap)
            if job_levels:
                filter_levels = {lvl.lower() for lvl in job_levels}
                item_levels = {lvl.lower() for lvl in item.job_levels}
                if not (filter_levels & item_levels):
                    keep = False
                    
            # Languages filter (case-insensitive overlap, defaulting empty to English)
            if keep and languages:
                filter_langs = {l.lower() for l in languages}
                item_langs = {l.lower() for l in item.languages}
                if not item_langs:
                    # Treat empty as English
                    item_langs = {"english", "english (usa)", "english international"}
                
                # Check for matching substring or exact match
                matched_lang = False
                for fl in filter_langs:
                    for il in item_langs:
                        if fl in il or il in fl:
                            matched_lang = True
                            break
                    if matched_lang:
                        break
                if not matched_lang:
                    keep = False
            
            if keep:
                results.append(item)
                if len(results) >= top_k:
                    break
                    
        return results

