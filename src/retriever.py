"""
Document Retrieval with Supabase pgvector

This module provides vector search using Supabase's pgvector extension.

Features:
- Similarity search with pgvector
- Complex SQL-based filtering
- Support for image results

Usage:
    retriever = SupabaseRetriever(config)
    results = retriever.similarity_search("What is RAG?", k=5)
"""

import os
import yaml
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Sentence transformers for query embeddings
from sentence_transformers import SentenceTransformer

# LangChain document structure
from langchain_core.documents import Document

from src.configuration import resolve_config_path
from src.supabase_client import SupabaseClient


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    resolved_path = resolve_config_path(config_path)
    with open(resolved_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class SupabaseRetriever:
    """Retriever using Supabase pgvector for semantic search."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Supabase retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize Supabase client
        self.supabase_client = SupabaseClient()
        
        # Initialize OpenAI client for embeddings (to match database)
        from openai import OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for embeddings")
        self.openai_client = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"  # Must match what was used to embed docs (1536 dims)
        
        print(f"Using OpenAI embeddings: {self.embedding_model}")
        
        # Get document count
        doc_count = self.supabase_client.get_document_count()
        print(f"✓ Retriever initialized. Documents in database: {doc_count}")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search using pgvector.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {"source": "doc.pdf"})
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        print(f"\nQuery: '{query}'")
        print(f"Searching for top {k} similar chunks...")
        
        # Encode query using OpenAI (to match database embeddings)
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        query_embedding = response.data[0].embedding
        
        # Search using Supabase
        results = self.supabase_client.vector_search(
            query_embedding=query_embedding,
            k=k,
            filter_metadata=filter_metadata,
            score_threshold=score_threshold
        )
        
        print(f"Found {len(results)} results")
        return results
    
    def search_with_filter(
        self,
        query: str,
        source_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Advanced search with metadata filtering.
        
        Args:
            query: Search query
            source_filter: Filter by source file (e.g., "doc.pdf")
            type_filter: Filter by type ("text" or "image")
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        # Build filter dictionary
        filter_dict = {}
        
        if source_filter:
            filter_dict['source'] = source_filter
        
        if type_filter:
            filter_dict['type'] = type_filter
        
        return self.similarity_search(
            query,
            k=k,
            filter_metadata=filter_dict if filter_dict else None
        )
    
    def get_images_only(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Search for images only.
        
        Args:
            query: Search query
            k: Number of image results
            
        Returns:
            List of image Document tuples with URLs in metadata
        """
        return self.search_with_filter(query, type_filter="image", k=k)
    
    def get_text_only(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for text chunks only.
        
        Args:
            query: Search query
            k: Number of text results
            
        Returns:
            List of text Document tuples
        """
        results = self.similarity_search(query, k=k)
        # Filter out images
        text_results = [
            (doc, score) for doc, score in results
            if doc.metadata.get('type') != 'image'
        ]
        return text_results[:k]
    
    def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Maximal Marginal Relevance (MMR) search for diverse results.
        
        MMR balances relevance AND diversity to avoid returning similar chunks.
        
        How it works:
        1. Fetch more candidates than needed (fetch_k)
        2. Select most relevant chunk first
        3. For remaining, balance:
           - Similarity to query (relevance)
           - Dissimilarity to already selected (diversity)
        
        Lambda parameter:
        - lambda = 1.0: Pure relevance (same as similarity search)
        - lambda = 0.5: Balanced (recommended)
        - lambda = 0.0: Pure diversity
        
        Args:
            query: Search query
            k: Number of final results
            fetch_k: Number of candidates to fetch initially
            lambda_mult: Balance between relevance (1.0) and diversity (0.0)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples with diverse results
        """
        import numpy as np
        
        print(f"\nMMR Query: '{query}'")
        print(f"Fetching {fetch_k} candidates, selecting {k} diverse results...")
        print(f"Lambda: {lambda_mult} (1.0=relevance, 0.0=diversity)")
        
        # Step 1: Get query embedding using OpenAI
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Step 2: Fetch more candidates than we need
        candidates = self.supabase_client.vector_search(
            query_embedding=query_embedding.tolist(),
            k=fetch_k,
            filter_metadata=filter_metadata
        )
        
        if not candidates:
            return []
        
        # Step 3: Get embeddings for all candidates using OpenAI
        candidate_texts = [doc.page_content for doc, score in candidates]
        candidate_response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=candidate_texts
        )
        candidate_embeddings = np.array([item.embedding for item in candidate_response.data])
        
        # Step 4: MMR selection algorithm
        selected_indices = []
        selected_embeddings = []
        
        # Always select the first (most relevant) document
        selected_indices.append(0)
        selected_embeddings.append(candidate_embeddings[0])
        
        # Step 5: Select remaining k-1 documents using MMR formula
        while len(selected_indices) < k and len(selected_indices) < len(candidates):
            best_score = -float('inf')
            best_idx = None
            
            # Evaluate each non-selected candidate
            for i in range(len(candidates)):
                if i in selected_indices:
                    continue
                
                candidate_emb = candidate_embeddings[i]
                
                # Relevance: similarity to query (higher = better)
                relevance = -np.linalg.norm(query_embedding - candidate_emb)
                
                # Diversity: dissimilarity to selected docs
                if len(selected_embeddings) > 0:
                    similarities = [
                        -np.linalg.norm(candidate_emb - sel_emb)
                        for sel_emb in selected_embeddings
                    ]
                    max_similarity = max(similarities)
                else:
                    max_similarity = 0
                
                # MMR score: balance relevance and diversity
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            # Add the best candidate to selected set
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_embeddings.append(candidate_embeddings[best_idx])
        
        # Step 6: Return selected documents with their scores
        results = [candidates[idx] for idx in selected_indices]
        
        print(f"Selected {len(results)} diverse results")
        return results



def print_results(results: List[Tuple[Document, float]], show_content: bool = True) -> None:
    """
    Display search results in readable format.
    
    Args:
        results: List of (Document, score) tuples
        show_content: Whether to display content
    """
    print("\n" + "="*60)
    print("SEARCH RESULTS")
    print("="*60)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[{i}] Similarity: {score:.4f}")
        
        # Show metadata
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            
            # Check if image
            if doc.metadata.get('type') == 'image':
                print(f"Type: IMAGE")
                print(f"URL: {doc.metadata.get('image_url', 'N/A')}")
                print(f"Page: {doc.metadata.get('page', 'N/A')}")
        
        # Show content
        if show_content:
            content = doc.page_content
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"Content: {content}")
        
        print("-" * 60)


def main():
    """CLI for testing Supabase retriever."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieve documents using Supabase pgvector")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--source", type=str, default=None, help="Filter by  source file")
    parser.add_argument("--images-only", action="store_true", help="Return only images")
    parser.add_argument("--text-only", action="store_true", help="Return only text")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Initialize retriever
    print("\n" + "="*60)
    print("INITIALIZING SUPABASE RETRIEVER")
    print("="*60)
    retriever = SupabaseRetriever(config)
    
    # Perform search
    print("\n" + "="*60)
    print("SEARCHING")
    print("="*60)
    
    if args.images_only:
        results = retriever.get_images_only(args.query, k=args.k)
    elif args.text_only:
        results = retriever.get_text_only(args.query, k=args.k)
    else:
        results = retriever.search_with_filter(
            args.query,
            source_filter=args.source,
            k=args.k
        )
    
    # Display results
    print_results(results)
    
    print("\n" + "="*60)
    print("RETRIEVAL COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
