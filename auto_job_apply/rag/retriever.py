"""
Module for handling semantic search using FAISS.
"""
from typing import List, Dict, Any, Optional, Union
import os
import pickle
from pathlib import Path
import numpy as np
import faiss

from .embedder import TextEmbedder


class VectorRetriever:
    def __init__(self, embedder: Optional[TextEmbedder] = None):
        """
        Initialize the vector retriever with an embedder.
        
        Args:
            embedder: Optional pre-initialized text embedder
        """
        self.embedder = embedder or TextEmbedder()
        self.index = None
        self.metadata = []
        
    def _initialize_index(self, embedding_dim: int):
        """Initialize a new FAISS index."""
        self.index = faiss.IndexFlatL2(embedding_dim)
        
    def add_documents(self, texts: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None):
        """
        Add documents to the retriever.
        
        Args:
            texts: List of document texts
            metadata_list: Optional list of metadata dictionaries for each document
        """
        if not texts:
            return
            
        if metadata_list is None:
            metadata_list = [{} for _ in texts]
        elif len(metadata_list) != len(texts):
            raise ValueError("metadata_list must have the same length as texts")
            
        # Generate embeddings
        embeddings = self.embedder.embed_texts(texts)
        
        # Initialize index if it doesn't exist
        if self.index is None:
            self._initialize_index(embeddings.shape[1])
            
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata with text
        self.metadata.extend([
            {"text": text, **meta}
            for text, meta in zip(texts, metadata_list)
        ])
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find similar documents to the query.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dictionaries containing similar documents and their metadata
        """
        if self.index is None or not self.metadata:
            return []
            
        # Generate query embedding
        query_embedding = self.embedder.embed_texts([query])
        
        # Search the index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(k, len(self.metadata))
        )
        
        # Convert distances to similarities (1 - normalized distance)
        similarities = 1 - (distances / 2)  # Since cosine distance = 1 - cosine_similarity
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], similarities[0]):
            if score >= score_threshold and 0 <= idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)
                
                # Stop if we have enough results
                if len(results) >= k:
                    break
                    
        return results
    
    def save(self, path: Union[str, Path]):
        """
        Save the retriever to disk.
        
        Args:
            path: Path to save the retriever
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(path) + ".index")
            
        # Save metadata and config
        with open(path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'model_name': self.embedder.model_name,
                'embedding_dim': self.embedder.embedding_dimension
            }, f)
    
    @classmethod
    def load(
        cls, 
        path: Union[str, Path], 
        embedder: Optional[TextEmbedder] = None
    ) -> 'VectorRetriever':
        """
        Load a retriever from disk.
        
        Args:
            path: Path to the saved retriever
            embedder: Optional pre-initialized embedder
            
        Returns:
            Loaded VectorRetriever instance
        """
        path = Path(path)
        
        # Load metadata and config
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        # Create instance with appropriate embedder
        if embedder is None:
            embedder = TextEmbedder(model_name=data.get('model_name'))
            
        instance = cls(embedder=embedder)
        instance.metadata = data['metadata']
        
        # Load FAISS index if it exists
        index_path = str(path) + ".index"
        if os.path.exists(index_path):
            instance.index = faiss.read_index(index_path)
            
        return instance