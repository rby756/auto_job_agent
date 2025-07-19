"""
Vector store module for handling job description embeddings and similarity search.
"""
from typing import List, Dict, Any, Optional
import os
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

class JobVectorStore:
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5', persist_dir: str = 'data/vector_store'):
        """
        Initialize the vector store with a sentence transformer model and FAISS index.
        
        Args:
            model_name: Name of the sentence transformer model to use
            persist_dir: Directory to store the vector store
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.job_metadata = []
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
    def _initialize_index(self):
        """Initialize a new FAISS index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
    def add_jobs(self, job_texts: List[str], metadata_list: List[Dict[str, Any]] = None):
        """
        Add job descriptions to the vector store.
        
        Args:
            job_texts: List of job description texts
            metadata_list: List of metadata dictionaries for each job
        """
        if not job_texts:
            return
            
        if metadata_list is None:
            metadata_list = [{} for _ in job_texts]
            
        # Encode the job descriptions
        embeddings = self.model.encode(job_texts, convert_to_tensor=False)
        
        # Convert to numpy array if not already
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.numpy()
            
        # Initialize index if it doesn't exist
        if self.index is None:
            self._initialize_index()
            
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.job_metadata.extend([
            {"text": text, **meta}
            for text, meta in zip(job_texts, metadata_list)
        ])
    
    def get_similar_jobs(self, query: str, k: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Find similar jobs to the query.
        
        Args:
            query: Query text
            k: Number of similar jobs to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dictionaries containing similar jobs and their metadata
        """
        if self.index is None or not self.job_metadata:
            return []
            
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Convert distances to similarities (1 - normalized distance)
        similarities = 1 - (distances / 2)  # Since cosine distance = 1 - cosine_similarity
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], similarities[0]):
            if score >= threshold and idx < len(self.job_metadata):
                result = self.job_metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)
                
        return results
    
    def save(self, path: Optional[str] = None):
        """Save the vector store to disk."""
        if path is None:
            path = self.persist_dir / "job_vector_store.pkl"
            
        # Create directory if it doesn't exist
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(path) + ".index")
            
        # Save metadata
        with open(path, 'wb') as f:
            pickle.dump({
                'job_metadata': self.job_metadata,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load a vector store from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        # Create new instance
        instance = cls(model_name=data['model_name'])
        instance.job_metadata = data['job_metadata']
        
        # Load FAISS index if it exists
        index_path = str(path) + ".index"
        if os.path.exists(index_path):
            instance.index = faiss.read_index(index_path)
            
        return instance
