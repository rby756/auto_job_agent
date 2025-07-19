"""
Vector store module for handling job description embeddings and similarity search.
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)
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
        self.model_name = model_name
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
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """
        Save the vector store to disk.
        
        Args:
            path: Optional path to save the vector store. If None, uses the persist_dir.
        """
        if path is None:
            path = self.persist_dir / "job_descriptions.pkl"
        else:
            path = Path(path)
            
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data to save
        data = {
            'model_name': self.model_name,
            'job_metadata': self.job_metadata
        }
        
        # Save metadata
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved vector store metadata to {path}")
            
            # Save FAISS index if it exists
            if self.index is not None:
                index_path = str(path) + ".index"
                faiss.write_index(self.index, index_path)
                logger.info(f"Saved FAISS index to {index_path}")
                
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'JobVectorStore':
        """
        Load a JobVectorStore from disk.
        
        Args:
            path: Path to the saved vector store
            
        Returns:
            Loaded JobVectorStore instance
        """
        path = Path(path)
        
        try:
            # Load metadata
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            # Get model name from data or use default
            model_name = data.get('model_name')
            
            # If model_name is None or 'unknown', use the default model
            if not model_name or model_name == 'unknown':
                logger.warning(f"Invalid model name '{model_name}' found, using default model")
                model_name = 'BAAI/bge-small-en-v1.5'
            
            # Create new instance with the correct model
            try:
                instance = cls(model_name=model_name)
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {str(e)}. Using default model.")
                instance = cls()  # Use default model
            
            # Update metadata
            instance.job_metadata = data.get('job_metadata', [])
            
            # Load FAISS index if it exists
            index_path = str(path) + ".index"
            if os.path.exists(index_path):
                try:
                    instance.index = faiss.read_index(index_path)
                    logger.info(f"Successfully loaded FAISS index from {index_path}")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {str(e)}")
                    instance.index = None
            else:
                logger.warning(f"FAISS index not found at {index_path}")
                instance.index = None
                
            logger.info(f"Loaded vector store with {len(instance.job_metadata)} jobs")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading vector store from {path}: {str(e)}")
            # Return a new instance if loading fails
            logger.info("Returning a new vector store instance")
            return cls()
