import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Wrapper for embedding models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"])
            self.dimension = len(test_embedding[0])
            
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode single text into embedding vector"""
        try:
            if not self.model:
                raise ValueError("Model not loaded")
            
            embedding = self.model.encode([text])[0]
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode multiple texts into embedding vectors"""
        try:
            if not self.model:
                raise ValueError("Model not loaded")
            
            embeddings = []
            
            # Process in batches for memory efficiency
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch)
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for text (for caching purposes)"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

class EmbeddingCache:
    """Simple in-memory cache for embeddings"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, text_hash: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        if text_hash in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(text_hash)
            self.access_order.append(text_hash)
            return self.cache[text_hash]
        return None
    
    def put(self, text_hash: str, embedding: np.ndarray):
        """Store embedding in cache"""
        if text_hash in self.cache:
            # Update existing
            self.access_order.remove(text_hash)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[text_hash] = embedding
        self.access_order.append(text_hash)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)

class CachedEmbeddingModel(EmbeddingModel):
    """Embedding model with caching"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1000):
        super().__init__(model_name)
        self.cache = EmbeddingCache(cache_size)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text with caching"""
        text_hash = self.get_text_hash(text)
        
        # Check cache first
        cached_embedding = self.cache.get(text_hash)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate new embedding
        embedding = super().encode_text(text)
        
        # Cache it
        self.cache.put(text_hash, embedding)
        
        return embedding
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode texts with caching"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = self.get_text_hash(text)
            cached_embedding = self.cache.get(text_hash)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = super().encode_texts(uncached_texts, batch_size)
            
            # Fill in the placeholders and cache new embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                text_hash = self.get_text_hash(texts[idx])
                self.cache.put(text_hash, embedding)
        
        return embeddings
