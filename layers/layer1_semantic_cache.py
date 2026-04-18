"""
Layer 1: Semantic Cache
Speed multiplier - skip LLMs on repeated questions using semantic similarity
"""

import json
import numpy as np
from typing import Optional, Dict, Any, Tuple
from redis import Redis
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import date, datetime
from decimal import Decimal

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle dates and decimals from the database."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

class SemanticCache:
    """
    Semantic cache that stores query embeddings and returns cached answers
    if similarity exceeds threshold.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.92,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the semantic cache.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            ttl_seconds: Time-to-live for cached entries
            similarity_threshold: Minimum cosine similarity to return cached result
            embedding_model: Sentence transformer model name
        """
        self.redis_client = Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False  # We need bytes for numpy arrays
        )
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(embedding_model)

    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence transformers."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)
        return cosine_similarity(embedding1, embedding2)[0][0]

    def get(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Check if a query has a cached response.

        Args:
            query: The user query to check
            filters: Optional dictionary of key-value pairs that must match the cached metadata.

        Returns:
            Cached response dict if similarity > threshold, None otherwise
        """
        query_embedding = self._embed(query)

        # Scan all cached entries
        for key in self.redis_client.scan_iter("cache:*"):
            cached_data = self.redis_client.get(key)
            if cached_data:
                try:
                    cached = json.loads(cached_data)
                    
                    if filters:
                        match = True
                        for k, v in filters.items():
                            if cached.get("metadata", {}).get(k) != v:
                                match = False
                                break
                        if not match:
                            continue

                    cached_embedding = np.array(cached["embedding"])

                    similarity = self._compute_similarity(
                        query_embedding,
                        cached_embedding
                    )

                    if similarity >= self.similarity_threshold:
                        # Return cached answer, reset TTL
                        self.redis_client.expire(key, self.ttl_seconds)
                        cached["cache_hit"] = True
                        cached["similarity"] = float(similarity)
                        return cached
                except (json.JSONDecodeError, KeyError):
                    continue

        return None

    def set(
        self,
        query: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a query-answer pair in the cache.

        Args:
            query: The user query
            answer: The generated answer
            metadata: Optional additional metadata to store

        Returns:
            True if stored successfully
        """
        embedding = self._embed(query)

        cache_entry = {
            "query": query,
            "answer": answer,
            "embedding": embedding.tolist(),
            "metadata": metadata or {}
        }

        # Use hash of query and metadata for uniqueness
        import hashlib
        unique_str = query
        if metadata:
            unique_str += json.dumps(metadata, sort_keys=True, default=str)
        cache_key = f"cache:{hashlib.md5(unique_str.encode()).hexdigest()}"

        self.redis_client.setex(
            cache_key,
            self.ttl_seconds,
            json.dumps(cache_entry, cls=CustomJSONEncoder)
        )
        return True

    def clear(self) -> int:
        """Clear all cached entries. Returns count of deleted keys."""
        count = 0
        for key in self.redis_client.scan_iter("cache:*"):
            self.redis_client.delete(key)
            count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        keys = list(self.redis_client.scan_iter("cache:*"))
        return {
            "total_entries": len(keys),
            "ttl_seconds": self.ttl_seconds,
            "similarity_threshold": self.similarity_threshold
        }

    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False


# Factory function for creating cache from config
def create_semantic_cache(config: Dict[str, Any]) -> SemanticCache:
    """Create a SemanticCache instance from configuration."""
    cache_config = config.get("semantic_cache", {})
    return SemanticCache(
        redis_host=config.get("redis_host", "localhost"),
        redis_port=config.get("redis_port", 6379),
        redis_db=config.get("redis_db", 0),
        ttl_seconds=cache_config.get("ttl_seconds", 3600),
        similarity_threshold=cache_config.get("similarity_threshold", 0.92),
        embedding_model=cache_config.get("embedding_model", "all-MiniLM-L6-v2")
    )


if __name__ == "__main__":
    # Example usage
    cache = SemanticCache()

    # Test connectivity
    if cache.is_healthy():
        print("Redis connection: OK")
        print(f"Cache stats: {cache.get_stats()}")
    else:
        print("Warning: Redis not connected. Run: docker run -d -p 6379:6379 redis")
