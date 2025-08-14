import logging
from typing import List, Dict, Any, Optional, Tuple
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, SearchRequest
)
from qdrant_client.http.exceptions import ResponseHandlingException
import numpy as np

from .text_chunker import TextChunk

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """Qdrant vector database interface"""
    
    def __init__(self, url: str, api_key: str, collection_name: str):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.client = None
        self.dimension = None
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant"""
        try:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30
            )
            logger.info(f"Connected to Qdrant at {self.url}")
            
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            raise
    
    def create_collection(self, dimension: int, distance_metric: Distance = Distance.COSINE):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                # Get collection info to verify dimension
                collection_info = self.client.get_collection(self.collection_name)
                existing_dimension = collection_info.config.params.vectors.size
                if existing_dimension != dimension:
                    logger.warning(f"Collection dimension mismatch: {existing_dimension} vs {dimension}")
                self.dimension = existing_dimension
                
                # Ensure source field index exists for filtering
                self._ensure_source_index()
            else:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=distance_metric
                    )
                )
                self.dimension = dimension
                logger.info(f"Created collection '{self.collection_name}' with dimension {dimension}")
                
                # Create index for source field to enable filtering
                self._create_source_index()
                
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def _create_source_index(self):
        """Create index for source field to enable filtering"""
        try:
            # Try different import approaches for different Qdrant versions
            try:
                from qdrant_client.models import CreateIndexRequest, IndexType, FieldIndexParams
                self.client.create_index(
                    collection_name=self.collection_name,
                    request=CreateIndexRequest(
                        field_name="source",
                        field_schema=FieldIndexParams(
                            field_type=IndexType.KEYWORD
                        )
                    )
                )
            except ImportError:
                # Fallback for older versions
                try:
                    from qdrant_client.models import IndexType
                    self.client.create_index(
                        collection_name=self.collection_name,
                        field_name="source",
                        field_schema=IndexType.KEYWORD
                    )
                except Exception as e2:
                    logger.warning(f"Could not create index with fallback method: {e2}")
                    # Try direct API call
                    self.client.create_index(
                        collection_name=self.collection_name,
                        field_name="source",
                        field_schema="keyword"
                    )
            
            logger.info("Created index for source field")
        except Exception as e:
            logger.warning(f"Could not create source field index: {e}")
            # Try one more approach - direct payload index creation
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source",
                    field_schema="keyword"
                )
                logger.info("Created payload index for source field")
            except Exception as e2:
                logger.error(f"All methods to create source index failed: {e2}")
    
    def _ensure_source_index(self):
        """Ensure source field index exists, create if missing"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Try different ways to access indexes depending on Qdrant version
            existing_indexes = []
            try:
                # Newer Qdrant versions
                if hasattr(collection_info.config.params, 'indexes'):
                    existing_indexes = collection_info.config.params.indexes or []
                elif hasattr(collection_info.config, 'indexes'):
                    existing_indexes = collection_info.config.indexes or []
                elif hasattr(collection_info, 'indexes'):
                    existing_indexes = collection_info.indexes or []
            except Exception:
                existing_indexes = []
            
            # Check if source index exists
            source_index_exists = any(
                hasattr(idx, 'field_name') and idx.field_name == "source" 
                for idx in existing_indexes
            )
            
            if not source_index_exists:
                logger.info("Source field index not found, creating...")
                self._create_source_index()
            else:
                logger.info("Source field index already exists")
                
        except Exception as e:
            logger.warning(f"Could not check source field index: {e}")
            # Try to create index anyway
            self._create_source_index()
    
    def recreate_source_index(self) -> bool:
        """Recreate the source field index (useful for fixing index issues)"""
        try:
            # Try to delete existing index first
            try:
                self.client.delete_index(
                    collection_name=self.collection_name,
                    field_name="source"
                )
                logger.info("Deleted existing source field index")
            except Exception as e:
                logger.info(f"No existing source index to delete: {e}")
            
            # Create new index
            self._create_source_index()
            logger.info("Successfully recreated source field index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recreate source field index: {e}")
            return False
    
    def add_chunks(self, chunks: List[TextChunk], embeddings: List[np.ndarray]) -> bool:
        """Add text chunks with embeddings to vector store"""
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks and embeddings must match")
            
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "page_number": chunk.page_number,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "metadata": chunk.metadata or {}
                    }
                )
                points.append(point)
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            return False
    
    def search_similar(
        self, 
        query_embedding: np.ndarray, 
        limit: int = 5,
        score_threshold: float = 0.0,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        try:
            # Prepare filter
            query_filter = None
            if source_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source_filter)
                        )
                    ]
                )
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for point in search_result:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "chunk_id": point.payload.get("chunk_id", ""),
                    "source": point.payload.get("source", ""),
                    "page_number": point.payload.get("page_number"),
                    "metadata": point.payload.get("metadata", {})
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def delete_by_source(self, source: str) -> bool:
        """Delete all chunks from a specific source"""
        try:
            # First try to delete using filter (requires index)
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source)
                    )
                ]
            )
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition
            )
            
            logger.info(f"Deleted all chunks from source: {source}")
            return True
            
        except Exception as e:
            logger.warning(f"Filter-based deletion failed: {e}")
            
            # Fallback: scroll to find points and delete by ID
            try:
                logger.info(f"Falling back to scroll-based deletion for source: {source}")
                deleted_count = self._delete_by_source_scroll(source)
                if deleted_count > 0:
                    logger.info(f"Successfully deleted {deleted_count} chunks from source: {source}")
                    return True
                else:
                    logger.warning(f"No chunks found for source: {source}")
                    return True  # Consider this success if no chunks to delete
            except Exception as scroll_error:
                logger.error(f"Scroll-based deletion also failed: {scroll_error}")
                return False
    
    def _delete_by_source_scroll(self, source: str) -> int:
        """Delete chunks by source using scroll and delete by ID (fallback method)"""
        try:
            deleted_count = 0
            offset = None
            
            while True:
                # Scroll through ALL points (without filter since we don't have index)
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True  # We need payload to filter by source
                )
                
                points = scroll_result[0]
                if not points:
                    break
                
                # Filter points by source in Python
                source_points = [
                    point for point in points 
                    if point.payload.get("source") == source
                ]
                
                # Delete points by ID
                if source_points:
                    point_ids = [point.id for point in source_points]
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=point_ids
                    )
                    deleted_count += len(point_ids)
                    logger.info(f"Deleted {len(point_ids)} chunks from source: {source}")
                
                # Update offset for next iteration
                offset = scroll_result[1]
                if not offset:
                    break
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error in scroll-based deletion: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Try different ways to access indexes depending on Qdrant version
            indexes = []
            try:
                if hasattr(collection_info.config.params, 'indexes'):
                    indexes = collection_info.config.params.indexes or []
                elif hasattr(collection_info.config, 'indexes'):
                    indexes = collection_info.config.indexes or []
                elif hasattr(collection_info, 'indexes'):
                    indexes = collection_info.indexes or []
            except Exception:
                indexes = []
            
            # Build stats with safe attribute access
            stats = {
                "points_count": getattr(collection_info, 'points_count', 0),
                "segments_count": getattr(collection_info, 'segments_count', 0),
                "config": {
                    "dimension": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value
                },
                "indexes": [
                    {
                        "field_name": getattr(idx, 'field_name', 'unknown'),
                        "field_type": getattr(idx, 'field_type', 'unknown')
                    }
                    for idx in indexes
                ]
            }
            
            # Add optional attributes if they exist
            if hasattr(collection_info, 'disk_data_size'):
                stats["disk_data_size"] = collection_info.disk_data_size
            if hasattr(collection_info, 'ram_data_size'):
                stats["ram_data_size"] = collection_info.ram_data_size
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def check_indexes(self) -> Dict[str, Any]:
        """Check the status of collection indexes"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Try different ways to access indexes depending on Qdrant version
            indexes = []
            try:
                if hasattr(collection_info.config.params, 'indexes'):
                    indexes = collection_info.config.params.indexes or []
                elif hasattr(collection_info.config, 'indexes'):
                    indexes = collection_info.config.indexes or []
                elif hasattr(collection_info, 'indexes'):
                    indexes = collection_info.indexes or []
            except Exception:
                indexes = []
            
            index_status = {}
            for idx in indexes:
                field_name = getattr(idx, 'field_name', 'unknown')
                field_type = 'unknown'
                try:
                    if hasattr(idx, 'field_schema') and hasattr(idx.field_schema, 'field_type'):
                        field_type = idx.field_schema.field_type.value
                    elif hasattr(idx, 'field_type'):
                        field_type = idx.field_type.value
                except Exception:
                    field_type = 'unknown'
                
                index_status[field_name] = {
                    "field_type": field_type,
                    "exists": True
                }
            
            # Check for required indexes
            required_indexes = ["source"]
            for required_idx in required_indexes:
                if required_idx not in index_status:
                    index_status[required_idx] = {"exists": False, "field_type": "missing"}
            
            return {
                "collection_name": self.collection_name,
                "indexes": index_status,
                "has_source_index": index_status.get("source", {}).get("exists", False)
            }
            
        except Exception as e:
            logger.error(f"Error checking indexes: {e}")
            return {"error": str(e)}
    
    def list_sources(self) -> List[str]:
        """List all unique sources in the collection"""
        try:
            # Scroll through all points to get sources
            sources = set()
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=["source"]
            )
            
            for point in scroll_result[0]:
                source = point.payload.get("source")
                if source:
                    sources.add(source)
            
            return list(sources)
            
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible"""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
