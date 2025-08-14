import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from pathlib import Path

from .pdf_processor import PDFProcessor
from .text_chunker import TextChunker, TextChunk
from .embeddings import CachedEmbeddingModel
from .vector_store import QdrantVectorStore
from .llm_client import OpenRouterClient, RAGPromptBuilder

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG (Retrieval-Augmented Generation) pipeline"""
    
    def __init__(
        self,
        openrouter_api_key: str,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "rag_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = CachedEmbeddingModel(model_name=embedding_model)
        self.vector_store = QdrantVectorStore(qdrant_url, qdrant_api_key, collection_name)
        self.llm_client = OpenRouterClient(openrouter_api_key)
        self.prompt_builder = RAGPromptBuilder()
        
        # Initialize vector store collection
        self.vector_store.create_collection(self.embedding_model.dimension)
        
        logger.info("RAG Pipeline initialized successfully")
    
    async def process_pdf(
        self, 
        file_content: bytes, 
        filename: str,
        chunking_strategy: str = "fixed_size"
    ) -> Dict[str, Any]:
        """Process PDF file and add to vector store"""
        try:
            # Validate PDF
            if not self.pdf_processor.validate_pdf(file_content):
                return {"success": False, "error": "Invalid PDF file"}
            
            # Extract text
            logger.info(f"Extracting text from {filename}")
            text = self.pdf_processor.extract_text(file_content)
            
            if not text.strip():
                return {"success": False, "error": "No text found in PDF"}
            
            # Get PDF info
            pdf_info = self.pdf_processor.get_pdf_info(file_content)
            
            # Chunk text
            logger.info(f"Chunking text from {filename}")
            chunks = self.text_chunker.chunk_text(text, filename, chunking_strategy)
            
            if not chunks:
                return {"success": False, "error": "No chunks created"}
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_model.encode_texts(chunk_texts)
            
            # Add to vector store
            logger.info(f"Adding chunks to vector store")
            success = self.vector_store.add_chunks(chunks, embeddings)
            
            if not success:
                return {"success": False, "error": "Failed to add chunks to vector store"}
            
            # Get statistics
            chunk_stats = self.text_chunker.get_chunk_stats(chunks)
            
            return {
                "success": True,
                "filename": filename,
                "pdf_info": pdf_info,
                "chunk_stats": chunk_stats,
                "total_chunks": len(chunks),
                "text_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            return {"success": False, "error": str(e)}
    
    async def query(
        self,
        question: str,
        model: str,
        max_chunks: int = 5,
        similarity_threshold: float = 0.1,
        source_filter: Optional[str] = None,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            if use_rag:
                # Generate embedding for question
                logger.info(f"Generating embedding for question: {question[:100]}...")
                question_embedding = self.embedding_model.encode_text(question)
                
                # Search similar chunks
                logger.info("Searching for similar chunks")
                similar_chunks = self.vector_store.search_similar(
                    question_embedding,
                    limit=max_chunks,
                    score_threshold=similarity_threshold,
                    source_filter=source_filter
                )
                
                if not similar_chunks:
                    logger.info("No similar chunks found")
                    # Fallback to simple prompt
                    messages = self.prompt_builder.build_simple_prompt(question)
                    context_used = []
                else:
                    # Build RAG prompt
                    messages = self.prompt_builder.build_rag_prompt(question, similar_chunks)
                    context_used = similar_chunks
                
            else:
                # Simple prompt without RAG
                messages = self.prompt_builder.build_simple_prompt(question)
                context_used = []
            
            # Generate response
            logger.info(f"Generating response with model: {model}")
            response = await self.llm_client.generate_response(
                messages=messages,
                model=model,
                max_tokens=1000,
                temperature=0.7
            )
            
            return {
                "success": True,
                "question": question,
                "answer": response,
                "model": model,
                "context_chunks": context_used,
                "rag_used": use_rag and len(context_used) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }
    
    async def query_stream(
        self,
        question: str,
        model: str,
        max_chunks: int = 5,
        similarity_threshold: float = 0.1,
        source_filter: Optional[str] = None,
        use_rag: bool = True
    ):
        """Query with streaming response"""
        try:
            context_used = []
            
            if use_rag:
                # Generate embedding and search
                question_embedding = self.embedding_model.encode_text(question)
                similar_chunks = self.vector_store.search_similar(
                    question_embedding,
                    limit=max_chunks,
                    score_threshold=similarity_threshold,
                    source_filter=source_filter
                )
                
                if similar_chunks:
                    messages = self.prompt_builder.build_rag_prompt(question, similar_chunks)
                    context_used = similar_chunks
                else:
                    messages = self.prompt_builder.build_simple_prompt(question)
            else:
                messages = self.prompt_builder.build_simple_prompt(question)
            
            # Stream response
            async for chunk in self.llm_client.generate_response_stream(
                messages=messages,
                model=model,
                max_tokens=1000,
                temperature=0.7
            ):
                yield {
                    "chunk": chunk,
                    "context_chunks": context_used,
                    "rag_used": use_rag and len(context_used) > 0
                }
                
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield {"error": str(e)}
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models from OpenRouter"""
        return await self.llm_client.get_available_models()
    
    def get_document_sources(self) -> List[str]:
        """Get list of document sources in vector store"""
        return self.vector_store.list_sources()
    
    def delete_document(self, source: str) -> bool:
        """Delete document from vector store"""
        return self.vector_store.delete_by_source(source)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        vector_stats = self.vector_store.get_collection_stats()
        cache_stats = {
            "embedding_cache_size": self.embedding_model.cache.size()
        }
        
        return {
            "vector_store": vector_stats,
            "embedding_cache": cache_stats,
            "documents": self.get_document_sources()
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components"""
        return {
            "vector_store": self.vector_store.health_check(),
            "embedding_model": self.embedding_model.model is not None
        }
