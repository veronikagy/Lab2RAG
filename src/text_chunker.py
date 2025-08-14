import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_id: str
    source: str
    page_number: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class TextChunker:
    """Text chunking with various strategies"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove page numbers and common PDF artifacts
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy or NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk_by_fixed_size(self, text: str, source: str) -> List[TextChunk]:
        """Create fixed-size chunks with overlap"""
        text = self.clean_text(text)
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If we're not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within the last 100 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + self.chunk_size * 0.7:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    chunk_id=f"{source}_chunk_{chunk_id}",
                    source=source,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def chunk_by_semantic_sections(self, text: str, source: str) -> List[TextChunk]:
        """Create chunks based on semantic sections (headers, paragraphs)"""
        text = self.clean_text(text)
        
        # Split by headers (lines that look like headers)
        header_pattern = r'\n([A-Z][A-Za-z\s]{5,50})\n'
        sections = re.split(header_pattern, text)
        
        chunks = []
        chunk_id = 0
        current_header = ""
        
        for i in range(0, len(sections), 2):
            if i + 1 < len(sections):
                header = sections[i + 1].strip()
                content = sections[i + 2] if i + 2 < len(sections) else ""
            else:
                header = ""
                content = sections[i]
            
            if header:
                current_header = header
            
            # If content is too long, split it further
            if len(content) > self.chunk_size:
                sub_chunks = self.chunk_by_fixed_size(content, source)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = f"{source}_section_{chunk_id}"
                    sub_chunk.metadata = {"header": current_header}
                    chunks.append(sub_chunk)
                    chunk_id += 1
            else:
                if content.strip():
                    chunk = TextChunk(
                        text=content.strip(),
                        chunk_id=f"{source}_section_{chunk_id}",
                        source=source,
                        metadata={"header": current_header}
                    )
                    chunks.append(chunk)
                    chunk_id += 1
        
        return chunks
    
    def chunk_text(self, text: str, source: str, strategy: str = "fixed_size") -> List[TextChunk]:
        """
        Main chunking method
        
        Args:
            text: Text to chunk
            source: Source identifier (filename, URL, etc.)
            strategy: Chunking strategy ("fixed_size" or "semantic")
            
        Returns:
            List of TextChunk objects
        """
        if strategy == "semantic":
            return self.chunk_by_semantic_sections(text, source)
        else:
            return self.chunk_by_fixed_size(text, source)
    
    def get_chunk_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths),
            "avg_length": sum(chunk_lengths) / len(chunk_lengths),
            "total_characters": sum(chunk_lengths)
        }
