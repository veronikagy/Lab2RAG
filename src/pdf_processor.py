import io
import logging
from typing import List, Optional
import PyPDF2
import pdfplumber
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF document processing and text extraction"""
    
    def __init__(self):
        self.supported_formats = [".pdf"]
    
    def extract_text_pypdf2(self, file_content: bytes) -> str:
        """Extract text using PyPDF2 (faster but less accurate)"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            return ""
    
    def extract_text_pdfplumber(self, file_content: bytes) -> str:
        """Extract text using pdfplumber (more accurate but slower)"""
        try:
            pdf_file = io.BytesIO(file_content)
            text = ""
            
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return ""
    
    def extract_text(self, file_content: bytes, method: str = "pdfplumber") -> str:
        """
        Extract text from PDF file
        
        Args:
            file_content: PDF file content as bytes
            method: Extraction method ("pypdf2" or "pdfplumber")
            
        Returns:
            Extracted text as string
        """
        if method == "pypdf2":
            text = self.extract_text_pypdf2(file_content)
        else:
            text = self.extract_text_pdfplumber(file_content)
        
        # Fallback to other method if first one fails
        if not text and method == "pdfplumber":
            logger.info("pdfplumber failed, trying PyPDF2")
            text = self.extract_text_pypdf2(file_content)
        elif not text and method == "pypdf2":
            logger.info("PyPDF2 failed, trying pdfplumber")
            text = self.extract_text_pdfplumber(file_content)
        
        return text
    
    def validate_pdf(self, file_content: bytes) -> bool:
        """Validate if file is a valid PDF"""
        try:
            pdf_file = io.BytesIO(file_content)
            PyPDF2.PdfReader(pdf_file)
            return True
        except Exception:
            return False
    
    def get_pdf_info(self, file_content: bytes) -> dict:
        """Get PDF metadata information"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            info = {
                "num_pages": len(pdf_reader.pages),
                "metadata": pdf_reader.metadata if pdf_reader.metadata else {},
                "is_encrypted": pdf_reader.is_encrypted
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting PDF info: {e}")
            return {"error": str(e)}
