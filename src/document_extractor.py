"""
Document Extractor - Quick text extraction from documents

Extracts text content from PDF, DOCX, PPTX files for immediate RAG queries.
Does NOT store in database - only provides context for current query.

Usage:
    from src.document_extractor import DocumentExtractor
    extractor = DocumentExtractor()
    text = extractor.extract_from_file(file_path)
"""

import os
from typing import Optional
from pathlib import Path


class DocumentExtractor:
    """
    Quick text extraction from documents for chat queries.
    
    Supports:
    - PDF files (via PyPDF)
    - DOCX files (via python-docx)
    - PPTX files (via python-pptx)
    """
    
    def __init__(self, max_chars: int = 50000):
        """
        Initialize document extractor.
        
        Args:
            max_chars: Maximum characters to extract (prevent huge context)
        """
        self.max_chars = max_chars
        print(f"✓ Document Extractor initialized (max: {max_chars} chars)")
    
    def extract_from_file(self, file_path: str) -> str:
        """
        Extract text from any supported document type.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted text content
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        print(f"\n📄 Extracting text from: {Path(file_path).name}")
        
        if ext == '.pdf':
            text = self._extract_pdf(file_path)
        elif ext == '.docx':
            text = self._extract_docx(file_path)
        elif ext == '.pptx':
            text = self._extract_pptx(file_path)
        else:
            return f"[Unsupported file type: {ext}]"
        
        # Truncate if too long
        if len(text) > self.max_chars:
            text = text[:self.max_chars] + f"\n\n[... truncated, original length: {len(text)} chars]"
            print(f"⚠️  Text truncated to {self.max_chars} chars")
        
        print(f"✓ Extracted {len(text)} characters")
        return text
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF."""
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            text_parts = []
            
            # Extract from all pages
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i+1} ---\n{page_text}")
                
                # Stop if exceeding max chars
                current_length = sum(len(p) for p in text_parts)
                if current_length > self.max_chars:
                    break
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            return f"[Error extracting PDF: {str(e)}]"
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX using python-docx."""
        try:
            from docx import Document
            
            doc = Document(file_path)
            text_parts = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
                
                # Stop if exceeding max chars
                current_length = sum(len(p) for p in text_parts)
                if current_length > self.max_chars:
                    break
            
            # Extract tables
            if sum(len(p) for p in text_parts) < self.max_chars:
                for table in doc.tables:
                    table_text = []
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells)
                        table_text.append(row_text)
                    text_parts.append("\n".join(table_text))
                    
                    if sum(len(p) for p in text_parts) > self.max_chars:
                        break
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            return f"[Error extracting DOCX: {str(e)}]"
    
    def _extract_pptx(self, file_path: str) -> str:
        """Extract text from PPTX using python-pptx."""
        try:
            from pptx import Presentation
            
            prs = Presentation(file_path)
            text_parts = []
            
            # Extract from all slides
            for i, slide in enumerate(prs.slides):
                slide_text = [f"--- Slide {i+1} ---"]
                
                # Extract from all shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text)
                
                text_parts.append("\n".join(slide_text))
                
                # Stop if exceeding max chars
                current_length = sum(len(p) for p in text_parts)
                if current_length > self.max_chars:
                    break
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            return f"[Error extracting PPTX: {str(e)}]"


def get_file_type(filename: str) -> str:
    """
    Determine file type from filename.
    
    Returns: 'image', 'pdf', 'docx', 'pptx', or 'unknown'
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
        return 'image'
    elif ext == '.pdf':
        return 'pdf'
    elif ext == '.docx':
        return 'docx'
    elif ext == '.pptx':
        return 'pptx'
    else:
        return 'unknown'


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_extractor.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    extractor = DocumentExtractor()
    text = extractor.extract_from_file(file_path)
    
    print("\n" + "="*60)
    print("EXTRACTED TEXT")
    print("="*60)
    print(text)
    print("="*60)
