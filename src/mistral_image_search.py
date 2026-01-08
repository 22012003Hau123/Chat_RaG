"""
Mistral Annotation Image Search

Simple semantic search using Mistral AI annotations.
No pixel comparison, no downloads - just text search!

Flow:
1. Annotate uploaded image with Mistral
2. Search document content for similar annotations
3. Extract image URLs from matched chunks
4. Return image + document info
"""

import os
import re
import json
import base64
import logging
from typing import Optional, Dict
from mistralai import Mistral
import requests
import img2pdf
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class MistralImageSearch:
    """Semantic image search using Mistral AI annotations."""
    
    def __init__(self, mistral_api_key: str):
        """
        Initialize with Mistral API.
        
        Args:
            mistral_api_key: Mistral API key
        """
        from mistralai import Mistral
        self.client = Mistral(api_key=mistral_api_key)
        self.api_key = mistral_api_key  # Store API key for direct use with OCR API
        logger.info("🔍 MistralImageSearch initialized")
    
    def annotate_image(self, image_path: str = None, image_bytes: bytes = None) -> str:
        """
        Annotate image using Mistral OCR API (converted to PDF for consistency).
        
        Wraps image in PDF so we can use OCR API - ensures IDENTICAL
        annotation style as document ingestion.
        
        Args:
            image_path: Path to image file
            image_bytes: Raw image bytes (optional, avoids file read)
            
        Returns:
            Annotation summary text
        """
        try:
            # Load image
            if image_bytes:
                image_data = image_bytes
            elif image_path:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
            else:
                logger.error("No image provided to annotate_image")
                return ""
            
            # Convert image to PDF so we can use OCR API
            logger.info("Converting image to PDF for OCR consistency...")
            
            try:
                pdf_bytes = img2pdf.convert(image_data)
            except Exception as e:
                logger.warning(f"img2pdf failed: {e}, using PIL fallback")
                img = Image.open(BytesIO(image_data))
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                pdf_buffer = BytesIO()
                img.save(pdf_buffer, 'PDF', resolution=100.0)
                pdf_bytes = pdf_buffer.getvalue()
            
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            
            # Define schema (same as ingestion)
            annotation_schema = {
                "type": "object",
                "properties": {
                    "image_type": {
                        "type": "string",
                        "description": "The type of the image (e.g., product, chart, diagram, photo)"
                    },
                    "short_description": {
                        "type": "string",
                        "description": "A brief description in English describing the image"
                    },
                    "summary": {
                        "type": "string",
                        "description": "A detailed summary of the image content"
                    }
                },
                "required": ["image_type", "short_description", "summary"],
                "additionalProperties": False
            }
            
            # Use OCR API (same as ingestion!)
            api_url = "https://api.mistral.ai/v1/ocr"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            
            payload = {
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                "bbox_annotation_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": annotation_schema,
                        "name": "bbox_annotation",
                        "strict": True
                    }
                }
            }
            
            response = requests.post(api_url, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"OCR API error {response.status_code}: {response.text[:200]}")
                return ""
            
            ocr_response = response.json()
            
            # Extract annotation
            pages = ocr_response.get("pages", [])
            if pages:
                page = pages[0]
                bboxes = page.get("bboxes", [])
                if bboxes:
                    bbox = bboxes[0]
                    annotation_obj = bbox.get("annotation", {})
                    summary = annotation_obj.get("summary", "")
                    if summary:
                        logger.info(f"✓ OCR annotation: {summary[:100]}...")
                        return summary
            
            logger.warning("No annotation found in OCR response")
            return ""
            
        except Exception as e:
            logger.error(f"Error in annotate_image: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def search_by_annotation(
        self,
        uploaded_image_path: str,
        rag_chain,
        top_k: int = 20  # Increased from 5 to 20 for better recall
    ) -> Optional[Dict]:
        """
        Search for similar images using annotation matching.
        
        Args:
            uploaded_image_path: Path to uploaded image
            rag_chain: RAG chain instance
            top_k: Number of results to search
            
        Returns:
            Match info or None
        """
        try:
            # Annotate uploaded image
            logger.info(f"📸 Annotating uploaded image...")
            annotation = self.annotate_image(uploaded_image_path)
            
            if not annotation:
                logger.error("Failed to annotate image")
                return None
            
            logger.info(f"🔍 Searching with annotation (using MMR like normal chat)...")
            
            # Use annotation summary as RAG query with MMR (same as normal chat)
            result = rag_chain.query(
                question=annotation,
                method="mmr",  # Use MMR like normal chat (diversity + relevance)
                return_context=True,  # Crucial: Request retrieved documents
                session=None
            )
            
            # Check retrieved docs for images
            # Note: rag_chain returns documents in 'context' key when return_context=True
            retrieved_docs = result.get("context", [])
            logger.info(f"   Retrieved {len(retrieved_docs)} documents")
            
            view_pattern = r'\[View:\s*(https?://[^\]]+)\]'
            
            for doc in retrieved_docs:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                urls = re.findall(view_pattern, content)
                
                if urls:
                    image_url = urls[0]
                    doc_match = re.search(r'/([^/]+)_page\d+_img\d+\.(png|jpg|jpeg)', image_url)
                    doc_name = doc_match.group(1) if doc_match else "Unknown"
                    
                    logger.info(f"✓ Found image: {image_url[:80]}...")
                    
                    return {
                        'image_url': image_url,
                        'doc_name': doc_name,
                        'annotation': annotation,
                        'full_result': result
                    }
            
            logger.info("No images found in retrieved documents")
            return None
            
        except Exception as e:
            logger.error(f"Error in annotation search: {e}")
            import traceback
            traceback.print_exc()
            return None


# Factory function for app.py import
def get_mistral_image_search(mistral_api_key: str) -> MistralImageSearch:
    """Create MistralImageSearch instance."""
    return MistralImageSearch(mistral_api_key)
