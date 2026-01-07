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

import logging
import re
from typing import Optional, Dict, List
from PIL import Image
import base64
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
        logger.info("🔍 MistralImageSearch initialized")
    
    def annotate_image(self, image_path: str) -> str:
        """
        Annotate image using Mistral AI with SAME schema as ingestion.
        
        Uses structured output to match pdf.py annotation format exactly.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Annotation summary text
        """
        try:
            # Load and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Define EXACT SAME schema as pdf.py (lines 396-414)
            annotation_schema = {
                "type": "object",
                "properties": {
                    "image_type": {
                        "type": "string",
                        "description": "The type of the image (e.g., chart, diagram, photo, interface, screenshot)"
                    },
                    "short_description": {
                        "type": "string",
                        "description": "A brief description in English describing the image"
                    },
                    "summary": {
                        "type": "string",
                        "description": "A detailed summary of the image content, including visible text, UI elements, and key information"
                    }
                },
                "required": ["image_type", "short_description", "summary"],
                "additionalProperties": False
            }
            
            # Call Mistral Vision API with structured output
            response = self.client.chat.complete(
                model="pixtral-12b-2409",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and provide a structured annotation. Focus on: type of image, brief description, and detailed summary including all visible text, UI components, and key elements."
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "image_annotation",
                        "schema": annotation_schema,
                        "strict": True
                    }
                }
            )
            
            # Parse JSON response
            import json
            annotation_json = json.loads(response.choices[0].message.content)
            
            # Extract summary (same format as ingestion)
            summary = annotation_json.get("summary", "")
            
            logger.info(f"✓ Image annotated: {annotation_json.get('image_type')} - {len(summary)} chars")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error annotating image: {e}")
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
                    doc_match = re.search(r'/([^/]+)_img-\d+\.(png|jpg|jpeg)', image_url)
                    doc_name = doc_match.group(1) if doc_match else "Unknown"
                    
                    logger.info(f"✓ Found image: {image_url[:80]}...")
                    
                    return {
                        'image_url': image_url,
                        'doc_name': doc_name,
                        'annotation': annotation[:200],
                        'full_result': result
                    }
            
            logger.info("No images found in retrieved documents")
            return None
            
        except Exception as e:
            logger.error(f"Error in annotation search: {e}")
            import traceback
            traceback.print_exc()
            return None


def get_mistral_image_search(mistral_api_key: str) -> MistralImageSearch:
    """Create MistralImageSearch instance."""
    return MistralImageSearch(mistral_api_key)
