"""Helper to process DOCX files using Mistral OCR"""
import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

try:
    import requests
except ImportError:
    requests = None


def _docx_to_base64(file_path: Path) -> Optional[str]:
    """Convert DOCX file to base64 string."""
    try:
        with open(file_path, "rb") as f:
            docx_bytes = f.read()
            return base64.b64encode(docx_bytes).decode("utf-8")
    except Exception as e:
        print(f"[DOCX] Error encoding {file_path.name} to base64: {e}")
        return None


def load_docx(
    file_path: Path,
    *,
    use_mistral: bool = True,
    include_images: bool = True,
    bbox_annotation_format: Optional[Any] = None,
    document_annotation_format: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process DOCX file using Mistral AI OCR (same as PDF).
    
    Args:
        file_path: Path to DOCX file
        use_mistral: Use Mistral AI (default: True)
        include_images: Extract images (default: True)
        bbox_annotation_format: Format for bbox annotation
        document_annotation_format: Format for document annotation
        
    Returns:
        Dict with keys: 'text', 'images', 'annotations' or None if error
    """
    if not file_path.exists():
        return None
    
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not (use_mistral and mistral_api_key and requests):
        # Fallback not implemented for DOCX
        print(f"[DOCX] Mistral API required but not available")
        return None
    
    try:
        # Encode DOCX to base64
        docx_base64 = _docx_to_base64(file_path)
        if not docx_base64:
            return None
        
        # Prepare Mistral OCR request (same as PDF)
        document_payload = {
            "type": "document_url",
            "document_url": f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{docx_base64}",
        }
        
        print(f"[DOCX] Processing {file_path.name} with Mistral OCR...")
        
        api_url = "https://api.mistral.ai/v1/ocr"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mistral_api_key}",
        }
        
        ocr_payload = {
            "model": "mistral-ocr-latest",
            "document": document_payload,
            "table_format": "markdown",
            "include_image_base64": include_images,
        }
        
        # Add annotation formats
        if bbox_annotation_format:
            if isinstance(bbox_annotation_format, dict):
                ocr_payload["bbox_annotation_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": bbox_annotation_format,
                        "name": "bbox_annotation",
                        "strict": True
                    }
                }
        
        if document_annotation_format:
            if isinstance(document_annotation_format, dict):
                ocr_payload["document_annotation_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": document_annotation_format,
                        "name": "document_annotation",
                        "strict": True
                    }
                }
        
        response = requests.post(api_url, json=ocr_payload, headers=headers, timeout=120)
        
        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = str(error_json)
            except:
                pass
            print(f"[DOCX] Mistral API error {response.status_code}: {error_detail[:500]}")
            return None
        
        ocr_response = response.json()
        
        # Extract content using same logic as PDF
        from src.pdf import _extract_mistral_content
        content = _extract_mistral_content(
            ocr_response,
            pdf_filename=file_path.stem
        )
        
        if content["text"]:
            print(f"[DOCX] ✓ Extracted {len(content['text'])} chars from {file_path.name}")
            if content["images"]:
                print(f"[DOCX] ✓ Found {len(content['images'])} images")
            return content
        else:
            print(f"[DOCX] No text extracted from {file_path.name}")
            return None
            
    except Exception as e:
        print(f"[DOCX] Error processing {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None
