"""Xử lý file PDF sử dụng Mistral AI OCR (hoặc pypdf fallback)."""

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import requests
except ImportError:
    requests = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def _pdf_to_base64(file_path: Path) -> Optional[str]:
    """Convert PDF file sang base64 string."""
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
            return base64.b64encode(pdf_bytes).decode("utf-8")
    except Exception as e:
        print(f"[PDF] Error encoding {file_path.name} to base64: {e}")
        return None


def _extract_mistral_content(
    ocr_response: Dict[str, Any],
    image_server_base_url: Optional[str] = None,
    project_path: Optional[str] = None,
    pdf_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract content từ Mistral OCR response.
    
    Args:
        ocr_response: Response từ Mistral OCR API
        image_server_base_url: Base URL của image server (e.g. "http://localhost:1200")
        project_path: Đường dẫn project trong data/ (e.g. "process auchan")
        pdf_filename: Tên file PDF (để tạo tên thư mục images)
    
    Returns:
        Dict với keys: 'text', 'images', 'annotations'
    """
    result = {
        "text": "",
        "images": [],
        "annotations": None,
    }
    
    # Extract markdown text từ tất cả pages
    pages = ocr_response.get("pages", [])
    markdown_parts = []
    image_annotations_map = {}  # Map image ID -> annotation để inject vào text
    
    for page in pages:
        # Main markdown content
        page_markdown = page.get("markdown", "")
        if page_markdown:
            markdown_parts.append(f"## Page {page.get('index', 0) + 1}\n\n{page_markdown}")
        
        # Header và footer nếu có
        header = page.get("header")
        footer = page.get("footer")
        if header:
            markdown_parts.append(f"**Header:** {header}")
        if footer:
            markdown_parts.append(f"**Footer:** {footer}")
        
        # Extract images info including bbox annotations
        images = page.get("images", [])
        if images:
            for img in images:
                img_id = img.get("id")  # e.g. "img-0.jpeg"
                bbox_annotation = img.get("image_annotation")
                
                if bbox_annotation:
                    print(f"[DEBUG] Found bbox annotation for {img_id}: {str(bbox_annotation)[:100]}")
                    if isinstance(bbox_annotation, str):
                        # Parse JSON string nếu cần
                        try:
                            bbox_annotation = json.loads(bbox_annotation)
                        except:
                            pass
                    
                    # Store annotation trong map để inject vào text sau
                    if img_id:
                        image_annotations_map[img_id] = bbox_annotation
                
                img_info = {
                    "index": len(result["images"]),
                    "id": img_id,
                    "bbox": img.get("bbox"),
                    "base64": img.get("image_base64"),
                    "annotation": bbox_annotation,
                }
                result["images"].append(img_info)
    
    # Combine markdown
    full_text = "\n\n".join(markdown_parts)
    
    # Inject image server URLs for ALL images (not just ones with annotations)
    import urllib.parse
    for img_info in result["images"]:
        img_id = img_info.get("id")
        if not img_id:
            continue
            
        # Placeholder format: ![img-0.jpeg](img-0.jpeg)
        old_placeholder = f"![{img_id}]({img_id})"
        
        # Build new URL with image server
        new_url = img_id  # Default: keep original
        if image_server_base_url and project_path and pdf_filename:
            try:
                img_index = img_id.split("-")[1].split(".")[0]  # "img-0.jpeg" -> "0"
                images_folder = f"{pdf_filename}_images"
                image_filename = f"image_{img_index}.png"
                # URL encode the path
                encoded_path = urllib.parse.quote(f"{project_path}/{images_folder}/{image_filename}")
                new_url = f"{image_server_base_url}/images/{encoded_path}"
            except Exception as e:
                print(f"[DEBUG] Error building image URL for {img_id}: {e}")
        
        new_placeholder = f"![{img_id}]({new_url})"
        
        # Add annotation text if exists
        annotation_text = ""
        annotation = image_annotations_map.get(img_id)
        if annotation and isinstance(annotation, dict) and "summary" in annotation:
            annotation_text = f"\n{annotation['summary']}"
        
        # Replace old placeholder with new placeholder + annotation
        full_text = full_text.replace(old_placeholder, new_placeholder + annotation_text)
    
    result["text"] = full_text
    
    # Extract annotations nếu có
    if "document_annotation" in ocr_response:
        annotation_data = ocr_response["document_annotation"]
        # Nếu annotation là string JSON, parse nó thành dict
        if isinstance(annotation_data, str):
            try:
                result["annotations"] = json.loads(annotation_data)
            except:
                result["annotations"] = annotation_data
        else:
            result["annotations"] = annotation_data
    
    return result


def load_pdf(
    file_path: Path,
    *,
    use_mistral: bool = True,
    mistral_public_url: Optional[str] = None,  # Nếu có URL công khai của PDF
    include_images: bool = True,  # Mặc định bật để lấy ảnh
    table_format: str = "markdown",
    extract_header: bool = False,
    extract_footer: bool = False,
    bbox_annotation_format: Optional[Any] = None,
    document_annotation_format: Optional[Any] = None,
    image_server_base_url: Optional[str] = None,  # Base URL image server
    project_path: Optional[str] = None,  # Project path trong data/
) -> Optional[Dict[str, Any]]:
    """
    Đọc PDF, ưu tiên dùng Mistral AI OCR nếu có, fallback về pypdf.
    
    Args:
        file_path: Đường dẫn file PDF (local)
        use_mistral: Ưu tiên dùng Mistral AI (default: True)
        mistral_public_url: URL công khai của PDF (nếu có, sẽ dùng thay vì local file)
        include_images: Lấy ảnh base64 từ PDF (default: True)
        table_format: Format bảng ("markdown" hoặc "html", default: "markdown")
        extract_header: Tách header riêng (default: False)
        extract_footer: Tách footer riêng (default: False)
        bbox_annotation_format: Format cho bbox annotation (Pydantic model hoặc dict)
        document_annotation_format: Format cho document annotation (Pydantic model hoặc dict)
        
    Returns:
        Dict với keys: 'text', 'images', 'annotations' hoặc None nếu lỗi
        Nếu dùng pypdf fallback, chỉ có 'text', 'images' và 'annotations' sẽ là empty list/None
    """
    if not file_path.exists():
        return None
    
    # Thử Mistral AI OCR trước (nếu có API key và requests)
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if use_mistral and mistral_api_key and requests:
        try:
            # Theo document: có thể dùng public URL, base64 encoded PDF, hoặc uploaded PDF
            # Ưu tiên: public URL > base64 > fallback pypdf
            document_payload = None
            
            if mistral_public_url:
                # Option 1: Public URL (theo document line 37, 40-46)
                document_payload = {
                    "type": "document_url",
                    "document_url": mistral_public_url,
                }
                print(f"[PDF] Using Mistral OCR with public URL: {mistral_public_url[:80]}...")
            else:
                # Option 2: Base64 encoded PDF (theo document line 37, 43)
                pdf_base64 = _pdf_to_base64(file_path)
                if pdf_base64:
                    # Theo document, base64 PDF có thể dùng data URI trong document_url
                    # Thử format data URI (theo pattern của image base64 trong document)
                    document_payload = {
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{pdf_base64}",
                    }
                    print(f"[PDF] Using Mistral OCR with base64 encoded PDF (data URI format)")
                else:
                    print(f"[PDF] Cannot encode {file_path.name} to base64")
                    raise ValueError("Cannot create base64 from PDF")
            
            # Gọi Mistral OCR API qua HTTP (theo document line 89-100)
            api_url = "https://api.mistral.ai/v1/ocr"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {mistral_api_key}",
            }
            
            ocr_payload = {
                "model": "mistral-ocr-latest",
                "document": document_payload,
                "table_format": table_format,
                "include_image_base64": include_images,
            }
            
            if extract_header:
                ocr_payload["extract_header"] = True
            if extract_footer:
                ocr_payload["extract_footer"] = True
            
            # Convert annotation formats to proper JSON schema format
            # Theo doc (line 463-481): phải wrap trong {"type": "json_schema", "json_schema": {...}}
            if bbox_annotation_format:
                if isinstance(bbox_annotation_format, dict):
                    # Nếu đã là dict, wrap trong json_schema format
                    ocr_payload["bbox_annotation_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "schema": bbox_annotation_format,
                            "name": "bbox_annotation",
                            "strict": True
                        }
                    }
                else:
                    # Nếu là Pydantic model hoặc dạng khác, pass trực tiếp
                    # (SDK sẽ tự convert)
                    ocr_payload["bbox_annotation_format"] = bbox_annotation_format
            
            if document_annotation_format:
                if isinstance(document_annotation_format, dict):
                    # Nếu đã là dict, wrap trong json_schema format
                    ocr_payload["document_annotation_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "schema": document_annotation_format,
                            "name": "document_annotation",
                            "strict": True
                        }
                    }
                else:
                    # Nếu là Pydantic model hoặc dạng khác, pass trực tiếp
                    ocr_payload["document_annotation_format"] = document_annotation_format
            
            print(f"[PDF] Processing {file_path.name} with Mistral OCR (HTTP API)...")
            if mistral_public_url:
                print(f"[PDF] Using document URL: {mistral_public_url[:80]}...")
            else:
                print(f"[PDF] Using base64 encoded PDF ({len(pdf_base64)} chars)")
            
            response = requests.post(api_url, json=ocr_payload, headers=headers, timeout=120)
            
            # Xử lý lỗi chi tiết hơn
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = str(error_json)
                except:
                    pass
                error_msg = f"[PDF] Mistral API error {response.status_code}: {error_detail[:500]}"
                print(error_msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {error_detail[:500]}")
            
            ocr_response = response.json()
            
            # Extract content with image server URL params
            content = _extract_mistral_content(
                ocr_response,
                image_server_base_url=image_server_base_url,
                project_path=project_path,
                pdf_filename=file_path.stem,  # Tên file không có .pdf
            )
            
            if content["text"]:
                print(f"[PDF] ✓ Extracted {len(content['text'])} chars from {file_path.name}")
                if content["images"]:
                    print(f"[PDF] ✓ Found {len(content['images'])} images")
                if content["annotations"]:
                    print(f"[PDF] ✓ Found annotations")
                return content  # Return full dict với text, images, annotations
            else:
                print(f"[PDF] No text extracted from {file_path.name}, fallback to pypdf")
        
        except ValueError as e:
            # ValueError là do không có public URL, fallback ngay
            pass
        except Exception as e:
            print(f"[PDF] Mistral OCR error for {file_path.name}: {e}, fallback to pypdf")
    
    # Fallback: dùng pypdf (chỉ có text, không có images/annotations)
    if PdfReader:
        try:
            reader = PdfReader(str(file_path))
            pages_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            result_text = "\n\n".join(pages_text) if pages_text else None
            if result_text:
                print(f"[PDF] ✓ Extracted text using pypdf from {file_path.name}")
                # Return dict format tương tự Mistral OCR
                return {
                    "text": result_text,
                    "images": [],  # pypdf không extract images
                    "annotations": None,  # pypdf không có annotations
                }
            return None
        except Exception as e:
            print(f"[PDF] Error reading {file_path.name} with pypdf: {e}")
            return None
    
    print(f"[PDF] No PDF reader available (pypdf not installed)")
    return None


if __name__ == "__main__":
    """Test script để chạy riêng pdf.py"""
    import sys
    import io
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Fix encoding cho Windows console
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Load .env từ root project
    root_dir = Path(__file__).parent.parent  # src/pdf.py -> src -> RAG_Chat
    env_file = root_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    # Lấy file PDF từ command line hoặc dùng file mặc định
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        # Tìm file PDF đầu tiên trong data/
        data_dir = root_dir / "data"
        pdf_files = list(data_dir.rglob("*.pdf"))
        if pdf_files:
            pdf_path = pdf_files[0]
            print(f"[TEST] Using first PDF found: {pdf_path}")
        else:
            print("[TEST] Error: No PDF file provided and no PDF found in data/")
            print("[TEST] Usage: python pdf.py <path_to_pdf>")
            sys.exit(1)
    
    if not pdf_path.exists():
        print(f"[TEST] Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"[TEST] Testing PDF: {pdf_path}")
    print(f"[TEST] File size: {pdf_path.stat().st_size / 1024:.2f} KB")
    print("-" * 60)
    
    # Test với Mistral AI (nếu có key)
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if mistral_key:
        print("[TEST] Testing with Mistral AI OCR...")
        print("-" * 60)
        
        # Define annotation schemas theo document (line 318-411)
        # BBox annotation format - cho các ảnh/figure trong PDF
        bbox_schema = {
            "type": "object",
            "properties": {
                "image_type": {
                    "type": "string",
                    "description": "The type of the image (e.g., chart, diagram, photo)"
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
        
        # Document annotation format - cho toàn bộ document
        doc_schema = {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "description": "The type of document (e.g., invoice, receipt, contract)"
                },
                "main_topic": {
                    "type": "string",
                    "description": "The main topic or subject of the document"
                },
                "key_information": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key information extracted from document"
                }
            },
            "required": ["document_type", "main_topic"],
            "additionalProperties": False
        }
        
        # Image server configuration
        image_server_url = os.getenv("IMAGE_SERVER_URL")
        # Get project path relative to data/ directory
        data_dir = root_dir / "data"
        try:
            project_relative = pdf_path.parent.relative_to(data_dir)
        except ValueError:
            # If PDF is not in data/, don't use project path
            project_relative = None
        
        result = load_pdf(
            pdf_path,
            use_mistral=True,
            include_images=True,  # Bật để lấy ảnh
            extract_header=False,
            extract_footer=False,
            bbox_annotation_format=bbox_schema,  # Enable bbox annotation
            document_annotation_format=doc_schema,  # Enable document annotation
            image_server_base_url=image_server_url,  # Inject image server URL
            project_path=str(project_relative) if project_relative else None,
        )
        
        if result and result.get("text"):
            text = result["text"]
            images = result.get("images", [])
            annotations = result.get("annotations")
            
            print(f"\n[TEST] ✓ Success! Extracted {len(text)} characters")
            print(f"[TEST] Found {len(images)} images with URLs injected")
            if annotations:
                print(f"[TEST] Found document annotations")
            print(f"\n[TEST] Preview (first 800 chars):\n")
            print(text[:800])
            if len(text) > 800:
                print("\n... (truncated)")
            
            print(f"\n[TEST] Result returned as dict with keys: {list(result.keys())}")
            print(f"[TEST] Ready for importDB to process and chunk!")
        else:
            print("[TEST] ✗ Failed to extract text with Mistral")
    else:
        print("[TEST] MISTRAL_API_KEY not found, testing with pypdf fallback...")
        print("-" * 60)
        result = load_pdf(pdf_path, use_mistral=False)
        
        if result and result.get("text"):
            text = result["text"]
            print(f"\n[TEST] ✓ Success! Extracted {len(text)} characters")
            print(f"[TEST] Preview (first 500 chars):\n")
            print(text[:500])
            if len(text) > 500:
                print("\n... (truncated)")
            
            # Lưu ra file txt
            output_file = pdf_path.with_suffix('.txt')
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"\n[TEST] ✓ Saved full text to: {output_file}")
            except Exception as e:
                print(f"\n[TEST] ✗ Error saving to file: {e}")
        else:
            print("[TEST] ✗ Failed to extract text")
    
    print("\n" + "=" * 60)
    print("[TEST] Done!")
