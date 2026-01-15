#!/usr/bin/env python3
"""
Batch Process All Documents - Full RAG Pipeline

Process tất cả documents trong data/:
1. Extract với Mistral OCR (text + images + annotations)
2. Upload images to Supabase Storage
3. Embed text chunks
4. Insert to Supabase database

Usage:
    python batch_process_all.py
    python batch_process_all.py --dry-run  # Preview only
"""

import os
import sys
import json
import logging
import base64
import numpy as np
import io
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Imports for processing
from src.pdf import load_pdf
from src.supabase_client import SupabaseClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import openai
from src.document_extractor import DocumentExtractor


load_dotenv()

print("="*80)
print("BATCH PROCESSING - FULL RAG PIPELINE")
print("="*80)

# Create output directory
output_dir = Path("batch_outputs")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = output_dir / f"batch_process_{timestamp}.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize components
try:
    # Initialize Document Extractor for DOCX/PPTX
    doc_extractor = DocumentExtractor()
    
    # Initialize Mistral Annotator for DOCX/PPTX images
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    mistral_annotator = MistralImageSearch(mistral_api_key) if mistral_api_key else None
    
    openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    supabase = SupabaseClient()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    embedding_model = "text-embedding-3-small"
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    sys.exit(1)

try:
    logger.info("="*80)
    logger.info("PHASE 1: DISCOVERY")
    logger.info("="*80)
    
    # Find all supported files
    data_dir = Path("data")
    supported_extensions = ['.pdf', '.docx', '.pptx']
    
    files_to_process = []
    # Check if data dir exists
    if data_dir.exists():
        for ext in supported_extensions:
            files_to_process.extend(data_dir.glob(f"*{ext}"))
    
    files_to_process = sorted(files_to_process)
    
    logger.info(f"\n📁 Found {len(files_to_process)} files to process:")
    total_size = 0
    for f in files_to_process:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        logger.info(f"  • {f.name} ({size_mb:.1f} MB)")
    
    logger.info(f"\n📊 Total size: {total_size:.1f} MB")
    
    # Check for single-file mode
    single_file_mode = '--single-file' in sys.argv
    single_storage_path = None
    
    if single_file_mode:
        # Get file path from args
        try:
            file_arg_index = sys.argv.index('--single-file') + 1
            if file_arg_index < len(sys.argv):
                single_file_path = Path(sys.argv[file_arg_index])
                if single_file_path.exists():
                    files_to_process = [single_file_path]
                    logger.info(f"\n🎯 SINGLE FILE MODE: Processing {single_file_path.name}")
                else:
                    logger.error(f"File not found: {single_file_path}")
                    sys.exit(1)
            else:
                logger.error("--single-file requires a file path argument")
                sys.exit(1)
                
            # Check for storage path arg (CRITICAL for fixing temporary filename issue)
            if '--storage-path' in sys.argv:
                path_index = sys.argv.index('--storage-path') + 1
                if path_index < len(sys.argv):
                    single_storage_path = sys.argv[path_index]
                    logger.info(f"🔗 Linking to storage path: {single_storage_path}")
        except ValueError:
            pass

    if '--dry-run' in sys.argv:
        logger.info("Dry run complete. Exiting.")
        sys.exit(0)

    # ---------------------------------------------------------
    # Processing Loop
    # ---------------------------------------------------------
    all_documents = []
    all_embeddings = []
    extraction_results = []
    
    for f in files_to_process:
        logger.info(f"\nPROCESSING: {f.name}")
        
        # Determine extraction method based on type
        extraction_data = None
        ext = f.suffix.lower()
        
        # Define annotation schemas for Mistral OCR (used by both PDF and DOCX)
        doc_schema = {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "description": "The type of document (e.g., invoice, receipt, contract, article, slide)"
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
        
        bbox_schema = {
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
        
        try:
            if ext == '.pdf':
                # Use Mistral-enhanced PDF loader
                extraction_data = load_pdf(
                    f, 
                    use_mistral=True,
                    include_images=True,
                    bbox_annotation_format=bbox_schema,
                    document_annotation_format=doc_schema
                )
            elif ext == '.docx':
                # Handle DOCX with Mistral OCR (same as PDF for proper image positioning)
                from src.docx import load_docx
                
                logger.info(f"  Processing DOCX with Mistral OCR...")
                extraction_data = load_docx(
                    f,
                    use_mistral=True,
                    include_images=True,
                    bbox_annotation_format=bbox_schema,
                    document_annotation_format=doc_schema
                )
            elif ext == '.pptx':
                # Handle PPTX with DocumentExtractor (works well for slide context)
                logger.info(f"  Extracting content from PPTX file...")
                extracted_result = doc_extractor.extract_from_file(str(f))
                
                raw_text = extracted_result.get("text", "")
                extracted_images = extracted_result.get("images", [])
                
                processed_images = []
                
                # Annotate extracted images using Mistral
                if extracted_images and mistral_annotator:
                    logger.info(f"  Found {len(extracted_images)} images. Annotating with Mistral...")
                    for i, img in enumerate(extracted_images):
                        try:
                            # img is dict: {'index': 0, 'name': '...', 'bytes': b'...', 'ext': '...'}
                            logger.info(f"  Analysing image {i+1}/{len(extracted_images)}: {img['name']}")
                            
                            # Annotate using Mistral (passing bytes directly)
                            summary = mistral_annotator.annotate_image(image_bytes=img['bytes'])
                            
                            # Construct annotation object matching PDF schema
                            annotation = {
                                "image_type": "extracted_image",
                                "short_description": f"Image extracted from {f.name}",
                                "summary": summary if summary else "No description available"
                            }
                            
                            # Convert bytes to base64 for compatibility with upload loop
                            img_b64 = base64.b64encode(img['bytes']).decode('utf-8')
                            
                            # Construct image object
                            processed_images.append({
                                "id": img['name'],
                                "base64": img_b64,
                                "annotation": annotation,
                                "index": i,
                                "page_index": img.get('page', 0)
                            })
                            
                        except Exception as ann_err:
                            logger.error(f"  Failed to annotate image {img.get('name')}: {ann_err}")
                
                extraction_data = {
                    'text': raw_text,
                    'images': processed_images,
                    'annotations': {
                        'document_type': 'presentation',
                        'main_topic': f.stem
                    }
                }
            else:
                logger.warning(f"Skipping {f.name} (Unsupported format for batch processing)")
                continue
                
            if not extraction_data:
                logger.error(f"Failed to extract content from {f.name}")
                continue
            
            extraction_results.append(extraction_data)
            
            # --- UPLOAD EXTRACTED IMAGES ---
            # extraction_data['images'] contains list of:
            # { "id": "img-0", "base64": "...", "bbox": [...], "page": 0, ... }
            if 'images' in extraction_data and extraction_data['images']:
                images = extraction_data['images']
                logger.info(f"  Found {len(images)} images. Uploading to storage...")
                
                for img_data in images:
                    try:
                        if 'base64' not in img_data: continue
                        
                        img_b64 = img_data['base64']
                        # Sanitize Base64: Remove data URI prefix if present
                        if ',' in img_b64 and 'data:' in img_b64[:50]:
                            img_b64 = img_b64.split(',', 1)[1]
                        
                        # Decode image
                        raw_bytes = base64.b64decode(img_b64)

                        
                        # CONVERT TO PNG using Pillow
                        # This ensures consistent format (fixes potential JP2/raw issues from Mistral)
                        try:
                            image = Image.open(io.BytesIO(raw_bytes))
                            output_buffer = io.BytesIO()
                            # Convert to RGB if needed (e.g. if CMYK or RGBA issues, though PNG handles RGBA)
                            if image.mode in ('CMYK', 'P'):
                                image = image.convert('RGB')
                            image.save(output_buffer, format='PNG')
                            img_bytes = output_buffer.getvalue()
                            logger.info(f"  ✓ Converted image {img_data.get('id')} to PNG successfully")
                        except Exception as e:
                            logger.error(f"  Failed to convert image {img_data.get('id')} to PNG: {e}")
                            # Fallback to raw bytes if conversion fails (risky but better than crashing)
                            img_bytes = raw_bytes
                        
                        # Create filename: {pdf_name}_page{X}_img{Y}.png
                        # Use original filename (single_storage_path) if available, not temp filename
                        original_name = single_storage_path if single_storage_path else f.name
                        # Remove extension but KEEP spaces/characters as requested
                        # Example: "1_TRACT_PDF BD.pdf" -> "1_TRACT_PDF BD"
                        base_name = Path(original_name).stem
                        
                        # Get page index and image index for proper naming
                        page_idx = img_data.get('page_index', 0)
                        img_idx = img_data.get('index', 0)
                        
                        # Format: {filename}_page{X}_img{Y}.png (1-based page numbering)
                        # Example: 1_TRACT_PDF BD_page1_img0.png
                        img_filename = f"{base_name}_page{page_idx + 1}_img{img_idx}.png"


                        # Upload to 'alpagino' bucket
                        img_url = supabase.upload_image(
                            img_bytes, 
                            img_filename, 
                            bucket='alpagino'
                        )
                        
                        if img_url:
                            img_data['public_url'] = img_url
                            
                            # AUTO-CACHE: Save locally for CLIP embeddings
                            try:
                                from pathlib import Path
                                cache_dir = Path("./image_cache")
                                cache_dir.mkdir(exist_ok=True)
                                local_path = cache_dir / img_filename
                                with open(local_path, 'wb') as cache_file:
                                    cache_file.write(img_bytes)
                                logger.info(f"    → Cached locally: {img_filename}")
                            except Exception as cache_err:
                                logger.warning(f"    → Failed to cache locally: {cache_err}")
                    except Exception as e:
                        logger.error(f"  Failed to upload image {img_data.get('id')}: {e}")
                        
                logger.info(f"  ✓ Uploaded extracted images to 'alpagino' bucket")

            # --- INJECT SUPABASE URLS INTO TEXT ---
            # Replace placeholder ![img-0.jpeg](img-0.jpeg) with [View: public_url]
            # This needs to run after images are uploaded and public_urls are available
            if 'images' in extraction_data and extraction_data['images']:
                images = extraction_data['images']
                text = extraction_data.get('text', '')
                for img_data in images:
                    if 'public_url' in img_data and img_data.get('id'):
                        img_id = img_data['id']
                        public_url = img_data['public_url']
                        
                        # Build annotation text from bbox data
                        annotation = img_data.get('annotation', {})
                        img_type = annotation.get('image_type', 'image') if annotation else 'image'
                        description = annotation.get('summary', '') if annotation else ''
                        
                        # Create new format: [IMAGE: type]\nDescription: ...\n[View: url]
                        new_block = f"\n\n[IMAGE: {img_type}]"
                        if description:
                            new_block += f"\nDescription: {description}"
                        new_block += f"\n[View: {public_url}]\n"
                        
                        # Replace old placeholder
                        old_placeholder = f"![{img_id}]({img_id})"
                        text = text.replace(old_placeholder, new_block)
                        
                extraction_data['text'] = text
                logger.info(f"  ✓ Injected image URLs into text")

            # --- UPLOAD SOURCE FILE ---
            try:
                # Read file content
                f_bytes = f.read_bytes()
                
                # Determine storage path
                # If single_storage_path is set (from arguments), use it to store proper filename
                storage_filename = single_storage_path if single_storage_path else f.name
                
                logger.info(f"  Uploading source file to 'source-documents' bucket: {storage_filename}")
                src_url = supabase.upload_image(
                    f_bytes,
                    storage_filename,
                    bucket='source-documents'
                )
                logger.info(f"  ✓ Uploaded source file: {src_url}")
            except Exception as e:
                logger.error(f"  Failed to upload source file {f.name}: {e}")

            # Split text into chunks
            text = extraction_data.get('text', '')
            if not text:
                logger.warning(f"No text extracted from {f.name}")
                continue
                
            chunks = text_splitter.split_text(text)
            logger.info(f"  ✓ Split into {len(chunks)} chunks")
            
            # Create Document objects with metadata
            # NOTE: load_pdf returns 'annotations' key, not 'document_annotation'
            doc_annotation = extraction_data.get('annotations', {}) or {}
            
            # Determine correct source name for Metadata
            final_source_name = single_storage_path if single_storage_path else f.name
            
            file_documents = []
            for i, chunk_text in enumerate(chunks):
                metadata = {
                    'source': final_source_name, # Correct filename
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'document_type': doc_annotation.get('document_type', 'unknown') if doc_annotation else 'unknown',
                    'main_topic': doc_annotation.get('main_topic', '') if doc_annotation else '',
                    'extraction_timestamp': timestamp
                }
                
                # Add storage path if available
                if single_storage_path:
                    metadata['storage_path'] = single_storage_path
                
                doc = Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
                file_documents.append(doc)
            
            all_documents.extend(file_documents)
            
            # Generate embeddings for this file immediately
            if file_documents:
                # DEDUPLICATION: Remove existing documents for this source before inserting new ones
                supabase.delete_documents_by_source(final_source_name)
                
                chunk_texts = [d.page_content for d in file_documents]
                response = openai_client.embeddings.create(
                    model=embedding_model,
                    input=chunk_texts
                )
                file_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(file_embeddings)
                logger.info(f"  ✓ Generated {len(file_embeddings)} embeddings")

        except Exception as e:
            logger.error(f"Error processing {f.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    # ---------------------------------------------------------
    # Insertion Phase
    # ---------------------------------------------------------
    if all_documents:
        logger.info(f"\n📊 Total: {len(all_documents)} chunks, {len(all_embeddings)} embeddings")
        
        logger.info("\n💾 Inserting to Supabase...")
        # Convert list of lists to numpy array for SupabaseClient
        embeddings_array = np.array(all_embeddings)
        
        inserted = supabase.insert_documents(all_documents, embeddings_array)
        logger.info(f"✓ Inserted {inserted} documents to Supabase")
        
        # Verify
        logger.info("\nVerifying database...")
        total_docs = supabase.get_document_count()
        logger.info(f"✓ Total documents in database: {total_docs}")
    else:
        logger.warning("\n⚠️ No documents to insert.")
    
    logger.info("\n" + "="*80)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info(f"\n📊 Summary:")
    logger.info(f"  • Files processed: {len(extraction_results)}")
    logger.info(f"  • Images uploaded: {sum(len(e.get('images', [])) for e in extraction_results)}")
    logger.info(f"  • Documents embedded: {len(all_documents)}")
    logger.info(f"  • Database total: {total_docs}")
    logger.info(f"\n📁 Output directory: {output_dir}/")
    logger.info(f"📄 Log file: {log_file}")
    logger.info("="*80)

except Exception as e:
    logger.error(f"\n❌ PIPELINE ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
