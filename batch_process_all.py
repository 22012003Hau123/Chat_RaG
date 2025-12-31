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
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

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
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

try:
    logger.info("="*80)
    logger.info("PHASE 1: DISCOVERY")
    logger.info("="*80)
    
    # Find all supported files
    data_dir = Path("data")
    supported_extensions = ['.pdf', '.docx', '.pptx']
    
    files_to_process = []
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
            
        # Check for storage path arg
        if '--storage-path' in sys.argv:
            path_index = sys.argv.index('--storage-path') + 1
            if path_index < len(sys.argv):
                single_storage_path = sys.argv[path_index]
                logger.info(f"🔗 Linking to storage path: {single_storage_path}")

    # ... (skipping some lines) ...

        # Create documents
        doc_annotation = extraction_data.get('document_annotation', {})
        for i, chunk in enumerate(chunks):
            metadata = {
                'source': extraction_data['pdf_name'],
                'chunk_index': i,
                'total_chunks': len(chunks),
                'document_type': doc_annotation.get('document_type', 'unknown'),
                'main_topic': doc_annotation.get('main_topic', ''),
                'extraction_timestamp': timestamp
            }
            
            # Add storage path if available (for single file mode)
            if single_storage_path:
                metadata['storage_path'] = single_storage_path
                
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            all_documents.append(doc)
        
        # Generate embeddings with OpenAI
        chunk_texts = [chunk for chunk in chunks]
        
        response = openai_client.embeddings.create(
            model=embedding_model,
            input=chunk_texts
        )
        chunk_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(chunk_embeddings)
        
        logger.info(f"  ✓ Generated {len(chunks)} embeddings")
    
    logger.info(f"\n📊 Total: {len(all_documents)} chunks, {len(all_embeddings)} embeddings")
    
    # Insert to Supabase
    logger.info("\n💾 Inserting to Supabase...")
    embeddings_array = np.array(all_embeddings)
    
    inserted = supabase.insert_documents(all_documents, embeddings_array)
    logger.info(f"✓ Inserted {inserted} documents to Supabase")
    
    # Verify
    logger.info("\nVerifying database...")
    total_docs = supabase.get_document_count()
    logger.info(f"✓ Total documents in database: {total_docs}")
    
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
