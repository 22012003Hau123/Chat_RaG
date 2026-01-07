#!/usr/bin/env python3
"""
Single file ingestion - wrapper around batch_process_all.py logic
Processes one document at a time for upload feature
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any

def ingest_single_file(filepath: str, file_url: str = None, progress_callback=None, storage_path: str = None) -> Dict[str, Any]:
    """
    Ingest a single document file using batch_process_all.py logic
    
    Args:
        filepath: Path to the document file
        file_url: Public URL of the file in Supabase Storage (optional)
        progress_callback: Optional function to call with progress updates
        storage_path: Optional override for the storage path/filename (crucial for temp files)
        
    Returns:
        Dict with ingestion results
    """
    filename = Path(filepath).name
    # If storage_path is provided, use it for display name too
    if storage_path:
        display_name = storage_path
    else:
        display_name = filename
    
    def update_progress(message: str, percent: int = None):
        if progress_callback:
            progress_callback(message, percent)
        print(f"[{percent}%] {message}" if percent else message)
    
    try:
        update_progress(f"📄 Processing {display_name}...", 0)
        
        # Run batch_process_all.py on single file
        # This reuses all existing logic: Mistral OCR, embedding, Supabase insertion
        # Prepare command
        cmd = [sys.executable, 'batch_process_all.py', '--single-file', filepath]
        
        # Add storage path if provided explicitly OR via URL
        final_storage_path = None
        
        if storage_path:
            final_storage_path = storage_path
        elif file_url:
            # Extract relative path from URL if possible
            # Example: https://.../storage/v1/object/public/source-documents/file.pdf -> file.pdf
            if '/storage/v1/object/public/' in file_url:
                extracted = file_url.split('/storage/v1/object/public/')[-1]
                # Remove bucket name if present (assuming 'source-documents')
                if extracted.startswith('source-documents/'):
                    extracted = extracted.replace('source-documents/', '', 1)
                final_storage_path = extracted
            else:
                # Fallback: use filename
                final_storage_path = filename
                
        if final_storage_path:
            cmd.extend(['--storage-path', final_storage_path])
            update_progress(f"🔗 Linking to storage: {final_storage_path}", 5)
            
        # Run batch_process_all.py on single file
        # This reuses all existing logic: Mistral OCR, embedding, Supabase insertion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            update_progress(f"✅ Successfully processed {filename}!", 100)
            return {
                'success': True,
                'filename': filename,
                'file_url': file_url,
                'output': result.stdout
            }
        else:
            error_msg = result.stderr or result.stdout
            update_progress(f"❌ Error: {error_msg}", -1)
            return {
                'success': False,
                'filename': filename,
                'error': error_msg
            }
            
    except Exception as e:
        error_msg = f"❌ Error processing {filename}: {str(e)}"
        update_progress(error_msg, -1)
        return {
            'success': False,
            'filename': filename,
            'error': str(e)
        }


class DocumentIngester:
    """Simple wrapper for compatibility with app.py"""
    
    def ingest_file(self, filepath: str, file_url: str = None, progress_callback=None) -> Dict[str, Any]:
        return ingest_single_file(filepath, file_url, progress_callback)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest a single file.')
    parser.add_argument('filepath', nargs='?', help='Path to the file to ingest')
    parser.add_argument('--file', dest='file_arg', help='Path to the file (alternative flag)')
    parser.add_argument('--storage-path', help='Original filename/path for metadata (optional)')
    
    args = parser.parse_args()
    
    # Handle both positional and flag arguments
    filepath = args.file_arg if args.file_arg else args.filepath
    
    if not filepath:
        print("Usage: python ingest_single_file.py <filepath> OR --file <filepath>")
        sys.exit(1)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    print("="*80)
    print("SINGLE FILE INGESTION")
    print("="*80)
    
    result = ingest_single_file(filepath, storage_path=args.storage_path)
    
    print("\n" + "="*80)
    if result['success']:
        print("✅ SUCCESS")
        print(f"Filename: {result['filename']}")
        print("\n--- Raw Output ---")
        print(result.get('output', 'No output captured'))
        print("------------------")
    else:
        print("❌ FAILED")
        print(f"Error: {result['error']}")
    print("="*80)
