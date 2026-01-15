#!/usr/bin/env python3
"""
Sync images from Supabase and build CLIP embeddings cache.
Run this to download existing images and enable CLIP image search.
"""

import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.clip_image_search import get_clip_image_search

def main():
    print("=" * 60)
    print("CLIP IMAGE SYNC")
    print("=" * 60)
    
    searcher = get_clip_image_search()
    
    if not searcher:
        print("❌ Failed to initialize CLIP search. Check Supabase credentials.")
        sys.exit(1)
    
    print("\n📥 Downloading images from Supabase...")
    downloaded = searcher.sync_from_supabase()
    print(f"✓ Downloaded {downloaded} new images")
    
    print(f"\n📂 Building CLIP cache from local ./image_cache/ folder...")
    new_count = searcher.build_cache_from_local()
    print(f"✓ Generated {new_count} new embeddings")
    
    print(f"\n✅ Total cached: {len(searcher.embeddings)} images")
    print("=" * 60)

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    main()
