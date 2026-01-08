#!/usr/bin/env python3
"""
Sync images from Supabase and build ORB feature cache.
Run this to download existing images and enable ORB image search.
"""

import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from src.orb_image_search import get_orb_image_search

def main():
    print("=" * 60)
    print("ORB IMAGE SYNC")
    print("=" * 60)
    
    searcher = get_orb_image_search()
    
    if searcher is None:
        print("❌ Failed to initialize ORB search. Check Supabase credentials.")
        return 1
    
    print(f"\n📥 Downloading images from Supabase bucket...")
    downloaded = searcher.sync_from_supabase()
    print(f"   Downloaded: {downloaded} new images")
    
    print(f"\n📂 Building ORB cache from local ./image_cache/ folder...")
    new_count = searcher.build_cache_from_local()
    
    print(f"\n✅ Cache ready!")
    print(f"   New features computed: {new_count}")
    print(f"   Total cached: {len(searcher.features_cache)}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())
