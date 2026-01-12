"""
ORB Image Search

Uses OpenCV ORB feature matching to find similar images.
Caches ORB features locally from ./image_cache/ folder.
"""

import os
import cv2
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class ORBImageSearch:
    """ORB-based image similarity search."""
    
    def __init__(self, supabase_url: str, supabase_key: str, bucket_name: str = "alpagino"):
        """
        Initialize ORB search.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            bucket_name: Storage bucket name
        """
        from supabase import create_client
        self.supabase = create_client(supabase_url, supabase_key)
        self.bucket_name = bucket_name
        
        # Local cache paths
        self.cache_dir = Path("./image_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.features_file = self.cache_dir / "orb_features.pkl"
        
        # ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Feature matcher (BFMatcher with Hamming distance for ORB)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Cached features: {filename: descriptors}
        self.features_cache: Dict[str, np.ndarray] = {}
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"🔍 ORB Image Search initialized. Cached: {len(self.features_cache)} images")
    
    def _load_cache(self):
        """Load cached ORB features from disk."""
        if self.features_file.exists():
            try:
                with open(self.features_file, 'rb') as f:
                    self.features_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.features_cache)} cached features")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.features_cache = {}
    
    def _save_cache(self):
        """Save ORB features to disk."""
        try:
            with open(self.features_file, 'wb') as f:
                pickle.dump(self.features_cache, f)
            logger.info(f"Saved {len(self.features_cache)} features to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _compute_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Compute ORB features for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ORB descriptors or None if failed
        """
        try:
            # Read image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not read image: {image_path}")
                return None
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.orb.detectAndCompute(img, None)
            
            if descriptors is None or len(descriptors) < 10:
                logger.warning(f"Not enough features in: {image_path}")
                return None
            
            return descriptors
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return None
    
    def sync_from_supabase(self) -> int:
        """
        Download all images from Supabase bucket to local cache.
        
        Returns:
            Number of images downloaded
        """
        import requests
        
        logger.info(f"📥 Syncing images from Supabase bucket: {self.bucket_name}")
        
        try:
            # List all files in bucket
            files = self.supabase.storage.from_(self.bucket_name).list()
            
            # Filter image files
            image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
            image_files = [f for f in files if f['name'].lower().endswith(image_extensions)]
            
            logger.info(f"Found {len(image_files)} images in bucket")
            
            downloaded = 0
            updated = 0
            for file_info in image_files:
                filename = file_info['name']
                local_path = self.cache_dir / filename
                remote_size = file_info.get('metadata', {}).get('size', 0)
                
                # Check if need to download
                need_download = False
                if not local_path.exists():
                    need_download = True
                elif remote_size > 0 and local_path.stat().st_size != remote_size:
                    # File size changed - need to re-download
                    need_download = True
                    updated += 1
                    logger.info(f"↻ File changed: {filename}")
                
                if not need_download:
                    continue
                
                try:
                    # Get public URL and download
                    url = self.supabase.storage.from_(self.bucket_name).get_public_url(filename)
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        with open(local_path, 'wb') as f:
                            f.write(response.content)
                        downloaded += 1
                        
                        # Invalidate cached features for this file
                        if filename in self.features_cache:
                            del self.features_cache[filename]
                            logger.info(f"✓ Re-downloaded & invalidated: {filename}")
                        else:
                            logger.info(f"✓ Downloaded: {filename}")
                    else:
                        logger.warning(f"Failed to download: {filename}")
                        
                except Exception as e:
                    logger.error(f"Error downloading {filename}: {e}")
                    continue
            
            logger.info(f"✅ Downloaded {downloaded} images from Supabase")
            return downloaded
            
        except Exception as e:
            logger.error(f"Error syncing from Supabase: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def build_cache_from_local(self) -> int:
        """
        Build ORB features cache from local ./image_cache/ folder.
        
        Returns:
            Number of images processed
        """
        logger.info(f"📂 Building ORB cache from: {self.cache_dir}")
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        new_count = 0
        
        for img_file in self.cache_dir.iterdir():
            if not img_file.suffix.lower() in image_extensions:
                continue
            
            filename = img_file.name
            
            # Skip if already cached
            if filename in self.features_cache:
                continue
            
            # Compute ORB features
            features = self._compute_features(str(img_file))
            if features is not None:
                self.features_cache[filename] = features
                new_count += 1
                logger.info(f"✓ Computed features: {filename}")
        
        # Save updated cache
        if new_count > 0:
            self._save_cache()
        
        logger.info(f"✅ Cache built. New: {new_count}, Total: {len(self.features_cache)}")
        return new_count
    
    def search(self, query_image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar images using ORB matching.
        
        Args:
            query_image_path: Path to query image
            top_k: Number of results to return
            
        Returns:
            List of matches with filename, score, and URL
        """
        logger.info(f"🔍 ORB searching for: {query_image_path}")
        
        # Compute features for query image
        query_features = self._compute_features(query_image_path)
        
        if query_features is None:
            logger.error("Could not compute features for query image")
            return []
        
        results = []
        
        for filename, cached_features in self.features_cache.items():
            try:
                # Match features
                matches = self.matcher.match(query_features, cached_features)
                
                # Sort by distance
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Score = number of good matches (distance < threshold)
                good_matches = [m for m in matches if m.distance < 50]
                score = len(good_matches)
                
                if score > 0:
                    # Get public URL from Supabase
                    url = self.supabase.storage.from_(self.bucket_name).get_public_url(filename)
                    
                    # Extract document name from filename
                    # e.g., "Production Auchan_page1_img3.png" -> "Production Auchan"
                    import re
                    doc_match = re.search(r'^(.+?)_page\d+_img\d+', filename)
                    doc_name = doc_match.group(1) if doc_match else filename
                    
                    # Try to find source document URL in 'source-documents' bucket
                    source_url = None
                    source_filename = None
                    try:
                        files_bucket = self.supabase.storage.from_('source-documents')
                        # List files in bucket to check which extension exists
                        bucket_files = files_bucket.list()
                        bucket_filenames = [f['name'] for f in bucket_files] if bucket_files else []
                        
                        for ext in ['.pptx', '.pdf', '.docx']:
                            potential_name = f"{doc_name}{ext}"
                            if potential_name in bucket_filenames:
                                source_url = files_bucket.get_public_url(potential_name)
                                source_filename = potential_name
                                logger.info(f"Found source document: {potential_name}")
                                break
                    except Exception as e:
                        logger.warning(f"Error finding source document: {e}")
                    
                    results.append({
                        'filename': filename,
                        'doc_name': doc_name,
                        'source_filename': source_filename,  # Full filename with extension
                        'score': score,
                        'image_url': url,
                        'source_url': source_url
                    })
                    
            except Exception as e:
                logger.warning(f"Error matching {filename}: {e}")
                continue
        
        # Sort by score (higher = better match)
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        if results:
            logger.info(f"✅ Found {len(results)} matches. Best: {results[0]['filename']} (score: {results[0]['score']})")
        else:
            logger.info("No matches found")
        
        return results[:top_k]


# Factory function
_orb_instance = None

def get_orb_image_search() -> Optional[ORBImageSearch]:
    """Get or create ORB image search instance."""
    global _orb_instance
    
    if _orb_instance is None:
        from dotenv import load_dotenv
        load_dotenv()
        
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        bucket = os.environ.get("SUPABASE_BUCKET_NAME", "alpagino")
        
        if url and key:
            _orb_instance = ORBImageSearch(url, key, bucket)
        else:
            logger.warning("Supabase credentials not found")
    
    return _orb_instance
