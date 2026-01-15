"""
CLIP Image Search

Uses CLIP embeddings for semantic image similarity search.
Deep learning-based image understanding for better matching.
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Lazy load heavy dependencies
_clip_model = None
_clip_processor = None


def _load_clip_model():
    """Lazy load CLIP model."""
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        model_name = "openai/clip-vit-base-patch32"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"🔄 Loading CLIP model: {model_name} on {device}")
        _clip_model = CLIPModel.from_pretrained(model_name).to(device)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        logger.info("✅ CLIP model loaded")
    
    return _clip_model, _clip_processor


class CLIPImageSearch:
    """CLIP-based image similarity search."""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None, bucket_name: str = "alpagino"):
        """
        Initialize CLIP search.
        
        Args:
            supabase_url: Supabase project URL (for getting image URLs)
            supabase_key: Supabase API key
            bucket_name: Storage bucket name
        """
        # Supabase connection (optional, for URL generation)
        self.supabase = None
        self.bucket_name = bucket_name
        
        if supabase_url and supabase_key:
            from supabase import create_client
            self.supabase = create_client(supabase_url, supabase_key)
        
        # Local cache paths
        self.cache_dir = Path("./image_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings_file = self.cache_dir / "clip_embeddings.pkl"
        
        # Embeddings cache: {filename: embedding_vector}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"🔍 CLIP Image Search initialized. Cached: {len(self.embeddings)} images")
    
    def _load_cache(self):
        """Load cached embeddings from disk."""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"📂 Loaded {len(self.embeddings)} CLIP embeddings from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.embeddings = {}
    
    def _save_cache(self):
        """Save embeddings to disk."""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info(f"💾 Saved {len(self.embeddings)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Get CLIP embedding for an image."""
        try:
            import torch
            from PIL import Image
            
            model, processor = _load_clip_model()
            device = next(model.parameters()).device
            
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
            
            # Normalize embedding
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error getting embedding for {image_path}: {e}")
            return None
    
    def sync_from_supabase(self) -> int:
        """
        Download images from Supabase and ensure they're in local cache.
        Returns number of images downloaded.
        """
        if not self.supabase:
            logger.warning("Supabase not configured, skipping sync")
            return 0
        
        import requests
        
        logger.info(f"📥 Syncing images from Supabase bucket: {self.bucket_name}")
        
        try:
            files = self.supabase.storage.from_(self.bucket_name).list(
                path='',
                options={'limit': 10000}
            )
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
            image_files = [f for f in files if f['name'].lower().endswith(image_extensions)]
            
            logger.info(f"Found {len(image_files)} images in bucket")
            
            downloaded = 0
            for file_info in image_files:
                filename = file_info['name']
                local_path = self.cache_dir / filename
                
                if not local_path.exists():
                    try:
                        url = self.supabase.storage.from_(self.bucket_name).get_public_url(filename)
                        response = requests.get(url, timeout=30)
                        
                        if response.status_code == 200:
                            with open(local_path, 'wb') as f:
                                f.write(response.content)
                            downloaded += 1
                    except Exception as e:
                        logger.warning(f"Failed to download {filename}: {e}")
            
            logger.info(f"✅ Downloaded {downloaded} new images")
            return downloaded
            
        except Exception as e:
            logger.error(f"Error syncing from Supabase: {e}")
            return 0
    
    def build_cache_from_local(self) -> int:
        """
        Build CLIP embeddings for all images in local cache.
        Returns number of new embeddings created.
        """
        logger.info(f"📂 Building CLIP embeddings from: {self.cache_dir}")
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        new_count = 0
        
        image_files = [f for f in self.cache_dir.iterdir() 
                       if f.suffix.lower() in image_extensions]
        
        total = len(image_files)
        for i, img_file in enumerate(image_files):
            filename = img_file.name
            
            # Skip if already cached
            if filename in self.embeddings:
                continue
            
            embedding = self._get_embedding(str(img_file))
            if embedding is not None:
                self.embeddings[filename] = embedding
                new_count += 1
                
                if new_count % 50 == 0:
                    logger.info(f"  Progress: {new_count} new embeddings...")
        
        if new_count > 0:
            self._save_cache()
        
        logger.info(f"✅ Built {new_count} new embeddings. Total: {len(self.embeddings)}")
        return new_count
    
    def search(self, query_image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar images using CLIP embeddings.
        
        Args:
            query_image_path: Path to query image
            top_k: Number of results to return
            
        Returns:
            List of matches with filename, score, and URL
        """
        logger.info(f"🔍 CLIP searching for: {query_image_path}")
        
        if len(self.embeddings) == 0:
            logger.warning("No embeddings in cache. Run build_cache_from_local() first.")
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query_image_path)
        if query_embedding is None:
            logger.error("Could not get embedding for query image")
            return []
        
        # Calculate similarities (cosine similarity)
        results = []
        for filename, embedding in self.embeddings.items():
            similarity = float(np.dot(query_embedding, embedding))
            results.append({
                'filename': filename,
                'score': similarity
            })
        
        # Sort by similarity (highest first)
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Add URLs and document info
        import re
        for r in results[:top_k]:
            filename = r['filename']
            
            # Get image URL from Supabase
            if self.supabase:
                r['image_url'] = self.supabase.storage.from_(self.bucket_name).get_public_url(filename)
            else:
                r['image_url'] = f"/image_cache/{filename}"
            
            # Extract document name
            doc_match = re.search(r'^(.+?)_page\d+_img\d+', filename)
            r['doc_name'] = doc_match.group(1) if doc_match else filename
            
            # Try to find source document URL
            r['source_url'] = None
            r['source_filename'] = None
            
            if self.supabase:
                try:
                    doc_bucket = self.supabase.storage.from_('source-documents')
                    bucket_files = doc_bucket.list()
                    bucket_filenames = [f['name'] for f in bucket_files] if bucket_files else []
                    
                    for ext in ['.pdf', '.docx', '.pptx']:
                        potential_name = f"{r['doc_name']}{ext}"
                        if potential_name in bucket_filenames:
                            r['source_url'] = doc_bucket.get_public_url(potential_name)
                            r['source_filename'] = potential_name
                            break
                except Exception as e:
                    logger.debug(f"Could not find source document: {e}")
        
        if results:
            logger.info(f"✅ Found {len(results)} matches. Best: {results[0]['filename']} (score: {results[0]['score']:.4f})")
        else:
            logger.info("No matches found")
        
        return results[:top_k]


# Factory function
_clip_instance = None


def get_clip_image_search() -> Optional[CLIPImageSearch]:
    """Get or create CLIP image search instance."""
    global _clip_instance
    
    if _clip_instance is None:
        from dotenv import load_dotenv
        load_dotenv()
        
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        bucket = os.environ.get("SUPABASE_BUCKET_NAME", "alpagino")
        
        _clip_instance = CLIPImageSearch(url, key, bucket)
    
    return _clip_instance
