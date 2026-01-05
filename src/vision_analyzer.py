"""
Vision Analyzer - Image Analysis using GPT-4o-mini

This module handles image analysis for the RAG chat system:
- Analyzes technical drawings, layouts, planograms
- Extracts text, measurements, and spatial information
- Integrates with existing OpenAI infrastructure

Usage:
    from src.vision_analyzer import VisionAnalyzer
    analyzer = VisionAnalyzer()
    description = analyzer.analyze_image(image_path, user_question)
"""

import os
import base64
from typing import Optional
from openai import OpenAI


class VisionAnalyzer:
    """
    Image analysis using GPT-4o-mini Vision API.
    
    Optimized for technical images:
    - Engineering drawings
    - Floor plans and layouts
    - Product specifications
    - Planograms and shelf layouts
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize Vision Analyzer with OpenAI client.
        
        Args:
            model: Vision model to use (default: gpt-4o-mini for cost efficiency)
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        print(f"✓ Vision Analyzer initialized with {self.model}")
    
    def analyze_image(
        self, 
        image_path: str, 
        user_question: Optional[str] = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Analyze image and extract relevant information.
        
        Args:
            image_path: Path to image file
            user_question: Optional user question for context
            max_tokens: Maximum response length
            
        Returns:
            Detailed description of image content
        """
        print(f"\n📸 Analyzing image: {image_path}")
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine image type for file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(file_ext, 'image/jpeg')
        
        # Create optimized prompt for technical images
        system_prompt = self._create_technical_prompt(user_question)
        
        # Call GPT-4o-mini Vision API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_image}",
                                    "detail": "high"  # High detail mode for technical images
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.1  # Low temperature for factual accuracy
            )
            
            description = response.choices[0].message.content
            print(f"✓ Image analysis complete ({len(description)} chars)")
            
            return description
            
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _create_technical_prompt(self, user_question: Optional[str] = None) -> str:
        """
        Create optimized prompt for technical image analysis.
        
        Args:
            user_question: User's specific question about the image
            
        Returns:
            Formatted prompt string
        """
        base_prompt = """Analyze this image in detail and extract ALL information for document search.

**CRITICAL: Extract for Search**
- Product names, codes, identifiers (exact spelling)
- File names, document references
- Technical terms, specifications
- Key phrases and terminology

**Layout & Structure:**
- Overall layout and spatial organization
- Dimensions and measurements (exact values if visible)
- Zones, sections, or areas

**Text & Labels:**
- All visible text, labels, and annotations  
- Product codes, part numbers, or identifiers
- Headings and section titles
- **FILE NAMES** (très important!)

**Technical Details:**
- Specifications and parameters
- Tables and data (extract all values)
- Symbols and technical notation

**Visual Elements:**
- Products, components, or objects
- Colors, materials, or finishes
- Relationships between elements

**IMPORTANT FORMAT:**
Start with: "KEYWORDS FOR SEARCH: [list key terms separated by commas]"
Then provide full detailed analysis.

Be precise with numbers and technical terminology."""

        if user_question:
            return f"""{base_prompt}

**User's Question:**
{user_question}

**IMPORTANT:** 
If user asks to "find" or "search" for something in the image, LIST ALL SEARCHABLE TERMS prominently so the RAG system can find relevant documents."""
        
        return base_prompt


def save_uploaded_image(upload_file, upload_dir: str = "static/temp_images") -> str:
    """
    Save uploaded image to temporary directory.
    
    Args:
        upload_file: FastAPI UploadFile object
        upload_dir: Directory to save images
        
    Returns:
        Path to saved image file
    """
    import uuid
    from pathlib import Path
    
    # Create upload directory if not exists
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_ext = os.path.splitext(upload_file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        content = upload_file.file.read()
        f.write(content)
    
    print(f"✓ Image saved: {file_path}")
    return file_path


def cleanup_temp_image(file_path: str) -> None:
    """
    Delete temporary image file.
    
    Args:
        file_path: Path to image file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Cleaned up temp image: {file_path}")
    except Exception as e:
        print(f"⚠️  Could not delete temp image: {e}")


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vision_analyzer.py <image_path> [question]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyzer = VisionAnalyzer()
    result = analyzer.analyze_image(image_path, question)
    
    print("\n" + "="*60)
    print("VISION ANALYSIS RESULT")
    print("="*60)
    print(result)
    print("="*60)
