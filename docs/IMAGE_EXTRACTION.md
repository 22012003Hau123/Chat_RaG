# Image Extraction from Multiple Document Types

## Overview

The `image_processor.py` module now supports extracting images from:
- **PDF** files (.pdf)
- **Word** documents (.docx)
- **PowerPoint** presentations (.pptx)

## Installation

Install required dependencies:

```bash
pip install python-docx>=1.2.0
pip install python-pptx>=0.6.21
pip install PyMuPDF>=1.23.0
pip install Pillow>=10.0.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Auto-detect File Type

```python
from src.image_processor import extract_images

# Automatically detects file type and extracts images
images = extract_images("document.pdf")  # or .docx, .pptx

for image, metadata in images:
    print(f"Extracted image: {metadata}")
```

### Using Specific Extractors

```python
from src.image_processor import (
    extract_images_from_pdf,
    extract_images_from_docx,
    extract_images_from_pptx
)

# PDF
pdf_images = extract_images_from_pdf("report.pdf")

# Word
docx_images = extract_images_from_docx("document.docx")

# PowerPoint
pptx_images = extract_images_from_pptx("presentation.pptx")
```

### Generate Captions

```python
from src.image_processor import (
    extract_images, 
    generate_caption_gpt4v,
    generate_caption_fallback
)

images = extract_images("document.pdf")

for img, metadata in images:
    # Using GPT-4 Vision (requires OPENAI_API_KEY)
    caption = generate_caption_gpt4v(img, detail="low")
    
    # Or use fallback (no API required)
    caption = generate_caption_fallback(img, metadata)
    
    print(f"Caption: {caption}")
```

## Example Script

Run the included example script:

```bash
python example_extract_images.py path/to/document.pdf
python example_extract_images.py path/to/document.docx
python example_extract_images.py path/to/presentation.pptx
```

## Metadata Format

Extracted images return a tuple of `(PIL.Image, metadata)`:

**PDF metadata:**
```python
{
    'source': 'path/to/file.pdf',
    'page': 1,
    'image_index': 0,
    'width': 1920,
    'height': 1080,
    'format': 'png'
}
```

**Word (.docx) metadata:**
```python
{
    'source': 'path/to/file.docx',
    'image_index': 0,
    'width': 1920,
    'height': 1080,
    'format': 'png'
}
```

**PowerPoint (.pptx) metadata:**
```python
{
    'source': 'path/to/file.pptx',
    'slide': 5,
    'image_index': 0,
    'width': 1920,
    'height': 1080,
    'format': 'png'
}
```

## Features

- ✅ **Auto-detection**: Use `extract_images()` to automatically detect file type
- ✅ **Size filtering**: Automatically skips images smaller than 100x100 pixels (logos/icons)
- ✅ **Consistent metadata**: All extractors return the same data structure
- ✅ **Error handling**: Gracefully handles corrupt images and extraction errors
- ✅ **Caption generation**: Support for both GPT-4V and fallback captions

## Notes

- Images smaller than 100x100 pixels are automatically filtered out (assumed to be logos/icons)
- For PowerPoint, only images embedded in shapes are extracted
- Caption generation with GPT-4V requires `OPENAI_API_KEY` environment variable
