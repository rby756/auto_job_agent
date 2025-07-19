"""
OCR module for processing resume documents and extracting text.
"""
from .processor import OCRProcessor
from .utils import (
    is_pdf,
    is_image,
    convert_pdf_to_images,
    preprocess_image
)

__all__ = [
    'OCRProcessor',
    'is_pdf',
    'is_image',
    'convert_pdf_to_images',
    'preprocess_image'
]
