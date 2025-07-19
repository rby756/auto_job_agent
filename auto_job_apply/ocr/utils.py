"""
Utility functions for the OCR pipeline.
"""
import os
import re
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple
import cv2
import pytesseract
from PIL import Image

# Configure Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def is_pdf(file_path: Union[str, Path]) -> bool:
    """Check if the file is a PDF."""
    return str(file_path).lower().endswith('.pdf')

def is_image(file_path: Union[str, Path]) -> bool:
    """Check if the file is a supported image format."""
    img_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    return Path(file_path).suffix.lower() in img_extensions

def convert_pdf_to_images(
    pdf_path: Union[str, Path],
    dpi: int = 300,
    output_dir: Optional[Union[str, Path]] = None,
    output_prefix: str = 'page_'
) -> List[Image.Image]:
    """
    Convert a PDF file to a list of PIL Images.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for the output images
        output_dir: Directory to save images (if None, images won't be saved)
        output_prefix: Prefix for saved image files
        
    Returns:
        List of PIL Image objects
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    images = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        
        if output_dir is not None:
            img_path = output_dir / f"{output_prefix}{page_num + 1:03d}.png"
            img.save(img_path, 'PNG', dpi=(dpi, dpi))
    
    return images

def preprocess_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    dpi: int = 300,
    denoise: bool = True,
    threshold: bool = True,
    deskew: bool = True
) -> np.ndarray:
    """
    Preprocess an image for better OCR results.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        dpi: Target DPI for the image
        denoise: Whether to apply denoising
        threshold: Whether to apply thresholding
        deskew: Whether to deskew the image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif isinstance(image, Image.Image):
        img = np.array(image.convert('L'))
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image.copy()
    else:
        raise ValueError("Unsupported image format")
    
    # Resize based on DPI (assuming 72 DPI as standard)
    if dpi > 72:
        scale = dpi / 72
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    if denoise:
        img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Threshold
    if threshold:
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    # Deskew (simple implementation)
    if deskew:
        coords = np.column_stack(np.where(img > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), 
                           flags=cv2.INTER_CUBIC, 
                           borderMode=cv2.BORDER_REPLICATE)
    
    return img

def extract_text_from_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    lang: str = 'eng',
    config: str = '--psm 6 --oem 3',
    preprocess: bool = True
) -> str:
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        lang: Language for OCR (default: 'eng')
        config: Tesseract configuration
        preprocess: Whether to preprocess the image before OCR
        
    Returns:
        Extracted text
    """
    if preprocess:
        img = preprocess_image(image)
    else:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
    
    # Convert to RGB if needed (Tesseract expects RGB)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform OCR
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return text.strip()

def process_resume(
    file_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> str:
    """
    Process a resume file (PDF or image) and extract text.
    
    Args:
        file_path: Path to the resume file
        output_dir: Directory to save processed images (if any)
        **kwargs: Additional arguments for text extraction
        
    Returns:
        Extracted text from the resume
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if is_pdf(file_path):
        # Convert PDF to images
        images = convert_pdf_to_images(file_path, output_dir=output_dir)
        
        # Extract text from each page
        texts = []
        for i, img in enumerate(images):
            text = extract_text_from_image(img, **kwargs)
            if text.strip():
                texts.append(text)
        
        return "\n\n".join(texts)
    
    elif is_image(file_path):
        # Process single image
        return extract_text_from_image(file_path, **kwargs)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
def extract_text_from_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    lang: str = 'eng',
    config: str = '--psm 6 --oem 3',
    preprocess: bool = True
) -> str:
    """
    Extract text from an image using Tesseract OCR with enhanced preprocessing.
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Could not load image: {image}")
    elif isinstance(image, Image.Image):
        img = np.array(image)
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:  # RGB or RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    # Convert to grayscale for preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if preprocess:
        # Apply preprocessing
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # Apply dilation to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # Apply erosion to remove noise
        thresh = cv2.erode(thresh, kernel, iterations=1)
    else:
        thresh = gray
    
    # Use Tesseract with optimized configuration
    custom_config = (
        '--oem 3 '  # LSTM OCR Engine
        '--psm 6 '   # Assume a single uniform block of text
        '-c preserve_interword_spaces=1 '  # Preserve spaces
    )
    
    # Perform OCR with the custom configuration
    text = pytesseract.image_to_string(
        thresh, 
        lang=lang, 
        config=custom_config
    )
    
    # Clean up the extracted text
    lines = []
    for line in text.split('\n'):
        line = ' '.join(line.split())  # Normalize whitespace
        if line.strip():  # Only keep non-empty lines
            lines.append(line)
    
    return '\n'.join(lines)