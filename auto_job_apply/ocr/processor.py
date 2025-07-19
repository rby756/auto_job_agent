# auto_job_apply/ocr/processor.py (updated)
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
import cv2

from auto_job_apply.config import get_config
from .utils import is_pdf, is_image, convert_pdf_to_images, extract_text_from_image, process_resume

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Processes resume files (PDFs and images) to extract text using OCR."""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        lang: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the OCR processor with configuration.
        
        Args:
            output_dir: Directory to save processed files
            lang: Language for OCR (default: from config)
            config_path: Path to custom config file
        """
        # Load configuration
        self.config = get_config()
        if config_path:
            from ..config import ConfigLoader
            self.config = ConfigLoader(config_path).load()
            
        self.lang = lang or self.config["OCR_CONFIG"]["tesseract"]["lang"]
        self.output_dir = Path(output_dir or self.config["FILE_PATHS"]["resumes"]["processed"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set tesseract command if specified in config
        tesseract_cmd = self.config["OCR_CONFIG"]["tesseract"].get("cmd")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        # Create temp directory
        self.temp_dir = Path(self.config["OCR_CONFIG"]["temp"]["dir"])
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_temp_file(self, prefix: str = None, suffix: str = None) -> Path:
        """Generate a temporary file path."""
        import tempfile
        prefix = prefix or self.config["OCR_CONFIG"]["temp"]["prefix"]
        suffix = suffix or self.config["OCR_CONFIG"]["temp"]["suffix"]
        return Path(tempfile.mktemp(prefix=prefix, suffix=suffix, dir=str(self.temp_dir)))
        
    def process_file(
        self, 
        file_path: Union[str, Path],
        save_text: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        show_text: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single resume file and extract text.
        
        Args:
            file_path: Path to the resume file (PDF or image)
            save_text: Whether to save the extracted text to a file
            output_file: Path to save the extracted text (if save_text is True)
            show_text: Whether to print the extracted text to console
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Process the file using the process_resume function from utils
            text = process_resume(
                file_path=file_path,
                output_dir=self.temp_dir if self.config["OCR_CONFIG"]["temp"]["keep_temp_files"] else None,
                lang=self.lang,
                config=self.config["OCR_CONFIG"]
            )
            
            # Save text if requested
            if save_text:
                if not output_file:
                    output_file = self.output_dir / f"{file_path.stem}.txt"
                else:
                    output_file = Path(output_file)
                    
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                logger.info(f"Saved extracted text to: {output_file}")
            
            # Count pages (for PDFs) or set to 1 for images
            num_pages = 1
            if str(file_path).lower().endswith('.pdf'):
                try:
                    with fitz.open(file_path) as doc:
                        num_pages = len(doc)
                except Exception as e:
                    logger.warning(f"Could not determine number of pages: {e}")
            
            # Show text if requested
            if show_text and text:
                print("\n" + "=" * 80)
                print(f"EXTRACTED TEXT FROM: {file_path.name}")
                print("=" * 80)
                print(text[:1000])  # Show first 1000 characters
                if len(text) > 1000:
                    print("... [truncated]")
                print("=" * 80 + "\n")
            
            return {
                'text': text,
                'file_path': str(file_path),
                'file_type': 'pdf' if str(file_path).lower().endswith('.pdf') else 'image',
                'pages': num_pages,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            return {
                'text': '',
                'file_path': str(file_path),
                'file_type': 'pdf' if str(file_path).lower().endswith('.pdf') else 'image',
                'pages': 0,
                'success': False,
                'error': str(e)
            }
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        recursive: bool = False,
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process all supported files in a directory.
        
        Args:
            input_dir: Directory containing files to process
            output_dir: Directory to save processed files
            recursive: Whether to process subdirectories
            file_extensions: List of file extensions to process (default: ['.pdf', '.png', '.jpg', '.jpeg'])
            
        Returns:
            Dictionary mapping file paths to processing results
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
            
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        file_extensions = file_extensions or ['.pdf', '.png', '.jpg', '.jpeg']
        results = {}
        
        # Get all matching files
        files = []
        if recursive:
            for ext in file_extensions:
                files.extend(input_dir.rglob(f"*{ext}"))
        else:
            for ext in file_extensions:
                files.extend(input_dir.glob(f"*{ext}"))
                
        # Process each file
        for file_path in files:
            if file_path.is_file():
                rel_path = file_path.relative_to(input_dir)
                output_file = self.output_dir / rel_path.with_suffix('.txt')
                results[str(file_path)] = self.process_file(
                    file_path=file_path,
                    save_text=True,
                    output_file=output_file
                )
                
        return results