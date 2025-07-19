import pytest
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

class TestOCRProcessor:
    @pytest.fixture
    def mock_ocr_processor(self):
        with patch('auto_job_apply.ocr.processor.OCRProcessor') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    @patch('auto_job_apply.ocr.utils.convert_pdf_to_images')
    @patch('auto_job_apply.ocr.utils.extract_text_from_image')
    @patch('auto_job_apply.ocr.utils.fitz.open')
    @patch('auto_job_apply.ocr.utils.is_pdf')
    def test_process_resume(self, mock_is_pdf, mock_fitz_open, mock_extract, mock_convert, tmp_path):
        """Test processing a resume file."""
        from auto_job_apply.ocr.utils import process_resume
        
        # Setup test file
        test_file = tmp_path / "test_resume.pdf"
        test_file.write_text("Mock PDF content")  # Write some content to the file
        
        # Setup mocks
        # Mock is_pdf to return True for our test file
        mock_is_pdf.return_value = True
        
        # Mock fitz.open to return a mock document
        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = mock_doc  # For context manager
        mock_fitz_open.return_value = mock_doc
        
        # Mock convert_pdf_to_images to return a mock image
        mock_image = MagicMock()
        mock_convert.return_value = [mock_image]  # Return a list with one mock image
        
        # Mock extract_text_from_image to return sample text
        mock_extract.return_value = "Extracted text"
        
        # Test
        result = process_resume(str(test_file))
        
        # Verify the result
        assert result == "Extracted text"
        
        # Verify is_pdf was called once with the correct argument (string or Path)
        assert mock_is_pdf.call_count == 1
        called_with = mock_is_pdf.call_args[0][0]
        assert str(called_with) == str(test_file)
        
        # Verify convert_pdf_to_images was called once with the correct arguments (accepting string or Path)
        assert mock_convert.call_count == 1
        convert_args = mock_convert.call_args[0]
        convert_kwargs = mock_convert.call_args[1]
        assert len(convert_args) == 1
        assert str(convert_args[0]) == str(test_file)
        assert convert_kwargs == {'output_dir': None}
        
        # Verify extract_text_from_image was called once with the mock image
        assert mock_extract.call_count == 1
        assert mock_extract.call_args[0][0] == mock_image
        # Don't check the exact kwargs since they have default values in the function signature
    
    @patch('auto_job_apply.ocr.utils.pytesseract')
    @patch('auto_job_apply.ocr.utils.cv2')
    def test_extract_text_from_image(self, mock_cv2, mock_pytesseract, tmp_path):
        """Test text extraction from an image."""
        from auto_job_apply.ocr.utils import extract_text_from_image
        
        # Create a test image file path
        test_image = tmp_path / "test.jpg"
        test_image.touch()  # Create empty file
        
        # Setup mocks
        mock_cv2.imread.return_value = np.array([[0, 0], [0, 0]], dtype=np.uint8)  # Mock image data
        mock_cv2.cvtColor.return_value = "grayscale_image"
        mock_pytesseract.image_to_string.return_value = "extracted text"
        
        # Test with the actual file path
        result = extract_text_from_image(str(test_image))
        
        # Verify
        assert result == "extracted text"
        mock_cv2.imread.assert_called_once_with(str(test_image))
        mock_cv2.cvtColor.assert_called_once()
        mock_pytesseract.image_to_string.assert_called_once()
