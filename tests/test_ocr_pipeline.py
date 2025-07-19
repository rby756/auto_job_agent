# notebooks/test_ocr_pipeline.py (updated)
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from auto_job_apply.ocr.processor import OCRProcessor
from auto_job_apply.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("RESUME OCR TESTING TOOL")
    print("=" * 80)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process resume files with OCR")
    parser.add_argument("--resume", type=str, help="Path to a single resume file")
    parser.add_argument("--resume-dir", type=str, help="Directory containing resume files")
    parser.add_argument("--output-dir", type=str, help="Output directory for processed files")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path to config file")
    parser.add_argument("--show-text", action="store_true", 
                       help="Show extracted text in console")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        config_path = None

    # Initialize OCR processor
    ocr = OCRProcessor(
        output_dir=args.output_dir,
        config_path=str(config_path) if config_path else None
    )

    # Process files
    if args.resume:
        logger.info(f"Processing resume: {args.resume}")
        result = ocr.process_file(
            args.resume,
            save_text=True,
            show_text=args.show_text
        )
        
        if result['success']:
            logger.info(f"Successfully processed {args.resume} ({result['pages']} pages)")
        else:
            logger.error(f"Failed to process {args.resume}: {result.get('error', 'Unknown error')}")
            
    elif args.resume_dir:
        print(f"\nProcessing all supported files in: {args.resume_dir}")
        results = ocr.process_directory(
            args.resume_dir,
            recursive=True
        )
        
        # Print summary
        success_count = sum(1 for r in results.values() if r['success'])
        print("\n" + "=" * 80)
        print(f"PROCESSING COMPLETE: {success_count}/{len(results)} files processed successfully")
        print("=" * 80)
        
    else:
        print("No input specified. Use --resume or --resume-dir")
        parser.print_help()

if __name__ == "__main__":
    main()