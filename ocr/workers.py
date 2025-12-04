import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import sys
import logging
import traceback
import time
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - WORKER - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Worker starting... loading model...")

    try:
        from doctr.models import ocr_predictor
        from doctr.io import DocumentFile
        model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="sar_resnet31",
            pretrained=True,
            assume_straight_pages=False,
            preserve_aspect_ratio=True,
        )
        logger.info("✓ Model loaded")
    except Exception as e:
        logger.error("Startup failed", exc_info=True)
        print(json.dumps({"error": f"Startup failed: {e}", "raw_text": []}))
        sys.stdout.flush()
        sys.exit(1)

    logger.info("Worker ready. Waiting for image paths...")

    while True:
        line = sys.stdin.readline()

        # EOF or empty → avoid CPU spin
        if not line:
            time.sleep(0.01)
            continue

        line = line.strip()
        if not line:
            continue

        logger.info(f"Received: {line}")

        try:
            # Echo test
            if line == "test":
                print(json.dumps({"raw_text": ["test_success"], "error": None}))
                sys.stdout.flush()
                continue

            # Process image
            if os.path.exists(line):
                doc = DocumentFile.from_images(line)
                result = model(doc)

                # Group words by lines and preserve line structure
                structured_lines = []
                raw_words = []
                
                for page in result.pages:
                    for block in page.blocks:
                        for line_obj in block.lines:
                            # Get all words in this line
                            line_words = []
                            for word_obj in line_obj.words:
                                word_text = word_obj.value
                                line_words.append(word_text)
                                raw_words.append(word_text)
                            
                            # Join words in the line with spaces
                            if line_words:
                                line_text = " ".join(line_words)
                                structured_lines.append(line_text)
                
                # Calculate mean confidence if needed
                mean_confidence = 0.0
                confidences = []
                for page in result.pages:
                    for block in page.blocks:
                        for line_obj in block.lines:
                            for word_obj in line_obj.words:
                                if hasattr(word_obj, 'confidence'):
                                    confidences.append(word_obj.confidence)
                
                if confidences:
                    mean_confidence = sum(confidences) / len(confidences)

                output = {
                    "raw_text": structured_lines,  # Now returns lines instead of individual words
                    "mean_confidence": mean_confidence,
                    "error": None,
                    "line_count": len(structured_lines)
                }

                print(json.dumps(output))
                sys.stdout.flush()

            else:
                print(json.dumps({"error": f"File not found: {line}", "raw_text": []}))
                sys.stdout.flush()

        except Exception as e:
            logger.error("Processing error", exc_info=True)
            print(json.dumps({"error": f"Processing error: {e}", "raw_text": []}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()