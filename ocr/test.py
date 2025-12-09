# quick_test.py - Simple version
import os
import json
import glob

# Add current directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import enhanced_ktp_processing

def quick_test():
    # Folder with your KTP images
    image_folder = "test_images"  # Change this to your folder name
    
    # Find all images
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']:
        images.extend(glob.glob(os.path.join(image_folder, ext)))
    
    print(f"Found {len(images)} images")
    
    results = []
    
    for img_path in images:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        try:
            raw_text, extracted_data, output_json = enhanced_ktp_processing(img_path)
            
            result = {
                "filename": os.path.basename(img_path),
                "raw_text": raw_text,
                "extracted_data": extracted_data
            }
            results.append(result)
            
            print(f"  Success! Extracted {len(extracted_data)} fields")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "filename": os.path.basename(img_path),
                "error": str(e)
            })
    
    # Save to JSON
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to test_results.json")

if __name__ == "__main__":
    quick_test()