"""
Image processing functionality for the Comprehensive Survey Analyzer
"""

import os
from typing import Dict, Optional, Tuple
from PIL import Image
import requests
from config import MAX_IMAGE_DIMENSION, SUPPORTED_IMAGE_FORMATS

class ImageProcessor:
    """Handles image processing and analysis"""
    
    def __init__(self, output_directory: str):
        """Initialize the image processor"""
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
    
    def download_image(self, url: str, filename: str) -> Optional[str]:
        """Download an image from a URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filepath = os.path.join(self.output_directory, filename)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return filepath
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def process_image(self, image_path: str) -> Dict:
        """Process an image and return its metadata"""
        try:
            with Image.open(image_path) as img:
                # Get basic image info
                info = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "file_size": os.path.getsize(image_path)
                }
                
                # Resize if too large
                if max(img.size) > MAX_IMAGE_DIMENSION:
                    ratio = MAX_IMAGE_DIMENSION / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img.save(image_path)
                    info["resized"] = True
                    info["new_size"] = new_size
                
                return info
        except Exception as e:
            print(f"Error processing image: {e}")
            return {}
    
    def validate_image(self, filepath: str) -> Tuple[bool, str]:
        """Validate an image file"""
        try:
            # Check file extension
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in SUPPORTED_IMAGE_FORMATS:
                return False, f"Unsupported file format: {ext}"
            
            # Try to open the image
            with Image.open(filepath) as img:
                # Check image dimensions
                if max(img.size) > MAX_IMAGE_DIMENSION * 2:
                    return False, f"Image too large: {img.size}"
                
                # Check file size (10MB limit)
                if os.path.getsize(filepath) > 10 * 1024 * 1024:
                    return False, "File too large (max 10MB)"
                
                return True, "Valid image"
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def get_image_metadata(self, image_path: str) -> Dict:
        """Get metadata for an image"""
        try:
            with Image.open(image_path) as img:
                return {
                    "filename": os.path.basename(image_path),
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "file_size": os.path.getsize(image_path),
                    "dimensions": f"{img.size[0]}x{img.size[1]}",
                    "aspect_ratio": round(img.size[0] / img.size[1], 2)
                }
        except Exception as e:
            return {
                "filename": os.path.basename(image_path),
                "error": str(e)
            }
    
    def cleanup_images(self) -> None:
        """Clean up all downloaded images"""
        for filename in os.listdir(self.output_directory):
            if any(filename.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
                try:
                    os.remove(os.path.join(self.output_directory, filename))
                except:
                    pass 