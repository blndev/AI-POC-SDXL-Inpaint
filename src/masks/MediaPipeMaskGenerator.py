import os
import logging
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
from pathlib import Path

from ..utils.singleton import singleton


@singleton
class MediaPipeMaskGenerator:
    """Singleton class for MediaPipe-based human segmentation mask generation."""
    
    SUPPORTED_MASK_TYPES = [
        "hair",
        "body_skin", 
        "face_skin",
        "clothes",
        "body",
        "background"
    ]
    
    # MediaPipe selfie segmentation model
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"
    MODEL_FILE = "selfie_segmenter.tflite"
    
    # Segmentation categories mapping
    CATEGORY_MAPPING = {
        "background": 0,
        "hair": 1,
        "body_skin": 2,
        "face_skin": 3,
        "clothes": 4,
        "body": [2, 3]  # Combined body and face skin
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._segmenter = None
        self._models_dir = None
        
    def download_models(self, models_dir: str) -> None:
        """Download MediaPipe segmentation model."""
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = self._models_dir / self.MODEL_FILE
        
        if not model_path.exists():
            self.logger.info("Downloading MediaPipe selfie segmentation model...")
            try:
                response = requests.get(self.MODEL_URL, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.logger.info("Downloaded MediaPipe model successfully")
            except Exception as e:
                self.logger.error(f"Failed to download MediaPipe model: {e}")
                raise
        else:
            self.logger.info("MediaPipe model already exists")
    
    def _load_segmenter(self) -> vision.ImageSegmenter:
        """Load MediaPipe image segmenter with caching."""
        if self._segmenter is not None:
            return self._segmenter
        
        if not self._models_dir:
            raise ValueError("Models directory not set. Call download_models() first.")
        
        model_path = self._models_dir / self.MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.logger.info("Loading MediaPipe segmenter...")
            
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_category_mask=True
            )
            
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
            self.logger.info("MediaPipe segmenter loaded successfully")
            
            return self._segmenter
            
        except Exception as e:
            self.logger.error(f"Failed to load MediaPipe segmenter: {e}")
            raise
    
    def get_supported_mask_types(self) -> List[str]:
        """Return list of all supported masking types."""
        return self.SUPPORTED_MASK_TYPES.copy()
    
    def generate_masks_by_type(self, image: Image.Image, mask_types: Optional[Dict[str, dict]] = None) -> Dict[str, Image.Image]:
        """Generate masks by specified types.
        
        Args:
            image: Input PIL image
            mask_types: Dict of mask type names and their parameters
            
        Returns:
            Dict mapping mask names to PIL mask images
        """
        if mask_types is None:
            mask_types = {"hair": {}, "clothes": {}}
        
        try:
            segmenter = self._load_segmenter()
            
            # Convert PIL to MediaPipe format
            image_np = np.array(image)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            
            # Perform segmentation
            segmentation_result = segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask.numpy_view()
            
            results = {}
            
            for mask_name in mask_types.keys():
                if mask_name not in self.SUPPORTED_MASK_TYPES:
                    self.logger.warning(f"Unsupported mask type: {mask_name}")
                    continue
                
                mask = self._create_mask_for_category(category_mask, mask_name)
                results[mask_name] = mask
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to generate masks: {e}")
            raise
    
    def generate_mask_for_category(self, image: Image.Image, category: str) -> Image.Image:
        """Generate mask for specific category.
        
        Args:
            image: Input PIL image
            category: Category name (hair, clothes, body_skin, etc.)
            
        Returns:
            PIL mask image
        """
        if category not in self.SUPPORTED_MASK_TYPES:
            raise ValueError(f"Unsupported category: {category}")
        
        try:
            segmenter = self._load_segmenter()
            
            image_np = np.array(image)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            
            segmentation_result = segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask.numpy_view()
            
            return self._create_mask_for_category(category_mask, category)
            
        except Exception as e:
            self.logger.error(f"Failed to generate mask for {category}: {e}")
            raise
    
    def _create_mask_for_category(self, category_mask: np.ndarray, category: str) -> Image.Image:
        """Create binary mask for specific category."""
        mask = np.zeros_like(category_mask, dtype=np.uint8)
        
        category_ids = self.CATEGORY_MAPPING.get(category)
        
        if isinstance(category_ids, list):
            # Multiple categories (e.g., body = body_skin + face_skin)
            for cat_id in category_ids:
                mask[category_mask == cat_id] = 255
        else:
            # Single category
            mask[category_mask == category_ids] = 255
        
        return Image.fromarray(mask, mode='L')
    
    def generate_tattoo_removal_mask(self, image: Image.Image) -> Image.Image:
        """Generate mask for tattoo removal (body skin areas).
        
        Args:
            image: Input PIL image
            
        Returns:
            PIL mask image covering body skin areas
        """
        return self.generate_mask_for_category(image, "body_skin")
    
    def generate_hair_recolor_mask(self, image: Image.Image) -> Image.Image:
        """Generate mask for hair recoloring.
        
        Args:
            image: Input PIL image
            
        Returns:
            PIL mask image covering hair areas
        """
        return self.generate_mask_for_category(image, "hair")
    
    def generate_clothes_recolor_mask(self, image: Image.Image) -> Image.Image:
        """Generate mask for clothes recoloring.
        
        Args:
            image: Input PIL image
            
        Returns:
            PIL mask image covering clothing areas
        """
        return self.generate_mask_for_category(image, "clothes")