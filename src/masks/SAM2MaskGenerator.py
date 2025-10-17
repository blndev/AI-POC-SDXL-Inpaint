import os
import logging
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import requests
from pathlib import Path

from ..utils.singleton import singleton


@singleton
class SAM2MaskGenerator:
    """Singleton class for SAM2-based mask generation with caching and multiple mask types."""
    
    SUPPORTED_MASK_TYPES = [
        "automatic",
        "point_prompt", 
        "text_prompt",
        "everything"
    ]
    
    MODEL_CONFIGS = {
        "vit_h": {
            "config": "sam2_hiera_l.yaml",
            "checkpoint": "sam2_hiera_large.pt",
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._predictor = None
        self._auto_mask_generator = None
        self._model_cache = {}
        self._models_dir = None
        
    def download_models(self, models_dir: str) -> None:
        """Download all required SAM2 models to specified directory."""
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, config in self.MODEL_CONFIGS.items():
            checkpoint_path = self._models_dir / config["checkpoint"]
            
            if not checkpoint_path.exists():
                self.logger.info(f"Downloading {model_name} model...")
                try:
                    response = requests.get(config["url"], stream=True)
                    response.raise_for_status()
                    
                    with open(checkpoint_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    self.logger.info(f"Downloaded {model_name} model successfully")
                except Exception as e:
                    self.logger.error(f"Failed to download {model_name}: {e}")
                    raise
            else:
                self.logger.info(f"{model_name} model already exists")
    
    def _load_model(self, model_name: str = "vit_h") -> Tuple[SAM2ImagePredictor, SAM2AutomaticMaskGenerator]:
        """Load SAM2 model with caching."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        
        if not self._models_dir:
            raise ValueError("Models directory not set. Call download_models() first.")
        
        config = self.MODEL_CONFIGS[model_name]
        checkpoint_path = self._models_dir / config["checkpoint"]
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        try:
            self.logger.info(f"Loading SAM2 {model_name} model...")
            
            # Build SAM2 model
            sam2_model = build_sam2(config["config"], str(checkpoint_path))
            
            # Create predictor and auto mask generator
            predictor = SAM2ImagePredictor(sam2_model)
            auto_mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
            
            self._model_cache[model_name] = (predictor, auto_mask_generator)
            self.logger.info(f"SAM2 {model_name} model loaded successfully")
            
            return predictor, auto_mask_generator
            
        except Exception as e:
            self.logger.error(f"Failed to load SAM2 model: {e}")
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
            mask_types = {"automatic": {}}
        
        try:
            predictor, auto_mask_generator = self._load_model()
            
            # Convert PIL to numpy
            image_np = np.array(image)
            predictor.set_image(image_np)
            
            results = {}
            
            for mask_name, params in mask_types.items():
                if mask_name == "automatic":
                    mask = self._generate_automatic_mask(image_np, auto_mask_generator)
                elif mask_name == "everything":
                    mask = self._generate_everything_mask(image_np, auto_mask_generator)
                else:
                    self.logger.warning(f"Unsupported mask type: {mask_name}")
                    continue
                
                results[mask_name] = mask
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to generate masks: {e}")
            raise
    
    def generate_mask_from_points(self, image: Image.Image, points: List[Tuple[int, int]], labels: Optional[List[int]] = None) -> Image.Image:
        """Generate mask from point coordinates.
        
        Args:
            image: Input PIL image
            points: List of (x, y) coordinates
            labels: List of point labels (1=foreground, 0=background)
            
        Returns:
            PIL mask image
        """
        try:
            predictor, _ = self._load_model()
            
            image_np = np.array(image)
            predictor.set_image(image_np)
            
            if labels is None:
                labels = [1] * len(points)
            
            input_points = np.array(points)
            input_labels = np.array(labels)
            
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )
            
            # Convert to PIL image
            mask = masks[0].astype(np.uint8) * 255
            return Image.fromarray(mask, mode='L')
            
        except Exception as e:
            self.logger.error(f"Failed to generate mask from points: {e}")
            raise
    
    def generate_mask_from_prompt(self, image: Image.Image, prompt: str) -> Image.Image:
        """Generate mask from text prompt.
        
        Note: SAM2 doesn't natively support text prompts. This is a placeholder
        for future integration with text-to-mask models like CLIP-based segmentation.
        
        Args:
            image: Input PIL image
            prompt: Text description of object to mask
            
        Returns:
            PIL mask image
        """
        self.logger.warning("Text prompt masking not yet implemented for SAM2")
        # For now, return automatic mask as fallback
        try:
            _, auto_mask_generator = self._load_model()
            image_np = np.array(image)
            return self._generate_automatic_mask(image_np, auto_mask_generator)
        except Exception as e:
            self.logger.error(f"Failed to generate mask from prompt: {e}")
            raise
    
    def _generate_automatic_mask(self, image_np: np.ndarray, auto_mask_generator: SAM2AutomaticMaskGenerator) -> Image.Image:
        """Generate automatic segmentation mask."""
        masks = auto_mask_generator.generate(image_np)
        
        if not masks:
            # Return empty mask if no objects detected
            return Image.new('L', (image_np.shape[1], image_np.shape[0]), 0)
        
        # Combine all masks into single mask
        combined_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
        for mask_data in masks:
            combined_mask = np.logical_or(combined_mask, mask_data['segmentation'])
        
        return Image.fromarray((combined_mask * 255).astype(np.uint8), mode='L')
    
    def _generate_everything_mask(self, image_np: np.ndarray, auto_mask_generator: SAM2AutomaticMaskGenerator) -> Image.Image:
        """Generate mask covering everything in the image."""
        # Return full white mask
        return Image.new('L', (image_np.shape[1], image_np.shape[0]), 255)