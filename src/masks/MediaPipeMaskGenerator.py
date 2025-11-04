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
import cv2
from functools import reduce

# from ..utils.singleton import singleton

logger = logging.getLogger(__name__)

# @singleton


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
    # Infos: https://developers.google.com/mediapipe/solutions/vision/image_segmenter#multiclass-model
    MODEL_URL_BG_FG = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    MODEL_FILE = "selfie_segmenter.tflite"

    # Segmentation categories mapping
    CATEGORY_MAPPING = {
        "background": 0,
        "hair": 1,
        "body": 2,
        "face": 3,
        "clothes": 4,
        "accessories": 5
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
        self.logger.debug(f"checking '{model_path}'")

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
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
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

    def generate_mask(self,
                      image: Image.Image,
                      background: bool = False,
                      body: bool = False,
                      face: bool = False,
                      hair: bool = False,
                      clothes: bool = False,
                      accessories: bool = False
                      ) -> Image.Image:
        if not image:
            raise ValueError("image is required")
        if not (background or body or face or hair or clothes or accessories):
            return Image.new("L", image.size, 255)  # simple full mask

        segmenter = self._load_segmenter()
        image_np = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        segmented_masks = segmenter.segment(mp_image)
        # using segments defined by google
        # https://developers.google.com/mediapipe/solutions/vision/image_segmenter#multiclass-model
        # TODO: move to header
        # 0 - background
        # 1 - hair
        # 2 - body - skin
        # 3 - face - skin
        # 4 - clothes
        # 5 - others(accessories)

        masks = []
        if background:
            masks.append(segmented_masks.confidence_masks[0])
        if hair:
            masks.append(segmented_masks.confidence_masks[1])
        if body:
            masks.append(segmented_masks.confidence_masks[2])
        if face:
            masks.append(segmented_masks.confidence_masks[3])
        if clothes:
            masks.append(segmented_masks.confidence_masks[4])
        if accessories:
            masks.append(segmented_masks.confidence_masks[5])

        image_data = mp_image.numpy_view()
        image_shape = image_data.shape

        # convert the image shape from "rgb" to "rgba" aka add the alpha channel
        if image_shape[-1] == 3:
            image_shape = (image_shape[0], image_shape[1], 4)

        mask_background_array = np.zeros(image_shape, dtype=np.uint8)
        mask_background_array[:] = (0, 0, 0, 255)

        mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
        mask_foreground_array[:] = (255, 255, 255, 255)

        mask_arrays = []
        for i, mask in enumerate(masks):
            condition = np.stack((mask.numpy_view(),) * image_shape[-1], axis=-1) > 0.25
            mask_array = np.where(condition, mask_foreground_array, mask_background_array)
            mask_arrays.append(mask_array)

        # Merge our masks taking the maximum from each
        merged_mask_arrays = reduce(np.maximum, mask_arrays)

        # Create the image
        mask_image = Image.fromarray(merged_mask_arrays)

        return mask_image

    def add_blur_on_borders(self, pil_mask, expansion_pixels=10, blur_radius=15):
        """
        Erweitert und weichzeichnet die Inpainting-Maske.

        Args:
            pil_mask (PIL.Image): Die Maske, wie sie vom Gradio ImageEditor geliefert wird (Graustufe/Alpha).
            expansion_pixels (int): Die Anzahl der Pixel, um die die Maske erweitert werden soll (Dilatation).
            blur_radius (int): Der Radius für den Gaußschen Weichzeichner (muss ungerade sein).

        Returns:
            np.ndarray: Die endgültige, erweiterte und unscharfe Maske (Graustufen 0-255).
        """
        if pil_mask is None:
            return None

        # --- Schritt 1: Konvertierung von PIL zu OpenCV/NumPy ---
        # Sicherstellen, dass die Maske in einem 8-Bit-Graustufenformat vorliegt (0 oder 255)
        # Wenn gr.ImageEditor verwendet wird, kommt die Maske oft als Alphakanal (L oder LA).
        # Wir konvertieren sie zu Graustufe und dann zu NumPy Array.

        # Konvertiere in Graustufe ('L') und dann zu NumPy
        np_mask = np.array(pil_mask.convert('L'))
        # Sicherstellen, dass die Werte 0 und 255 sind
        _, binary_mask = cv2.threshold(np_mask, 1, 255, cv2.THRESH_BINARY)

        # --- Schritt 2: Maske erweitern (Dilatation) ---
        if expansion_pixels > 0:
            # Kernel-Größe für die Dilatation berechnen (z.B. 21x21 für 10 Pixel)
            kernel_size = 2 * expansion_pixels + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Dilatation durchführen, um die Maske zu vergrößern
            expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        else:
            expanded_mask = binary_mask

        # --- Schritt 3: Ränder weichzeichnen (Gaussian Blur) ---

        # Blur-Radius muss ungerade sein
        if blur_radius % 2 == 0:
            blur_radius += 1

        # Gaußschen Weichzeichner anwenden, um weiche Kanten zu erzeugen
        final_mask_np = cv2.GaussianBlur(expanded_mask, (blur_radius, blur_radius), 0)

        # OpenCV gibt die Maske als np.uint8 Array zurück.
        return Image.fromarray(final_mask_np, mode='L')
