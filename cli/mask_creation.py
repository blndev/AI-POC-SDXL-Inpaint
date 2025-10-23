#!/usr/bin/env python3
"""
Mask creation tool using SAM2 and MediaPipe mask generators.
"""

import os
import sys
import logging
import glob
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from masks.MediaPipeMaskGenerator import MediaPipeMaskGenerator

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def load_input_images(input_dir):
    """Load input images from directory."""
    images = []
    image_patterns = ['*.jpg','*.JPG', '*.jpeg', '*.png']

    for pattern in image_patterns:
        for img_path in glob.glob(os.path.join(input_dir, pattern)):
            if '_mask' not in os.path.basename(img_path):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                images.append((img_path, base_name))

    return images


def create_masks():
    """Create masks using both SAM2 and MediaPipe generators."""

    # Load config from .env
    input_path = os.getenv("INPUT_DIRECTORY", "./sources")
    output_path = os.getenv("OUTPUT_DIRECTORY")
    models_path = os.getenv("MODEL_DIRECTORY", "./models")

    if not output_path:
        print("Error: OUTPUT_DIRECTORY not set")
        return

    # Create masks output directory
    masks_output_dir = output_path + "_masks"
    os.makedirs(masks_output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return

    # Load input images
    input_images = load_input_images(input_path)
    print(f"Found {len(input_images)} input images")

    if not input_images:
        print("No input images found")
        return

    print("Initializing MediaPipe mask generator...")
    mp_gen = MediaPipeMaskGenerator()
    mp_gen.download_models(models_path)

    # Process each image
    for img_path, base_name in input_images:
        print(f"\nProcessing: {base_name}")

        try:
            image = Image.open(img_path).convert('RGB')
            print(f"Loaded image: {image.size}")

            # MediaPipe masks
            print("Generating MediaPipe masks...")
            mp_masks = {}
            mp_masks["background"] = mp_gen.generate_masks_for_test(image, background=True)
            mp_masks["body"] = mp_gen.generate_masks_for_test(image, body=True)
            mp_masks["face"] = mp_gen.generate_masks_for_test(image, face=True)
            mp_masks["clothes"] = mp_gen.generate_masks_for_test(image, clothes=True)
            mp_masks["accessoirs"] = mp_gen.generate_masks_for_test(image, accessories=True)
            mp_masks["hair"] = mp_gen.generate_masks_for_test(image, hair=True)
            mp_masks["body_cloth"] = mp_gen.generate_masks_for_test(image, body=True, clothes=True)
            mp_masks["cloth_accessoirs"] = mp_gen.generate_masks_for_test(image, clothes=True, accessories=True)

            print("MediaPipe masks generated...")

            for name, mask_image in mp_masks.items():
                output_path = os.path.join(masks_output_dir, f"{base_name}_mp_{name}_mask.png")
                mask_image.save(output_path)
                print(f"Saved: {output_path}")

            # # Specialized masks
            # hair_mask = mp_gen.generate_hair_recolor_mask(image)
            # hair_mask.save(os.path.join(masks_output_dir, f"{base_name}_hair_recolor_mask.png"))

            # tattoo_mask = mp_gen.generate_tattoo_removal_mask(image)
            # tattoo_mask.save(os.path.join(masks_output_dir, f"{base_name}_tattoo_removal_mask.png"))

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    print(f"\nMask creation complete! Results saved to: {masks_output_dir}")


if __name__ == "__main__":
    try:
        load_dotenv(override=True)
        create_masks()
    except KeyboardInterrupt:
        print("Shutdown")
        sys.exit()
