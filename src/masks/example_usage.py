#!/usr/bin/env python3
"""
Example usage of SAM2MaskGenerator and MediaPipeMaskGenerator for mask generation.
"""

import logging
from PIL import Image
from SAM2MaskGenerator import SAM2MaskGenerator
from MediaPipeMaskGenerator import MediaPipeMaskGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Demonstrate SAM2MaskGenerator usage."""
    
    # Initialize the singleton mask generator
    mask_gen = SAM2MaskGenerator()
    
    # Download models (only needed once)
    print("Downloading SAM2 models...")
    mask_gen.download_models("./models")
    
    # Load test image
    try:
        image = Image.open("../cli/sources/image1.png")
        print(f"Loaded image: {image.size}")
    except FileNotFoundError:
        print("Test image not found. Please provide a valid image path.")
        return
    
    # Get supported mask types
    supported_types = mask_gen.get_supported_mask_types()
    print(f"Supported mask types: {supported_types}")
    
    # Generate automatic masks
    print("Generating automatic masks...")
    masks = mask_gen.generate_masks_by_type(image, {"automatic": {}})
    
    for mask_name, mask_image in masks.items():
        output_path = f"output_{mask_name}_mask.png"
        mask_image.save(output_path)
        print(f"Saved {mask_name} mask to {output_path}")
    
    # Generate mask from points
    print("Generating mask from points...")
    points = [(image.width // 2, image.height // 2)]  # Center point
    point_mask = mask_gen.generate_mask_from_points(image, points)
    point_mask.save("output_point_mask.png")
    print("Saved point-based mask to output_point_mask.png")
    
    # Generate mask from text prompt (placeholder)
    print("Generating mask from text prompt...")
    text_mask = mask_gen.generate_mask_from_prompt(image, "person")
    text_mask.save("output_text_mask.png")
    print("Saved text-based mask to output_text_mask.png")
    
    # MediaPipe mask generation
    print("\n--- MediaPipe Mask Generation ---")
    mp_mask_gen = MediaPipeMaskGenerator()
    mp_mask_gen.download_models("./models")
    
    # Generate human segmentation masks
    print("Generating human segmentation masks...")
    mp_masks = mp_mask_gen.generate_masks_by_type(image, {
        "hair": {}, "clothes": {}, "body_skin": {}
    })
    
    for mask_name, mask_image in mp_masks.items():
        output_path = f"output_mp_{mask_name}_mask.png"
        mask_image.save(output_path)
        print(f"Saved MediaPipe {mask_name} mask to {output_path}")
    
    # Specialized masks for beauty applications
    hair_mask = mp_mask_gen.generate_hair_recolor_mask(image)
    hair_mask.save("output_hair_recolor_mask.png")
    print("Saved hair recolor mask")
    
    tattoo_mask = mp_mask_gen.generate_tattoo_removal_mask(image)
    tattoo_mask.save("output_tattoo_removal_mask.png")
    print("Saved tattoo removal mask")

if __name__ == "__main__":
    main()