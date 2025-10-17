# AI-POC-SDXL-Inpaint
A POC to test Mask generation and Inpainting using Segment Anything Model 2 (SAM2)

## Features

- 🎯 **SAM2 Mask Generation**: Advanced mask generation using Meta's Segment Anything Model 2
- 🖱️ **Point-based Masking**: Generate masks from user-defined coordinates
- 🤖 **Automatic Segmentation**: Intelligent object detection and masking
- 🔄 **Model Caching**: Efficient model loading with singleton pattern
- 📝 **Multiple Mask Types**: Support for various masking approaches

## Mask Generators

### SAM2 Mask Generator

The `SAM2MaskGenerator` class provides comprehensive mask generation capabilities:

**Supported Mask Types:**
- `automatic`: Automatic object segmentation
- `point_prompt`: Mask generation from coordinate points
- `text_prompt`: Text-based mask generation (placeholder)
- `everything`: Full image mask

### MediaPipe Mask Generator

The `MediaPipeMaskGenerator` class specializes in human segmentation for beauty and fashion applications:

**Supported Mask Types:**
- `hair`: Hair segmentation for recoloring
- `body_skin`: Body skin areas for tattoo removal
- `face_skin`: Facial skin segmentation
- `clothes`: Clothing segmentation for recoloring
- `body`: Combined body and face skin
- `background`: Background segmentation

### Usage Examples

#### SAM2 Mask Generator
```python
from src.masks.SAM2MaskGenerator import SAM2MaskGenerator
from PIL import Image

# Initialize the singleton mask generator
mask_gen = SAM2MaskGenerator()
mask_gen.download_models("./models")

image = Image.open("input.jpg")

# Generate automatic masks
masks = mask_gen.generate_masks_by_type(image, {"automatic": {}})

# Generate mask from points
points = [(100, 150), (200, 250)]
mask = mask_gen.generate_mask_from_points(image, points)
```

#### MediaPipe Mask Generator
```python
from src.masks.MediaPipeMaskGenerator import MediaPipeMaskGenerator
from PIL import Image

# Initialize the singleton mask generator
mp_mask_gen = MediaPipeMaskGenerator()
mp_mask_gen.download_models("./models")

image = Image.open("person.jpg")

# Generate hair mask for recoloring
hair_mask = mp_mask_gen.generate_hair_recolor_mask(image)

# Generate clothes mask for recoloring
clothes_mask = mp_mask_gen.generate_clothes_recolor_mask(image)

# Generate body skin mask for tattoo removal
skin_mask = mp_mask_gen.generate_tattoo_removal_mask(image)

# Generate multiple masks at once
masks = mp_mask_gen.generate_masks_by_type(image, {
    "hair": {}, "clothes": {}, "body_skin": {}
})
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-POC-SDXL-Inpaint
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download SAM2 models:
```python
from src.masks.SAM2MaskGenerator import SAM2MaskGenerator
mask_gen = SAM2MaskGenerator()
mask_gen.download_models("./models")
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- SAM2 model checkpoints (automatically downloaded)
- Sufficient GPU memory for model inference

## Model Information

- **Model**: SAM2 ViT-H (Hierarchical Vision Transformer)
- **Checkpoint**: `sam2_hiera_large.pt` (~2.4GB)
- **Config**: `sam2_hiera_l.yaml`
- **Source**: Meta's Segment Anything 2

## Error Handling

The mask generator includes comprehensive error handling:
- Model download failures
- Invalid input validation
- Memory management
- Detailed logging for debugging
