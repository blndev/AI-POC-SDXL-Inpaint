#!/usr/bin/env python3
import os
import time
import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
from datetime import datetime
from dotenv import load_dotenv
import gc
from PIL import Image
import configparser
import glob


def prepare_gpu():
    """Configure GPU memory settings based on .env configuration"""
    if os.getenv('GPU_ALLOW_XFORMERS', '0') == '1':
        os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    if os.getenv('GPU_ALLOW_MEMORY_OFFLOAD', '0') == '1':
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_filters():
    """Load models filter if existing"""
    ignore_filters = []
    include_filters = []
    
    if os.getenv("MODEL_FILTER", None):
        filter_file = os.path.join(os.path.dirname(__file__), os.getenv("MODEL_FILTER", None))
        try:
            config = configparser.ConfigParser()
            config.read(filter_file)
            
            if 'ignore' in config:
                ignore_filters = [item.strip() for item in config['ignore'] if config['ignore'][item].strip()]
            
            if 'include' in config:
                include_filters = [item.strip() for item in config['include'] if config['include'][item].strip()]
                
        except FileNotFoundError:
            print(f"Warning: {filter_file} not found")
    
    return ignore_filters, include_filters

def find_safetensor_models(models_path, cache_path):
    ignore_filters, include_filters = load_filters()
    safetensors_files = []
    
    for root, dirs, files in os.walk(models_path):
        if cache_path and os.path.abspath(cache_path) in os.path.abspath(root):
            continue
            
        for file in files:
            if file.endswith('.safetensors'):
                model_path = os.path.join(root, file)
                model_name_lower = model_path.lower()
                
                # Check ignore filters
                should_ignore = False
                for ignore_filter in ignore_filters:
                    if ignore_filter.startswith('!'):
                        # Reverse filter - ignore if NOT contains
                        if ignore_filter[1:].lower() not in model_name_lower:
                            should_ignore = True
                            break
                    else:
                        # Normal filter - ignore if contains
                        if ignore_filter.lower() in model_name_lower:
                            should_ignore = True
                            break
                
                if should_ignore:
                    continue
                
                # Check include filters
                if include_filters:
                    should_include = False
                    for include_filter in include_filters:
                        if include_filter.lower() in model_name_lower:
                            should_include = True
                            break
                    if not should_include:
                        continue
                
                if model_path not in safetensors_files:
                    safetensors_files.append(model_path)
    
    return sorted(safetensors_files)

def prepare_image(image, mask, max_size):
    """Resize image to fit max_size while maintaining aspect ratio and ensure divisible by 8"""
    print(f"Input image size: {image.width}x{image.height}")
    
    # Resize to max_size while maintaining aspect ratio
    image.thumbnail((max_size, max_size))
    print(f"Resized to max size: {image.width}x{image.height}")
    
    # Ensure dimensions are divisible by 8
    new_width = image.width - (image.width % 8)
    new_height = image.height - (image.height % 8)
    print(f"Cropping to divisible by 8: {new_width}x{new_height}")
    
    image = image.crop((0, 0, new_width, new_height))
    
    # Process mask if provided
    if mask:
        mask = mask.convert('L')
        mask.thumbnail((max_size, max_size))
        mask = mask.crop((0, 0, new_width, new_height))
    else:
        # Create default mask (white = inpaint everything)
        mask = Image.new("L", image.size, 255)
    
    print(f"Final image size: {image.width}x{image.height}")
    print(f"Final mask size: {mask.width}x{mask.height}")
    
    return image, mask

def generate_images(pipe, count, image, mask, prompt, output_dir, model_idx, prompt_idx, image_name, inference_steps, strength):
    """Generate images using the pipeline"""
    try:
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=inference_steps,
            strength=strength,
            num_images_per_prompt=count
        )
        
        # Save generated images
        for i, img in enumerate(result.images):
            filename = f"{image_name}_m{model_idx}_p{prompt_idx}_i{i+1}.jpg"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath, "JPEG", quality=95)
            print(f"Saved: {filepath}")
            
    except Exception as e:
        print(f"Error generating images: {e}")

def load_prompts(prompts_file):
    """Load prompts from file"""
    prompts = []
    try:
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip() and not line.strip().startswith(('#', ';'))]
    except FileNotFoundError:
        print(f"Warning: {prompts_file} not found, using default prompt")
        prompts = ["high quality, detailed"]
    return prompts

def load_input_images(input_dir):
    """Load input images and their corresponding masks"""
    images = []
    
    # Find all image files (excluding masks)
    image_patterns = ['*.jpg', '*.jpeg', '*.png']
    for pattern in image_patterns:
        for img_path in glob.glob(os.path.join(input_dir, pattern)):
            if '_mask' not in os.path.basename(img_path):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Look for corresponding mask
                mask_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_mask = os.path.join(input_dir, f"{base_name}_mask{ext}")
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break
                
                images.append((img_path, mask_path, base_name))
    
    return images

def run_inpainting():
    """Run inpainting process"""
    models_path = os.getenv("MODEL_DIRECTORY")
    cache_path = os.getenv("CACHE_DIR")
    output_path = os.getenv("OUTPUT_DIRECTORY")
    input_path = os.getenv("INPUT_DIRECTORY", "./sources")
    prompts_file = os.getenv("PROMPTS", "prompts.txt")
    image_count = int(os.getenv("IMAGES", "2"))
    max_size = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
    inference_steps = int(os.getenv("INFERENCE_STEPS", "20"))
    strength = float(os.getenv("STRENGTH", "0.8"))
    
    # Validate required paths
    if not models_path:
        print("Error: MODEL_DIRECTORY not set")
        return
    if not output_path:
        print("Error: OUTPUT_DIRECTORY not set")
        return
    
    # Check paths exist
    if not os.path.exists(models_path):
        print(f"Error: Models path does not exist: {models_path}")
        return
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    output_dir = os.path.join(output_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create cache directory if specified
    if cache_path:
        os.makedirs(cache_path, exist_ok=True)
    
    # Find models and save list
    safetensors_files = find_safetensor_models(models_path, cache_path)
    print(f"Found {len(safetensors_files)} models")
    
    # Save model list to output directory
    with open(os.path.join(output_dir, "models_used.txt"), 'w') as f:
        for i, model in enumerate(safetensors_files, 1):
            f.write(f"{i}. {model}\n")
    
    # Save prompts list to output directory
    with open(os.path.join(output_dir, "prompts_used.txt"), 'w') as f:
        for i, prompt in enumerate(prompts, 1):
            f.write(f"{i}. {prompt}\n")
    
    # Load prompts and input images
    prompts = load_prompts(os.path.join(os.path.dirname(__file__), prompts_file))
    input_images = load_input_images(input_path)
    
    print(f"Found {len(input_images)} input images")
    print(f"Using {len(prompts)} prompts")
    
    # Prepare all images once
    prepared_images = []
    for img_path, mask_path, base_name in input_images:
        print(f"Preparing image: {base_name}")
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') if mask_path else None
        prepared_image, prepared_mask = prepare_image(image, mask, max_size)
        prepared_images.append((prepared_image, prepared_mask, base_name))
    
    prepare_gpu()
    
    # Process each model
    for model_idx, model_path in enumerate(safetensors_files, 1):
        print(f"\nProcessing model {model_idx}/{len(safetensors_files)}: {os.path.basename(model_path)}")
        
        pipe = None
        try:
            # Determine pipeline type based on model name
            if "sdxl" in model_path.lower():
                pipe = StableDiffusionXLInpaintPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    cache_dir=cache_path,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                pipe = StableDiffusionInpaintPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    cache_dir=cache_path,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            pipe.to("cuda")
            
            if torch.cuda.is_available():
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass  # Ignore if xformers not available
            
            print(f"Model loaded successfully")
            
            # Process each prepared image
            for image, mask, base_name in prepared_images:
                print(f"Processing image: {base_name}")
                
                # Generate with each prompt
                for prompt_idx, prompt in enumerate(prompts, 1):
                    print(f"  Generating with prompt {prompt_idx}: {prompt[:50]}...")
                    generate_images(pipe, image_count, image, mask, prompt, 
                                  output_dir, model_idx, prompt_idx, base_name, inference_steps, strength)
            
        except Exception as e:
            print(f"Error with model {model_path}: {e}")
        finally:
            if pipe:
                del pipe
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
    
    print(f"\nInpainting complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    try:
        load_dotenv(override=True)
        run_inpainting()
    except KeyboardInterrupt:
        print("Shutdown")
        exit()
