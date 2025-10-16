#!/usr/bin/env python3

import os
import glob
from collections import defaultdict
from dotenv import load_dotenv


def load_models_mapping(models_file):
    """Load model mapping from models_used.txt"""
    models = {}
    try:
        with open(models_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '. ' in line:
                    idx, path = line.split('. ', 1)
                    model_name = os.path.basename(path).replace('.safetensors', '')
                    models[int(idx)] = model_name
    except FileNotFoundError:
        print(f"Error: {models_file} not found")
    return models


def analyze_images(output_dir):
    """Analyze images in all subdirectories"""
    model_counts = defaultdict(lambda: defaultdict(int))
    
    # Find all subdirectories (timestamp folders)
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        
        # Load models mapping for this run
        models_file = os.path.join(subdir_path, "models_used.txt")
        if not os.path.exists(models_file):
            continue
            
        models = load_models_mapping(models_file)
        
        # Count images by model
        for img_file in glob.glob(os.path.join(subdir_path, "*.jpg")):
            filename = os.path.basename(img_file)
            
            # Extract model index from filename pattern: *_m{model_idx}_*
            if '_m' in filename:
                try:
                    model_part = filename.split('_m')[1].split('_')[0]
                    model_idx = int(model_part)
                    
                    if model_idx in models:
                        model_name = models[model_idx]
                        model_counts[model_name][subdir] += 1
                except (ValueError, IndexError):
                    continue
    
    return model_counts


def print_top_models(model_counts, top_n=10):
    """Print top N models by total image count"""
    # Calculate totals and sort
    model_totals = []
    for model, folder_counts in model_counts.items():
        total = sum(folder_counts.values())
        model_totals.append((model, total, folder_counts))
    
    model_totals.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top {top_n} models by image count:")
    print("-" * 50)
    
    for i, (model, total, folder_counts) in enumerate(model_totals[:top_n], 1):
        folder_details = ", ".join([f"{folder}:{count}" for folder, count in sorted(folder_counts.items())])
        print(f"{model}: ({total}) {folder_details}")


def main():
    load_dotenv()
    
    output_path = os.getenv("OUTPUT_DIRECTORY")
    if not output_path:
        print("Error: OUTPUT_DIRECTORY not set in .env")
        return
    
    if not os.path.exists(output_path):
        print(f"Error: Output directory does not exist: {output_path}")
        return
    
    model_counts = analyze_images(output_path)
    
    if not model_counts:
        print("No image data found")
        return
    
    print_top_models(model_counts)


if __name__ == "__main__":
    main()