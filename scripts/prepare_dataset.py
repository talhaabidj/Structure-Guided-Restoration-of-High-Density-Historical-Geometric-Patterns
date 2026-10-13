"""
Dataset Preparation Script for Structure-Guided Image Inpainting
Mughal Heritage Tile Restoration Project

This script prepares the training dataset by:
1. Loading images from HuggingFace Islamic Art dataset
2. Adding custom Mughal architecture images
3. Generating synthetic damage masks
4. Extracting Canny edge maps
5. Creating train/val/test splits

Authors: Team Labaik
- Talha Abid (L1F23BSAI0058)
- Abdullah Zafar (L1F23BSAI0054)
- Abdul Rehman (L1F23BSAI0050)
- Waqar Hussain (L1F23BSAI0078)
"""

import os
import cv2
import numpy as np
from PIL import Image
import random
from pathlib import Path
from typing import Tuple, List
import argparse


def create_directory_structure(base_path: str) -> dict:
    """Create the directory structure for the dataset."""
    dirs = {
        'train_images': os.path.join(base_path, 'train', 'images'),
        'train_masks': os.path.join(base_path, 'train', 'masks'),
        'train_edges': os.path.join(base_path, 'train', 'edges'),
        'val_images': os.path.join(base_path, 'val', 'images'),
        'val_masks': os.path.join(base_path, 'val', 'masks'),
        'val_edges': os.path.join(base_path, 'val', 'edges'),
        'test_images': os.path.join(base_path, 'test', 'images'),
        'test_masks': os.path.join(base_path, 'test', 'masks'),
        'test_edges': os.path.join(base_path, 'test', 'edges'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return dirs


def generate_irregular_mask(height: int, width: int, 
                           min_strokes: int = 1, 
                           max_strokes: int = 4) -> np.ndarray:
    """
    Generate irregular free-form masks to simulate damage/deterioration.
    
    Args:
        height: Image height
        width: Image width
        min_strokes: Minimum number of brush strokes
        max_strokes: Maximum number of brush strokes
    
    Returns:
        Binary mask (255 = masked region, 0 = valid region)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    num_strokes = random.randint(min_strokes, max_strokes)
    
    for _ in range(num_strokes):
        # Random starting point
        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)
        
        # Random brush size
        brush_size = random.randint(10, 40)
        
        # Number of vertices in the stroke
        num_vertices = random.randint(4, 12)
        
        # Generate random walk for the stroke
        points = [(start_x, start_y)]
        for _ in range(num_vertices):
            angle = random.uniform(0, 2 * np.pi)
            distance = random.randint(20, 60)
            new_x = int(points[-1][0] + distance * np.cos(angle))
            new_y = int(points[-1][1] + distance * np.sin(angle))
            new_x = max(0, min(width - 1, new_x))
            new_y = max(0, min(height - 1, new_y))
            points.append((new_x, new_y))
        
        # Draw the stroke
        points = np.array(points, dtype=np.int32)
        cv2.polylines(mask, [points], isClosed=False, color=255, thickness=brush_size)
        
        # Add some circular regions for variety
        if random.random() > 0.5:
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)
            radius = random.randint(15, 50)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    return mask


def generate_rectangular_mask(height: int, width: int,
                              num_rectangles: int = None) -> np.ndarray:
    """
    Generate rectangular masks to simulate missing tile sections.
    
    Args:
        height: Image height
        width: Image width
        num_rectangles: Number of rectangular holes
    
    Returns:
        Binary mask (255 = masked region, 0 = valid region)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if num_rectangles is None:
        num_rectangles = random.randint(1, 3)
    
    for _ in range(num_rectangles):
        # Random rectangle dimensions (10-30% of image)
        rect_w = random.randint(int(0.1 * width), int(0.3 * width))
        rect_h = random.randint(int(0.1 * height), int(0.3 * height))
        
        # Random position
        x = random.randint(0, width - rect_w)
        y = random.randint(0, height - rect_h)
        
        cv2.rectangle(mask, (x, y), (x + rect_w, y + rect_h), 255, -1)
    
    return mask


def generate_crack_mask(height: int, width: int) -> np.ndarray:
    """
    Generate crack-like masks to simulate natural deterioration.
    
    Args:
        height: Image height
        width: Image width
    
    Returns:
        Binary mask (255 = masked region, 0 = valid region)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    num_cracks = random.randint(2, 5)
    
    for _ in range(num_cracks):
        # Start from edge
        if random.random() > 0.5:
            start_x = random.choice([0, width - 1])
            start_y = random.randint(0, height - 1)
        else:
            start_x = random.randint(0, width - 1)
            start_y = random.choice([0, height - 1])
        
        points = [(start_x, start_y)]
        num_segments = random.randint(5, 15)
        
        for _ in range(num_segments):
            angle = random.uniform(-np.pi/4, np.pi/4)  # Mostly forward
            distance = random.randint(10, 40)
            new_x = int(points[-1][0] + distance * np.cos(angle + np.pi/2))
            new_y = int(points[-1][1] + distance * np.sin(angle + np.pi/2))
            new_x = max(0, min(width - 1, new_x))
            new_y = max(0, min(height - 1, new_y))
            points.append((new_x, new_y))
        
        points = np.array(points, dtype=np.int32)
        thickness = random.randint(3, 12)
        cv2.polylines(mask, [points], isClosed=False, color=255, thickness=thickness)
    
    return mask


def extract_canny_edges(image: np.ndarray, 
                        low_threshold: int = 50, 
                        high_threshold: int = 150) -> np.ndarray:
    """
    Extract edge map using Canny edge detector.
    Critical for preserving geometric patterns in Islamic art.
    
    Args:
        image: Input RGB image
        low_threshold: Canny low threshold
        high_threshold: Canny high threshold
    
    Returns:
        Edge map (binary image)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Extract edges
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges


def process_image(image_path: str, 
                  output_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single image: resize, generate mask, extract edges.
    
    Args:
        image_path: Path to input image
        output_size: Target size (width, height)
    
    Returns:
        Tuple of (processed_image, mask, edge_map)
    """
    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    image = image.resize(output_size, Image.Resampling.LANCZOS)
    image_np = np.array(image)
    
    # Generate random mask type
    mask_type = random.choice(['irregular', 'rectangular', 'crack'])
    if mask_type == 'irregular':
        mask = generate_irregular_mask(output_size[1], output_size[0])
    elif mask_type == 'rectangular':
        mask = generate_rectangular_mask(output_size[1], output_size[0])
    else:
        mask = generate_crack_mask(output_size[1], output_size[0])
    
    # Extract edges from original (clean) image
    edges = extract_canny_edges(image_np)
    
    return image_np, mask, edges


def prepare_dataset(input_dir: str, 
                    output_dir: str,
                    image_size: int = 256,
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1) -> dict:
    """
    Prepare the complete dataset for training.
    
    Args:
        input_dir: Directory containing source images
        output_dir: Directory to save processed dataset
        image_size: Target image size
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
    
    Returns:
        Dictionary with dataset statistics
    """
    print(f"\n{'='*60}")
    print("MUGHAL HERITAGE TILE DATASET PREPARATION")
    print(f"{'='*60}\n")
    
    # Create directory structure
    dirs = create_directory_structure(output_dir)
    
    # Collect all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found! Please check the input directory.")
        return {'total': 0, 'train': 0, 'val': 0, 'test': 0}
    
    # Shuffle and split
    random.shuffle(image_files)
    
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(train_files)} images")
    print(f"  Validation: {len(val_files)} images")
    print(f"  Testing:    {len(test_files)} images")
    
    # Process each split
    splits = [
        ('train', train_files, dirs['train_images'], dirs['train_masks'], dirs['train_edges']),
        ('val', val_files, dirs['val_images'], dirs['val_masks'], dirs['val_edges']),
        ('test', test_files, dirs['test_images'], dirs['test_masks'], dirs['test_edges']),
    ]
    
    for split_name, files, img_dir, mask_dir, edge_dir in splits:
        print(f"\nProcessing {split_name} set...")
        for i, img_path in enumerate(files):
            try:
                image, mask, edges = process_image(img_path, (image_size, image_size))
                
                # Save processed files
                filename = f"{split_name}_{i:05d}.png"
                
                cv2.imwrite(os.path.join(img_dir, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(mask_dir, filename), mask)
                cv2.imwrite(os.path.join(edge_dir, filename), edges)
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(files)} images")
                    
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
    
    stats = {
        'total': n_total,
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files),
        'image_size': image_size
    }
    
    print(f"\n{'='*60}")
    print("DATASET PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {stats['total']}")
    print(f"Output directory: {output_dir}")
    
    return stats


def download_huggingface_dataset(output_dir: str) -> str:
    """
    Download Islamic Art dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save downloaded images
    
    Returns:
        Path to downloaded images
    """
    try:
        from datasets import load_dataset
        
        print("Downloading Islamic Art dataset from HuggingFace...")
        dataset = load_dataset("adhamelarabawy/islamic_art", split="train")
        
        img_dir = os.path.join(output_dir, "huggingface_islamic_art")
        os.makedirs(img_dir, exist_ok=True)
        
        for i, item in enumerate(dataset):
            image = item['image']
            image.save(os.path.join(img_dir, f"islamic_art_{i:05d}.png"))
            
            if (i + 1) % 100 == 0:
                print(f"  Downloaded {i + 1}/{len(dataset)} images")
        
        print(f"Downloaded {len(dataset)} images to {img_dir}")
        return img_dir
        
    except ImportError:
        print("Please install datasets library: pip install datasets")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Mughal Heritage Tile Dataset")
    parser.add_argument("--input", type=str, required=True, help="Input directory with images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--size", type=int, default=256, help="Image size (default: 256)")
    parser.add_argument("--download-hf", action="store_true", help="Download HuggingFace dataset")
    parser.add_argument("--test", action="store_true", help="Run in test mode with sample data")
    
    args = parser.parse_args()
    
    if args.download_hf:
        download_huggingface_dataset(args.input)
    
    if args.test:
        print("Running in test mode...")
        # Create sample test data
        test_input = os.path.join(args.input, "test_samples")
        os.makedirs(test_input, exist_ok=True)
        
        # Generate sample images for testing
        for i in range(10):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(test_input, f"sample_{i}.png"), img)
        
        stats = prepare_dataset(test_input, args.output, args.size)
    else:
        stats = prepare_dataset(args.input, args.output, args.size)
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
