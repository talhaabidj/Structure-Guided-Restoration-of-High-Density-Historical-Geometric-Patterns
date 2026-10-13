"""
Evaluation Metrics Script for Structure-Guided Image Inpainting

This script computes evaluation metrics for comparing inpainting methods:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE (Mean Absolute Error)
- FID (Fréchet Inception Distance)

Authors: Team Labaik
"""

import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
from typing import Dict, List, Tuple
import json


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    PSNR = 10 * log10(MAX^2 / MSE)
    
    Higher PSNR indicates better quality (typically 20-40 dB for good reconstruction).
    
    Args:
        img1: Ground truth image
        img2: Reconstructed image
    
    Returns:
        PSNR value in dB
    """
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index between two images.
    
    SSIM measures the perceived quality based on luminance, contrast, and structure.
    Range: [-1, 1], where 1 indicates perfect similarity.
    
    Args:
        img1: Ground truth image
        img2: Reconstructed image
    
    Returns:
        SSIM value
    """
    # Convert to grayscale if color
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    return ssim(img1_gray, img2_gray, data_range=255)


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error between two images.
    
    MAE = (1/n) * Σ|x - y|
    
    Lower MAE indicates better reconstruction.
    
    Args:
        img1: Ground truth image
        img2: Reconstructed image
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))


def calculate_l1_loss(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate L1 Loss (normalized).
    
    Args:
        img1: Ground truth image
        img2: Reconstructed image
    
    Returns:
        L1 loss value (normalized to [0, 1])
    """
    return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))) / 255.0


def calculate_l2_loss(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate L2 Loss (MSE).
    
    Args:
        img1: Ground truth image
        img2: Reconstructed image
    
    Returns:
        L2 loss value
    """
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def evaluate_single_image(gt_path: str, pred_path: str) -> Dict[str, float]:
    """
    Evaluate a single image pair.
    
    Args:
        gt_path: Path to ground truth image
        pred_path: Path to predicted/reconstructed image
    
    Returns:
        Dictionary with all metrics
    """
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)
    
    if gt is None:
        raise ValueError(f"Could not load ground truth: {gt_path}")
    if pred is None:
        raise ValueError(f"Could not load prediction: {pred_path}")
    
    # Ensure same size
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    return {
        'psnr': calculate_psnr(gt, pred),
        'ssim': calculate_ssim(gt, pred),
        'mae': calculate_mae(gt, pred),
        'l1': calculate_l1_loss(gt, pred),
        'l2': calculate_l2_loss(gt, pred)
    }


def evaluate_dataset(gt_dir: str, pred_dir: str) -> Dict[str, float]:
    """
    Evaluate entire dataset and compute average metrics.
    
    Args:
        gt_dir: Directory with ground truth images
        pred_dir: Directory with predicted images
    
    Returns:
        Dictionary with average metrics
    """
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    all_metrics = {
        'psnr': [],
        'ssim': [],
        'mae': [],
        'l1': [],
        'l2': []
    }
    
    print(f"Evaluating {len(gt_files)} images...")
    
    for i, filename in enumerate(gt_files):
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        
        if not os.path.exists(pred_path):
            print(f"Warning: Missing prediction for {filename}")
            continue
        
        try:
            metrics = evaluate_single_image(gt_path, pred_path)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(gt_files)} images")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Compute averages
    avg_metrics = {}
    for key in all_metrics:
        if len(all_metrics[key]) > 0:
            avg_metrics[f'{key}_mean'] = np.mean(all_metrics[key])
            avg_metrics[f'{key}_std'] = np.std(all_metrics[key])
    
    avg_metrics['num_images'] = len(all_metrics['psnr'])
    
    return avg_metrics


def compare_methods(gt_dir: str, 
                    method_dirs: Dict[str, str]) -> Dict[str, Dict]:
    """
    Compare multiple inpainting methods.
    
    Args:
        gt_dir: Directory with ground truth images
        method_dirs: Dictionary mapping method names to prediction directories
    
    Returns:
        Dictionary with metrics for each method
    """
    results = {}
    
    for method_name, pred_dir in method_dirs.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*50}")
        
        if not os.path.exists(pred_dir):
            print(f"Warning: Directory not found: {pred_dir}")
            continue
        
        metrics = evaluate_dataset(gt_dir, pred_dir)
        results[method_name] = metrics
    
    return results


def print_comparison_table(results: Dict[str, Dict]) -> str:
    """
    Print a formatted comparison table.
    
    Args:
        results: Dictionary with metrics for each method
    
    Returns:
        Formatted table as string
    """
    header = "| Method | PSNR (dB) | SSIM | MAE | L1 |"
    separator = "|--------|-----------|------|-----|-----|"
    
    rows = [header, separator]
    
    for method, metrics in results.items():
        psnr_val = metrics.get('psnr_mean', 0)
        ssim_val = metrics.get('ssim_mean', 0)
        mae_val = metrics.get('mae_mean', 0)
        l1_val = metrics.get('l1_mean', 0)
        
        row = f"| {method} | {psnr_val:.2f} ± {metrics.get('psnr_std', 0):.2f} | {ssim_val:.4f} | {mae_val:.2f} | {l1_val:.4f} |"
        rows.append(row)
    
    table = "\n".join(rows)
    print("\n" + table + "\n")
    return table


def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Inpainting Results")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth directory")
    parser.add_argument("--pred", type=str, required=True, help="Predictions directory")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--compare", nargs='+', help="Additional methods to compare (format: name:path)")
    
    args = parser.parse_args()
    
    if args.compare:
        method_dirs = {'Proposed': args.pred}
        for method in args.compare:
            name, path = method.split(':')
            method_dirs[name] = path
        
        results = compare_methods(args.gt, method_dirs)
        print_comparison_table(results)
    else:
        results = evaluate_dataset(args.gt, args.pred)
        print("\nEvaluation Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    save_results(results, args.output)
