import cv2
import numpy as np
import os
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(gt_path, pred_path):
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)
    
    if gt is None or pred is None:
        return 0, 0
        
    # Ensure same size (just in case)
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        
    p = psnr(gt, pred)
    s = ssim(gt, pred, channel_axis=2)
    return p, s

def main():
    extracted_dir = "extracted_restored"
    refined_dir = "refined_results"
    
    # We have 5 samples (0 to 4)
    results = {
        "Baseline (LaMa)": {"psnr": [], "ssim": []},
        "Sharpened": {"psnr": [], "ssim": []},
        "CLAHE": {"psnr": [], "ssim": []},
        "Denoised": {"psnr": [], "ssim": []},
        "Combined": {"psnr": [], "ssim": []}
    }
    
    for i in range(5):
        gt_path = os.path.join(extracted_dir, f"original_{i}.png")
        
        # 1. Baseline
        base_path = os.path.join(extracted_dir, f"restored_{i}.png")
        p, s = calculate_metrics(gt_path, base_path)
        results["Baseline (LaMa)"]["psnr"].append(p)
        results["Baseline (LaMa)"]["ssim"].append(s)
        
        # 2. Sharpened
        sharp_path = os.path.join(refined_dir, f"restored_{i}_sharpened.png")
        p, s = calculate_metrics(gt_path, sharp_path)
        results["Sharpened"]["psnr"].append(p)
        results["Sharpened"]["ssim"].append(s)
        
        # 3. CLAHE
        clahe_path = os.path.join(refined_dir, f"restored_{i}_clahe.png")
        p, s = calculate_metrics(gt_path, clahe_path)
        results["CLAHE"]["psnr"].append(p)
        results["CLAHE"]["ssim"].append(s)
        
        # 4. Denoised
        denoise_path = os.path.join(refined_dir, f"restored_{i}_denoised.png")
        p, s = calculate_metrics(gt_path, denoise_path)
        results["Denoised"]["psnr"].append(p)
        results["Denoised"]["ssim"].append(s)
        
        # 5. Combined
        comb_path = os.path.join(refined_dir, f"restored_{i}_refined_combined.png")
        p, s = calculate_metrics(gt_path, comb_path)
        results["Combined"]["psnr"].append(p)
        results["Combined"]["ssim"].append(s)

    print(f"{'Method':<20} | {'PSNR (dB)':<10} | {'SSIM':<10}")
    print("-" * 46)
    
    for method, metrics in results.items():
        avg_psnr = np.mean(metrics["psnr"])
        avg_ssim = np.mean(metrics["ssim"])
        print(f"{method:<20} | {avg_psnr:<10.2f} | {avg_ssim:<10.4f}")

if __name__ == "__main__":
    main()
