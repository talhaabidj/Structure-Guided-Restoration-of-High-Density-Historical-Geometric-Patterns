import cv2
import numpy as np
import os
from glob import glob

def refine_image(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        return

    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    
    # 1. Sharpening (Unsharp Mask)
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Needs LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    clahe_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 3. Denoising (Non-local Means) - Good for removing artifacts
    denoised_image = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # 4. Combined (Denoise -> Sharpen -> CLAHE)
    combined = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    gaussian_c = cv2.GaussianBlur(combined, (0, 0), 2.0)
    combined = cv2.addWeighted(combined, 1.3, gaussian_c, -0.3, 0)
    # Optional mild CLAHE
    lab_c = cv2.cvtColor(combined, cv2.COLOR_BGR2LAB)
    l_c, a_c, b_c = cv2.split(lab_c)
    clahe_c = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl_c = clahe_c.apply(l_c)
    limg_c = cv2.merge((cl_c,a_c,b_c))
    combined = cv2.cvtColor(limg_c, cv2.COLOR_LAB2BGR)

    # Save results
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_sharpened.png"), unsharp_image)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_clahe.png"), clahe_image)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_denoised.png"), denoised_image)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_refined_combined.png"), combined)
    
    print(f"Processed {filename}")

def main():
    input_dir = "extracted_restored"
    output_dir = "refined_results"
    os.makedirs(output_dir, exist_ok=True)
    
    images = glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(images)} images to refine.")
    
    for img_path in images:
        refine_image(img_path, output_dir)

if __name__ == "__main__":
    main()
