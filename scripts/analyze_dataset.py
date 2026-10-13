import os
import hashlib
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import concurrent.futures

DATASET_ROOT = "Dataset/IslamicPatterns"

def get_file_hash(filepath):
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def analyze_image(filepath):
    """Return stats for a single image."""
    try:
        file_hash = get_file_hash(filepath)
        size_bytes = os.path.getsize(filepath)
        with Image.open(filepath) as img:
            width, height = img.size
            mode = img.mode
        return {
            'path': filepath,
            'hash': file_hash,
            'size': size_bytes,
            'dims': (width, height),
            'mode': mode,
            'valid': True
        }
    except Exception as e:
        return {
            'path': filepath,
            'valid': False,
            'error': str(e)
        }

def analyze_dataset():
    print(f"Analyzing {DATASET_ROOT}...")
    
    all_files = []
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                all_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_files)} images.")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_image, f): f for f in all_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_files), unit="img"):
            results.append(future.result())
            
    # Process results
    valid_images = [r for r in results if r.get('valid')]
    corrupt_images = [r for r in results if not r.get('valid')]
    
    # Duplicate detection (Exact Hash)
    hash_map = defaultdict(list)
    for img in valid_images:
        hash_map[img['hash']].append(img)
        
    duplicates = {k: v for k, v in hash_map.items() if len(v) > 1}
    duplicate_count = sum(len(v) - 1 for v in duplicates.values())
    
    # Stats per folder
    folder_stats = defaultdict(lambda: {'count': 0, 'duplicates': 0})
    for img in valid_images:
        folder = os.path.dirname(img['path'])
        folder_name = os.path.basename(folder)
        folder_stats[folder_name]['count'] += 1
        
    # Count duplicates within folders vs across folders
    # (Simplified: just counting total duplicates found)
    
    print("\n" + "="*40)
    print("DATASET ANALYSIS REPORT")
    print("="*40)
    print(f"Total Files Scanned: {len(all_files)}")
    print(f"Valid Images: {len(valid_images)}")
    print(f"Corrupt/Invalid Images: {len(corrupt_images)}")
    print(f"Unique Content (by hash): {len(hash_map)}")
    print(f"Exact Duplicates to Remove: {duplicate_count}")
    print("-" * 40)
    print("Top Folders by Image Count:")
    sorted_folders = sorted(folder_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    for folder, stats in sorted_folders[:10]:
        print(f"  {folder}: {stats['count']} images")
    print("-" * 40)
    
    if duplicates:
        print(f"Found {len(duplicates)} sets of duplicates.")
        # Example duplicate
        first_dup_hash = next(iter(duplicates))
        print(f"Example duplicate set (Hash: {first_dup_hash[:8]}...):")
        for img in duplicates[first_dup_hash]:
            print(f"  - {img['path']} ({img['dims']}, {img['size']} bytes)")

if __name__ == "__main__":
    analyze_dataset()
