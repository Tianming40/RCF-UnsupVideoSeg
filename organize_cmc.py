#!/usr/bin/env python3
import os
import shutil
from pathlib import Path


CMC_BASE = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-10")
OUTPUT_BASE = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_medical_format")


DIR_STRUCTURE = [
    "JPEGImages/cmc_sequence",         
    "Annotations/cmc_sequence",         
    "Flows/cmc_sequence",                
    "ImageSets",
]

def create_directory_structure():
    for dir_path in DIR_STRUCTURE:
        (OUTPUT_BASE / dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Directory structure created: {OUTPUT_BASE}")

def copy_all_images_to_jpegs():
    """Copy all pre and post images to JPEGImages (Annotations remains empty)"""
    pre_dir = CMC_BASE / "pre"
    post_dir = CMC_BASE / "post"
    
    jpeg_dir = OUTPUT_BASE / "JPEGImages/cmc_sequence"
    
    # Copy pre images
    pre_files = list(pre_dir.glob("*.png"))
    for img in pre_files:
        shutil.copy2(img, jpeg_dir / img.name)
    
    # Copy post images
    post_files = list(post_dir.glob("*.png"))
    for img in post_files:
        shutil.copy2(img, jpeg_dir / img.name)
    
    total = len(pre_files) + len(post_files)
    print(f"✓ Copied {len(pre_files)} pre images to JPEGImages")
    print(f"✓ Copied {len(post_files)} post images to JPEGImages")
    print(f"✓ JPEGImages contains {total} images")
    print(f"✓ Annotations directory is empty (CMC has no segmentation labels)")

def create_imageset_files():
    """Create ImageSets split files"""
    imageset_dir = OUTPUT_BASE / "ImageSets"
    jpeg_dir = OUTPUT_BASE / "JPEGImages/cmc_sequence"
    
    # Get all filenames
    frame_names = [p.name for p in sorted(jpeg_dir.glob("*.png"))]
    
    # Format: path/ + space + filename1 space filename2 space ...
    content = "JPEGImages/cmc_sequence/ " + " ".join(frame_names)
    
    with open(imageset_dir / "all_frames.txt", "w") as f:
        f.write(content)
    
    total = len(frame_names)
    train_count = int(total * 0.8)
    
    with open(imageset_dir / "train.txt", "w") as f:
        f.write("JPEGImages/cmc_sequence/ " + " ".join(frame_names[:train_count]))
    
    with open(imageset_dir / "val.txt", "w") as f:
        f.write("JPEGImages/cmc_sequence/ " + " ".join(frame_names[train_count:]))
    
    with open(imageset_dir / "trainval.txt", "w") as f:
        f.write(content)
    
    # Create empty 4-fold cross-validation files
    for fold in range(1, 5):
        for split in ["train", "val"]:
            (imageset_dir / f"fold{fold}_{split}.txt").touch()
    
    print(f"✓ Created ImageSets files ({total} frames in total)")

def main():
    print("=" * 60)
    print("Convert CMC/grasp-10 to data_medical format")
    print("JPEGImages contains all pre+post images")
    print("Annotations remains empty")
    print("=" * 60)
    
    create_directory_structure()
    copy_all_images_to_jpegs()
    create_imageset_files()
    
    print(f"\n✅ Done! Output directory: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()