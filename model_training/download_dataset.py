#!/usr/bin/env python3
"""
Dataset Download Script for Human Detection Training

This script helps download and prepare datasets for training the human detection model.
It downloads sample images from publicly available datasets.

Usage:
    python download_dataset.py --output_dir ./data
"""

import os
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

def download_file(url, output_path):
    """Download a file from URL."""
    print(f"Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def create_directory_structure(base_dir):
    """Create the required directory structure."""
    dirs = [
        os.path.join(base_dir, 'train', 'human'),
        os.path.join(base_dir, 'train', 'non_human'),
        os.path.join(base_dir, 'validation', 'human'),
        os.path.join(base_dir, 'validation', 'non_human'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Created directory structure in: {base_dir}")

def download_lfw_dataset(output_dir):
    """
    Download LFW (Labeled Faces in the Wild) dataset for human face images.
    This provides a good source of human images.
    """
    print("\n=== Downloading LFW Dataset (Human Faces) ===")
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    tar_path = os.path.join(output_dir, "lfw.tgz")
    
    if download_file(url, tar_path):
        print("Extracting LFW dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_dir)
        os.remove(tar_path)
        print("LFW dataset extracted successfully")
        return os.path.join(output_dir, 'lfw')
    return None

def prepare_human_images(lfw_dir, output_dir, train_count=3000, val_count=500):
    """
    Copy human images to train/validation directories.
    """
    if not lfw_dir or not os.path.exists(lfw_dir):
        print("LFW directory not found, skipping human images preparation")
        return
    
    print(f"\nPreparing human images from LFW...")
    
    # Collect all image paths
    all_images = []
    for person_dir in os.listdir(lfw_dir):
        person_path = os.path.join(lfw_dir, person_dir)
        if os.path.isdir(person_path):
            for img in os.listdir(person_path):
                if img.endswith('.jpg'):
                    all_images.append(os.path.join(person_path, img))
    
    print(f"Found {len(all_images)} human images")
    
    # Copy to train and validation
    import random
    random.shuffle(all_images)
    
    train_images = all_images[:min(train_count, len(all_images))]
    val_images = all_images[train_count:train_count + min(val_count, len(all_images) - train_count)]
    
    train_human_dir = os.path.join(output_dir, 'train', 'human')
    val_human_dir = os.path.join(output_dir, 'validation', 'human')
    
    for i, img_path in enumerate(train_images):
        shutil.copy(img_path, os.path.join(train_human_dir, f'human_{i:05d}.jpg'))
    
    for i, img_path in enumerate(val_images):
        shutil.copy(img_path, os.path.join(val_human_dir, f'human_{i:05d}.jpg'))
    
    print(f"Copied {len(train_images)} images to train/human")
    print(f"Copied {len(val_images)} images to validation/human")

def create_sample_non_human_images(output_dir, train_count=3000, val_count=500):
    """
    Create placeholder non-human images.
    In production, you should download real datasets like:
    - Places365: http://places2.csail.mit.edu/
    - Textures: https://www.robots.ox.ac.uk/~vgg/data/dtd/
    - CIFAR-10 (non-person classes): https://www.cs.toronto.edu/~kriz/cifar.html
    """
    print("\n=== Creating Non-Human Image Placeholders ===")
    print("NOTE: For best results, manually add real non-human images from:")
    print("  - Places365: http://places2.csail.mit.edu/")
    print("  - DTD Textures: https://www.robots.ox.ac.uk/~vgg/data/dtd/")
    print("  - ImageNet (nature, objects categories)")
    
    try:
        from PIL import Image
        import numpy as np
        
        train_dir = os.path.join(output_dir, 'train', 'non_human')
        val_dir = os.path.join(output_dir, 'validation', 'non_human')
        
        print(f"Creating {train_count} synthetic non-human training images...")
        for i in range(train_count):
            # Create random pattern images
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(train_dir, f'non_human_{i:05d}.jpg'))
        
        print(f"Creating {val_count} synthetic non-human validation images...")
        for i in range(val_count):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(val_dir, f'non_human_{i:05d}.jpg'))
        
        print("Synthetic non-human images created")
        print("\nWARNING: Replace these with real non-human images for production use!")
        
    except ImportError:
        print("Pillow not installed. Please install with: pip install Pillow")
        print("Or manually add non-human images to the directories.")

def print_dataset_summary(output_dir):
    """Print summary of the prepared dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    for split in ['train', 'validation']:
        for category in ['human', 'non_human']:
            path = os.path.join(output_dir, split, category)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"{split}/{category}: {count} images")
    
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── human/")
    print(f"  │   └── non_human/")
    print(f"  └── validation/")
    print(f"      ├── human/")
    print(f"      └── non_human/")

def main():
    parser = argparse.ArgumentParser(description='Download datasets for human detection training')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for datasets')
    parser.add_argument('--skip_lfw', action='store_true',
                       help='Skip downloading LFW dataset')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Human Detection Dataset Preparation")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure(args.output_dir)
    
    # Download LFW dataset
    lfw_dir = None
    if not args.skip_lfw:
        lfw_dir = download_lfw_dataset(args.output_dir)
    
    # Prepare human images
    prepare_human_images(lfw_dir, args.output_dir)
    
    # Create non-human image placeholders
    create_sample_non_human_images(args.output_dir)
    
    # Clean up LFW directory
    if lfw_dir and os.path.exists(lfw_dir):
        shutil.rmtree(lfw_dir)
    
    # Print summary
    print_dataset_summary(args.output_dir)
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Replace synthetic non-human images with real images")
    print("2. Run: python train_with_real_data.py --data_dir ./data")

if __name__ == '__main__':
    main()
