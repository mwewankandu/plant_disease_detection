# ml_model/create_mini_dataset.py
import os
import shutil
import requests
from PIL import Image
import numpy as np

def create_mini_dataset():
    # We'll simulate with 3 classes for testing
    classes = ['Tomato_Early_Blight', 'Tomato_Healthy', 'Potato_Late_Blight']
    
    # Create directory structure
    base_dir = 'dataset_mini'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    
    for dir_path in [train_dir, val_dir]:
        for cls in classes:
            os.makedirs(os.path.join(dir_path, cls), exist_ok=True)
    
    # Create synthetic images (for testing without actual dataset)
    # In real scenario, you'd use actual images
    print("Mini dataset structure created!")
    print(f"Place your images in: {train_dir}/class_name/")
    
    return base_dir