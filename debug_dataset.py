import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import EarDataset

def visualize_generator(data_dir, output_file="debug_dataset.jpg"):
    """
    Pulls a batch of images and targets DIRECTLY from the generator
    and plots them to verify what the model is actually seeing.
    """
    print(f"Initializing Dataset at {data_dir}...")
    dataset = EarDataset(data_dir, img_size=224, test_split=0.1)
    train_gen, _ = dataset.get_generators(batch_size=4)
    
    print("Pulling first batch...")
    X, Y = train_gen[0]
    
    print(f"Batch X shape: {X.shape}, min: {np.min(X)}, max: {np.max(X)}")
    print(f"Batch Y shape: {Y.shape}, min: {np.min(Y)}, max: {np.max(Y)}")
    
    # Create a 2x2 grid for the 4 images
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(len(X)):
        img = X[i].astype(np.uint8) # Original image is float32 [0, 255]
        lms = Y[i].reshape(-1, 2)
        
        # Plot Image
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Plot Landmarks (Un-normalize by multiplying by resolution)
        for j, pt in enumerate(lms):
            x = pt[0] * 224
            y = pt[1] * 224
            axes[i].plot(x, y, 'ro', markersize=3)
            if j == 48: # Mark lobe specifically
                axes[i].plot(x, y, 'go', markersize=5)
                
        axes[i].set_title(f"Sample {i}")
        
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Visual verification saved to {output_file}")

if __name__ == "__main__":
    import sys
    d_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    visualize_generator(d_dir)
