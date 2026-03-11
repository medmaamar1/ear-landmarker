import os
import cv2
import numpy as np
import json
import argparse
import sys

def debug_augmentation(image_path, num_samples=6):
    """
    Loads an image and its JSON, applies the specialized augmentations,
    and returns a combined visual to prove landmarks follow the ear.
    """
    # 1. Load Data
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(os.path.dirname(image_path), name + ".json")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON {json_path} not found.")
        return

    # Extract Landmarks from JSON (Same logic as dataset.py)
    with open(json_path) as f:
        data = json.load(f)
        def get_label_id(s):
            try: return int(s.get('label', ''))
            except: return 0
        shapes = sorted(data['shapes'], key=get_label_id)
        all_pts = []
        for s in shapes:
            all_pts.extend(s['points'])
        lms_orig = np.array(all_pts).reshape(-1, 2)

    results = []
    
    # 2. Generate Augmented Samples
    for i in range(num_samples):
        # Sample parameters
        angle = np.random.uniform(-30, 30) # Increased range for clarity
        scale = np.random.uniform(0.7, 1.3)
        tx = np.random.uniform(-0.15, 0.15) * w
        ty = np.random.uniform(-0.15, 0.15) * h
        
        # Create Transformation Matrix
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Transform Image
        aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Transform Landmarks
        ones = np.ones(shape=(len(lms_orig), 1))
        pts_ext = np.hstack([lms_orig, ones])
        aug_lms = (M @ pts_ext.T).T
        
        # Draw for verification
        vis = aug_img.copy()
        for j, pt in enumerate(aug_lms):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        
        # Resize for the final grid
        vis = cv2.resize(vis, (256, 256))
        results.append(vis)

    # 3. Combine and Save
    # Create 2 rows of 3
    row1 = np.hstack(results[:3])
    row2 = np.hstack(results[3:])
    grid = np.vstack([row1, row2])
    
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    out_name = "debug_aug_proof.jpg"
    cv2.imwrite(out_name, grid_bgr)
    print(f"Success! Visual proof saved to: {out_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_augmentation.py <image_path>")
    else:
        debug_augmentation(sys.argv[1])
