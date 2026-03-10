import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class EarDataset:
    def __init__(self, data_dir, img_size=128, test_split=0.2):
        self.data_dir = data_dir
        self.img_size = img_size
        self.test_split = test_split
        self.images_dir = os.path.join(data_dir, 'images')
        self.landmarks_dir = os.path.join(data_dir, 'landmarks')
        self.filenames = [f.split('.')[0] for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))]
        
    def get_generators(self, batch_size=32):
        indices = np.arange(len(self.filenames))
        np.random.shuffle(indices)
        
        split = int(len(self.filenames) * (1 - self.test_split))
        train_idxs = indices[:split]
        val_idxs = indices[split:]
        
        train_gen = EarGenerator(self.filenames, train_idxs, self.images_dir, self.landmarks_dir, self.img_size, batch_size, augment=True)
        val_gen = EarGenerator(self.filenames, val_idxs, self.images_dir, self.landmarks_dir, self.img_size, batch_size, augment=False)
        
        return train_gen, val_gen

class EarGenerator(Sequence):
    def __init__(self, filenames, indices, img_dir, lm_dir, img_size, batch_size, augment=False):
        self.filenames = filenames
        self.indices = indices
        self.img_dir = img_dir
        self.lm_dir = lm_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_idxs = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_files = [self.filenames[i] for i in batch_idxs]
        
        X = []
        Y = []
        
        for f in batch_files:
            # 1. Load Image
            img_path = os.path.join(self.img_dir, f + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, f + '.png')
            
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape
            
            # 2. Load Landmarks (Assumed x1 y1 x2 y2... or JSON)
            lm_path = os.path.join(self.lm_dir, f + '.txt')
            if os.path.exists(lm_path):
                lms = np.loadtxt(lm_path).reshape(-1, 2)
            else:
                # Fallback for JSON format if found in AudioEar2D
                import json
                with open(os.path.join(self.lm_dir, f + '.json')) as jf:
                    data = json.load(jf)
                    lms = np.array(data['landmarks']).reshape(-1, 2)

            # 3. Dynamic ROI Cropping (CRITICAL for matching browser ROI)
            # Find the bounding box of landmarks
            min_x, min_y = np.min(lms, axis=0)
            max_x, max_y = np.max(lms, axis=0)
            
            lm_w, lm_h = max_x - min_x, max_y - min_y
            
            # Add random padding (mimics the FaceMesh-based ear ROI)
            pad_w = lm_w * np.random.uniform(0.2, 0.5)
            pad_h = lm_h * np.random.uniform(0.2, 0.5)
            
            # Final ROI box
            roi_x1 = max(0, int(min_x - pad_w))
            roi_y1 = max(0, int(min_y - pad_h))
            roi_x2 = min(w, int(max_x + pad_w))
            roi_y2 = min(h, int(max_y + pad_h))
            
            # Crop
            crop = img[roi_y1:roi_y2, roi_x1:roi_x2]
            crop_h, crop_w, _ = crop.shape
            
            # Normalize Landmarks relative to Crop [0, 1]
            lms_norm = lms.copy()
            lms_norm[:, 0] = (lms[:, 0] - roi_x1) / crop_w
            lms_norm[:, 1] = (lms[:, 1] - roi_y1) / crop_h
            
            # 4. Final Preprocessing
            img_resized = cv2.resize(crop, (self.img_size, self.img_size))
            img_normalized = img_resized / 255.0
            
            X.append(img_normalized)
            Y.append(lms_norm.flatten())
            
        return np.array(X), np.array(Y)
