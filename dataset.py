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
        
        # Flexibly detect directory structure
        potential_images_dir = os.path.join(data_dir, 'images')
        potential_landmarks_dir = os.path.join(data_dir, 'landmarks')
        
        if os.path.exists(potential_images_dir):
            self.images_dir = potential_images_dir
            self.landmarks_dir = potential_landmarks_dir if os.path.exists(potential_landmarks_dir) else potential_images_dir
        else:
            self.images_dir = data_dir
            self.landmarks_dir = data_dir
            
        print(f"Scanning {self.images_dir} for valid image/landmark pairs...")
        all_img_files = [os.path.splitext(f)[0] for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))]
        
        # Filter: Only keep files that have both image and landmark
        self.filenames = []
        for f in all_img_files:
            has_lm = any(os.path.exists(os.path.join(self.landmarks_dir, f + ext)) for ext in ['.txt', '.pts', '.json'])
            if has_lm:
                self.filenames.append(f)
        
        print(f"Found {len(self.filenames)} valid pairs out of {len(all_img_files)} images.")

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
    def __init__(self, filenames, indices, img_dir, lm_dir, img_size, batch_size, augment=False, **kwargs):
        super().__init__(**kwargs) # Required for Keras 3
        self.filenames = filenames
        self.indices = indices
        self.img_dir = img_dir
        self.lm_dir = lm_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch for better generalization."""
        np.random.shuffle(self.indices)
    
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
            
            # CRITICAL: Convert BGR to RGB for consistency with Browser/TF.js
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            
            # 2. Load Landmarks (Robust point collection)
            lms = None
            lm_path = os.path.join(self.lm_dir, f + '.txt')
            if not os.path.exists(lm_path):
                lm_path = os.path.join(self.lm_dir, f + '.pts')
                
            if os.path.exists(lm_path):
                try:
                    raw = []
                    with open(lm_path) as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    raw.append([float(parts[0]), float(parts[1])])
                                except: continue
                    lms = np.array(raw)
                    lms = lms[:55] # Ensure 55 landmarks
                except: continue
            else:
                # Fallback for JSON / LabelMe format
                try:
                    import json
                    json_path = os.path.join(self.landmarks_dir, f + '.json')
                    if os.path.exists(json_path):
                        with open(json_path) as jf:
                            data = json.load(jf)
                            
                            # Check for LabelMe 'shapes' structure
                            if 'shapes' in data:
                                # USE NUMERICAL SORTING for labels to ensure exact point order
                                def get_label_id(s):
                                    label = s.get('label', '')
                                    try: return int(label)
                                    except: return label
                                    
                                shapes = sorted(data['shapes'], key=get_label_id)
                                all_pts = []
                                for s in shapes:
                                    all_pts.extend(s['points'])
                                lms = np.array(all_pts).reshape(-1, 2)[:55]
                            else:
                                # Check for flat 'landmarks' or 'pts' keys
                                pts = data.get('landmarks') or data.get('pts') or data.get('points')
                                if pts:
                                    lms = np.array(pts).reshape(-1, 2)[:55]
                except: continue

            if lms is None or len(lms) < 55: continue

            # 3. Dynamic ROI Cropping with JITTER
            min_x, min_y = np.min(lms, axis=0)
            max_x, max_y = np.max(lms, axis=0)
            lm_w, lm_h = max_x - min_x, max_y - min_y
            
            pad_w = lm_w * np.random.uniform(0.15, 0.45)
            pad_h = lm_h * np.random.uniform(0.15, 0.45)
            
            shift_x = lm_w * np.random.uniform(-0.1, 0.1)
            shift_y = lm_h * np.random.uniform(-0.1, 0.1)
            
            roi_x1 = max(0, int(min_x - pad_w + shift_x))
            roi_y1 = max(0, int(min_y - pad_h + shift_y))
            roi_x2 = min(w, int(max_x + pad_w + shift_x))
            roi_y2 = min(h, int(max_y + pad_h + shift_y))
            
            crop = img[roi_y1:roi_y2, roi_x1:roi_x2]
            if crop.size == 0: continue
            crop_h, crop_w, _ = crop.shape
            
            lms_norm = lms.copy()
            lms_norm[:, 0] = (lms[:, 0] - roi_x1) / crop_w
            lms_norm[:, 1] = (lms[:, 1] - roi_y1) / crop_h
            
            # 4. Final Preprocessing
            img_resized = cv2.resize(crop, (self.img_size, self.img_size))
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            X.append(img_normalized)
            Y.append(lms_norm.flatten().astype(np.float32))
            
        if not X:
            return np.zeros((1, self.img_size, self.img_size, 3), dtype=np.float32), \
                   np.zeros((1, 110), dtype=np.float32)

        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
