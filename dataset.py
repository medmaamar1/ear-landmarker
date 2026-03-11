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
            
        print(f"DEBUG: Dataset root: {data_dir}")
        print(f"DEBUG: Images dir: {self.images_dir}")
        print(f"DEBUG: Landmarks dir: {self.landmarks_dir}")
        
        # list all image files
        all_img_files = [os.path.splitext(f)[0] for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))]
        
        # Filter: Only keep files that have both image and landmark
        self.filenames = []
        for f in all_img_files:
            has_lm = any(os.path.exists(os.path.join(self.landmarks_dir, f + ext)) for ext in ['.txt', '.pts', '.json'])
            if has_lm:
                self.filenames.append(f)
        
        print(f"DEBUG: Found {len(self.filenames)} valid pairs out of {len(all_img_files)} images.")

    def get_generators(self, batch_size=32):
        indices = np.arange(len(self.filenames))
        np.random.shuffle(indices)
        
        split = int(len(self.filenames) * (1 - self.test_split))
        train_idxs = indices[:split]
        val_idxs = indices[split:]
        
        # Training gets augmentation, validation does not
        train_gen = EarGenerator(self.filenames, train_idxs, self.images_dir, self.landmarks_dir, self.img_size, batch_size, augment=True)
        val_gen = EarGenerator(self.filenames, val_idxs, self.images_dir, self.landmarks_dir, self.img_size, batch_size, augment=False)
        
        return train_gen, val_gen
class EarGenerator(Sequence):
    def __init__(self, filenames, indices, img_dir, lm_dir, img_size, batch_size, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.filenames = filenames
        self.indices = indices
        self.img_dir = img_dir
        self.lm_dir = lm_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_idxs = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_files = [self.filenames[i] for i in batch_idxs]
        
        X = []
        Y = []
        
        for f in batch_files:
            img_path = os.path.join(self.img_dir, f + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, f + '.png')
            
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            
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
                    lms = lms[:55]
                except: continue
            else:
                try:
                    import json
                    json_path = os.path.join(self.lm_dir, f + '.json')
                    if os.path.exists(json_path):
                        with open(json_path) as jf:
                            data = json.load(jf)
                            if 'shapes' in data:
                                def get_label_id(s):
                                    label = s.get('label', '')
                                    try: return int(label)
                                    except: return 0
                                shapes = sorted(data['shapes'], key=get_label_id)
                                all_pts = []
                                for s in shapes:
                                    all_pts.extend(s['points'])
                                lms = np.array(all_pts).reshape(-1, 2)[:55]
                            else:
                                pts = data.get('landmarks') or data.get('pts') or data.get('points')
                                if pts:
                                    lms = np.array(pts).reshape(-1, 2)[:55]
                except: continue

            if lms is None or len(lms) < 55: continue

            # --- AUGMENTATION & NORMALIZATION ---
            
            # 1. Spatial Augmentation (Rotation, Shift, Zoom)
            if self.augment:
                # Rotation ±30 deg
                angle = np.random.uniform(-30, 30)
                # Zoom/Scale (0.9 to 1.1)
                scale = np.random.uniform(0.9, 1.1)
                # Translation/Shift (±10% of image size)
                tx = np.random.uniform(-0.1, 0.1) * w
                ty = np.random.uniform(-0.1, 0.1) * h
                
                # Get rotation matrix around image center
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                
                # Apply transformation to image
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                
                # Apply transformation to landmarks
                # Extend points to [x, y, 1] for matrix multiplication
                ones = np.ones(shape=(len(lms), 1))
                pts_ext = np.hstack([lms, ones])
                lms = (M @ pts_ext.T).T

            crop = img.copy()
            crop_h, crop_w, _ = crop.shape
            
            # 2. Normalize Landmarks relative to FULL image dimensions
            lms_norm = lms.copy()
            lms_norm[:, 0] = lms[:, 0] / crop_w
            lms_norm[:, 1] = lms[:, 1] / crop_h
            
            # 3. Color Jitter
            if self.augment:
                alpha = np.random.uniform(0.8, 1.2) # contrast
                beta = np.random.uniform(-20, 20)   # brightness
                crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
            
            # 4. Resize
            img_resized = cv2.resize(crop, (self.img_size, self.img_size))
            X.append(img_resized.astype(np.float32))
            Y.append(lms_norm.flatten().astype(np.float32))
            
        if not X:
            # Instead of zeros, which ruins training, return the next batch
            # or raise an error if this is a systemic issue.
            # Returning a smaller batch is better than returning zeros.
            print(f"WARNING: Batch at index {idx} was empty/invalid. Check your data paths.")
            # Simple fallback: recurrence
            return self.__getitem__((idx + 1) % len(self))

        return (np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32))
