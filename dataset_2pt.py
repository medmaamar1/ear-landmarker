"""
dataset_2pt.py
==============
Minimal Dataset Generator — outputs ONLY 2 landmark points (indices 15 and 19).
Paired with train_2pt.py for the 2-point Direct Regression model.
"""
import os
import json
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence

# ─────────────────────────────────────────────────────────────────────────────
# The two landmark indices we care about:
#   Index 15 → outer rim of helix (top)
#   Index 19 → bottom of helix / junction
# ─────────────────────────────────────────────────────────────────────────────
TARGET_INDICES = [15, 19]


class EarDataset2Pt:
    def __init__(self, data_dir, img_size=224, test_split=0.2):
        self.data_dir  = data_dir
        self.img_size  = img_size
        self.test_split = test_split

        # Support both flat and sub-directory layouts
        images_dir = data_dir
        landmarks_dir = data_dir
        if os.path.isdir(os.path.join(data_dir, 'images')):
            images_dir    = os.path.join(data_dir, 'images')
            landmarks_dir = os.path.join(data_dir, 'annotations') if \
                os.path.isdir(os.path.join(data_dir, 'annotations')) else data_dir

        self.images_dir    = images_dir
        self.landmarks_dir = landmarks_dir

        # Collect all valid (image, json) pairs
        self.filenames = []
        for fn in sorted(os.listdir(images_dir)):
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            base = os.path.splitext(fn)[0]
            json_path = os.path.join(landmarks_dir, base + '.json')
            if os.path.exists(json_path):
                self.filenames.append(base)

        print(f"2-Pt Dataset: Found {len(self.filenames)} valid pairs.")

    def get_generators(self, batch_size=16):
        n = len(self.filenames)
        indices = np.random.permutation(n)
        split   = int(n * (1 - self.test_split))
        train_idxs = indices[:split]
        val_idxs   = indices[split:]
        train_gen = EarGenerator2Pt(self.filenames, train_idxs,
                                    self.images_dir, self.landmarks_dir,
                                    self.img_size, batch_size, augment=True)
        val_gen   = EarGenerator2Pt(self.filenames, val_idxs,
                                    self.images_dir, self.landmarks_dir,
                                    self.img_size, batch_size, augment=False)
        return train_gen, val_gen


class EarGenerator2Pt(Sequence):
    def __init__(self, filenames, indices, img_dir, lm_dir,
                 img_size, batch_size, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.filenames  = filenames
        self.indices    = indices
        self.img_dir    = img_dir
        self.lm_dir     = lm_dir
        self.img_size   = img_size
        self.batch_size = batch_size
        self.augment    = augment

    def __len__(self):
        return max(1, len(self.indices) // self.batch_size)

    def __getitem__(self, idx):
        batch_idxs = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X, Y = [], []
        for i in batch_idxs:
            result = self._load_sample(self.filenames[i])
            if result is not None:
                img, coords = result
                X.append(img)
                Y.append(coords)
        if not X:
            dummy_x = np.zeros((1, self.img_size, self.img_size, 3), dtype=np.float32)
            dummy_y = np.zeros((1, len(TARGET_INDICES) * 2), dtype=np.float32)
            return dummy_x, dummy_y
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    def _load_sample(self, base):
        img_path  = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(self.img_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        json_path = os.path.join(self.lm_dir, base + '.json')
        if img_path is None or not os.path.exists(json_path):
            return None

        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Load all 55 landmarks
        with open(json_path) as f:
            data = json.load(f)
        shapes = sorted(data['shapes'], key=lambda s: int(s['label']))
        all_pts = []
        for s in shapes:
            all_pts.extend(s['points'])
        all_pts = np.array(all_pts)

        if len(all_pts) < max(TARGET_INDICES) + 1:
            return None

        # Extract only the 2 target landmarks, normalize to [0, 1]
        coords = []
        for idx in TARGET_INDICES:
            coords.append(all_pts[idx, 0] / w)  # x
            coords.append(all_pts[idx, 1] / h)  # y
        coords = np.array(coords, dtype=np.float32)  # shape (4,)

        # Optional augmentation (rotation + zoom)
        if self.augment:
            angle = np.random.uniform(-30, 30)
            zoom  = np.random.uniform(0.8, 1.2)
            cx, cy = w / 2, h / 2
            M = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
            # Transform each point
            for i, target_idx in enumerate(TARGET_INDICES):
                px = all_pts[target_idx, 0]
                py = all_pts[target_idx, 1]
                pt = np.dot(M, np.array([px, py, 1.0]))
                coords[i * 2]     = np.clip(pt[0] / w, 0.0, 1.0)
                coords[i * 2 + 1] = np.clip(pt[1] / h, 0.0, 1.0)

        img = cv2.resize(img, (self.img_size, self.img_size))
        return img.astype(np.float32), coords
