"""
Phase 22 Sanity Check
======================
Run this BEFORE training to ensure the entire pipeline is wired correctly.

Tests:
  [1] Dataset loads and outputs correct shapes
  [2] Heatmaps are correctly generated (one bright spot per landmark)
  [3] Model builds and outputs (56, 56, 55) tensor
  [4] A full forward pass succeeds (1 batch, no crash)
  [5] Visual confirmation image saved as 'sanity_output.jpg'

Usage:
  python sanity_check.py /kaggle/input/datasets/maamarmohamed/ear-landmark
"""

import sys
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless (no display needed on Kaggle)
import matplotlib.pyplot as plt

PASS  = "[PASS]"
FAIL  = "[FAIL]"
INFO  = "[INFO]"

errors = []

def check(condition, label, detail=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f": {detail}" if detail else ""))
        errors.append(label)

# ─── 1. DATASET ───────────────────────────────────────────────────────────────
print("\n[1] DATASET")
try:
    # Allow the script to run from any directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from dataset import EarDataset

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    ds = EarDataset(data_dir, img_size=224, heatmap_size=56, test_split=0.2)
    train_gen, val_gen = ds.get_generators(batch_size=4)

    check(len(ds.filenames) > 0,    "Dataset found images", f"Found {len(ds.filenames)}")
    check(len(train_gen) > 0,       "Training batches created")
    check(len(val_gen) > 0,         "Validation batches created")
    print(f"  {INFO}  Total images: {len(ds.filenames)}")
    print(f"  {INFO}  Training batches: {len(train_gen)}  |  Val batches: {len(val_gen)}")
except Exception as e:
    print(f"  {FAIL}  Dataset load crashed: {e}")
    errors.append("Dataset load")

# ─── 2. HEATMAP SHAPE & VALUES ────────────────────────────────────────────────
print("\n[2] HEATMAP GENERATION")
try:
    X, Y = train_gen[0]

    check(X.shape == (4, 224, 224, 3),  f"Image batch shape {X.shape} == (4, 224, 224, 3)")
    check(Y.shape == (4, 56, 56, 55),   f"Heatmap batch shape {Y.shape} == (4, 56, 56, 55)")
    check(float(np.min(X)) >= 0.0 and float(np.max(X)) <= 255.0, f"Image pixel range [{np.min(X):.0f}, {np.max(X):.0f}] ⊆ [0, 255]")
    check(float(np.min(Y)) >= 0.0 and float(np.max(Y)) <= 1.0,   f"Heatmap value range [{np.min(Y):.4f}, {np.max(Y):.4f}] ⊆ [0, 1]")

    # How many of the 55 heatmaps have a non-zero peak? (should be 55 if all in-bounds)
    hmap_sample = Y[0]  # shape (56, 56, 55)
    active = sum(np.max(hmap_sample[:, :, i]) > 0.5 for i in range(55))
    check(active >= 50, f"Active heatmap channels (peak > 0.5): {active}/55")
except Exception as e:
    print(f"  {FAIL}  Heatmap check crashed: {e}")
    errors.append("Heatmap generation")

# ─── 3. MODEL BUILD ───────────────────────────────────────────────────────────
print("\n[3] MODEL BUILD (Tiny U-Net)")
try:
    import tensorflow as tf
    from train import build_heatmap_model
    model = build_heatmap_model(input_shape=(224, 224, 3), num_landmarks=55)
    out_shape = model.output_shape  # (None, 56, 56, 55)
    check(out_shape == (None, 56, 56, 55), f"Model output shape {out_shape} == (None, 56, 56, 55)")

    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    total_params = model.count_params()
    print(f"  {INFO}  Trainable params  : {trainable_params:,}")
    print(f"  {INFO}  Total params      : {total_params:,}")
    check(total_params < 10_000_000, f"Total params < 10M (actual: {total_params:,})")
except Exception as e:
    print(f"  {FAIL}  Model build crashed: {e}")
    errors.append("Model build")

# ─── 4. FORWARD PASS ──────────────────────────────────────────────────────────
print("\n[4] FORWARD PASS")
try:
    dummy = np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
    pred = model.predict(dummy, verbose=0)
    check(pred.shape == (1, 56, 56, 55),  f"Prediction shape {pred.shape} == (1, 56, 56, 55)")
    check(float(np.min(pred)) >= 0.0 and float(np.max(pred)) <= 1.0,
          f"Prediction range [{np.min(pred):.4f}, {np.max(pred):.4f}] ⊆ [0, 1]")
except Exception as e:
    print(f"  {FAIL}  Forward pass crashed: {e}")
    errors.append("Forward pass")

# ─── 5. VISUAL SANITY ─────────────────────────────────────────────────────────
print("\n[5] VISUAL SANITY (saving sanity_output.jpg)")
try:
    X_batch, Y_batch = val_gen[0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Phase 22 Sanity Check – Input Images vs Heatmaps', fontsize=14)

    for i in range(4):
        # Top row: the raw training image
        ax_img = axes[0, i]
        ax_img.imshow(X_batch[i].astype(np.uint8))
        ax_img.set_title(f"Image {i}")
        ax_img.axis('off')

        # Overlay landmarks extracted from heatmaps
        lms_screen = []
        for li in range(55):
            hmap = Y_batch[i, :, :, li]
            if np.max(hmap) > 0:
                idx = np.unravel_index(np.argmax(hmap), hmap.shape)
                # Map from heatmap space (56x56) to image display space (224x224)
                x = idx[1] / 56 * 224
                y = idx[0] / 56 * 224
                lms_screen.append((x, y))

        if lms_screen:
            xs, ys = zip(*lms_screen)
            ax_img.scatter(xs, ys, c='lime', s=6, zorder=5)

        # Bottom row: Sum of all 55 heatmaps (shows the "footprint")
        ax_hmap = axes[1, i]
        combined = np.sum(Y_batch[i], axis=-1)  # (56, 56)
        ax_hmap.imshow(combined, cmap='hot')
        ax_hmap.set_title(f"Heatmap Sum {i}")
        ax_hmap.axis('off')

    plt.tight_layout()
    plt.savefig('sanity_output.jpg', dpi=100)
    plt.close()
    print(f"  {PASS}  Saved sanity_output.jpg")
    print(f"  {INFO}  Open the image and verify green dots sit ON the ear contours.")
except Exception as e:
    print(f"  {FAIL}  Visual output crashed: {e}")
    errors.append("Visual output")

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "="*55)
if not errors:
    print("  ALL CHECKS PASSED — safe to start training! 🚀")
else:
    print(f"  {len(errors)} CHECK(S) FAILED:")
    for e in errors:
        print(f"    • {e}")
    print("  Fix the issues above before running train.py")
print("="*55 + "\n")
