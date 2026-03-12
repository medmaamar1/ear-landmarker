"""
show_landmarks.py
=================
Draws ALL 55 landmarks from the JSON ground-truth on a sample ear image,
with each point labeled by its index number.

Use this to decide which landmark indices to keep for your AR app.

Usage:
  python show_landmarks.py <image_path>

Example:
  python show_landmarks.py /kaggle/input/datasets/maamarmohamed/ear-landmark/00001.png

Output:
  landmarks_numbered.jpg  (high-res image with colored, numbered dots)
"""

import sys
import os
import json
import cv2
import numpy as np

# ─── 0. PARSE INPUT ──────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python show_landmarks.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
base = os.path.splitext(os.path.basename(img_path))[0]
json_path = os.path.join(os.path.dirname(img_path), base + ".json")

if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    sys.exit(1)
if not os.path.exists(json_path):
    print(f"JSON not found: {json_path}")
    sys.exit(1)

# ─── 1. LOAD IMAGE & LANDMARKS ───────────────────────────────────────────────
img = cv2.imread(img_path)
h, w = img.shape[:2]

with open(json_path) as f:
    data = json.load(f)

# Collect all points in order (sorted by shape label so order is stable)
shapes = sorted(data["shapes"], key=lambda s: (int(s["label"]) if s["label"].isdigit() else 0))
landmarks = []
for shape in shapes:
    for pt in shape["points"]:
        landmarks.append(pt)

landmarks = np.array(landmarks)   # shape (55, 2)
print(f"Loaded {len(landmarks)} landmarks from {json_path}")

# ─── 2. DRAW ─────────────────────────────────────────────────────────────────
# Upscale for legibility if the image is tiny
DISPLAY_SIZE = 800
scale = DISPLAY_SIZE / max(h, w)
canvas = cv2.resize(img, (int(w * scale), int(h * scale)))

# Color palette: cycle through 5 distinct colors so adjacent points contrast
COLORS = [
    (0,  255,  0),    # Green
    (0,  180, 255),   # Orange
    (255, 80, 255),   # Magenta
    (80, 255, 255),   # Cyan
    (255, 255,  60),  # Yellow
]

for i, (lx, ly) in enumerate(landmarks):
    px = int(lx * scale)
    py = int(ly * scale)
    color = COLORS[i % len(COLORS)]

    # Draw filled circle
    cv2.circle(canvas, (px, py), 7, color, -1)
    # White outline for visibility
    cv2.circle(canvas, (px, py), 7, (255, 255, 255), 1)

    # Draw index number
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = str(i)
    font_scale = 0.38
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Black background box for legibility
    cv2.rectangle(canvas, (px + 6, py - th - 3), (px + 6 + tw + 2, py + 2), (0, 0, 0), -1)
    cv2.putText(canvas, label, (px + 7, py), font, font_scale, color, thickness, cv2.LINE_AA)

# ─── 3. SAVE ─────────────────────────────────────────────────────────────────
out_path = "landmarks_numbered.jpg"
cv2.imwrite(out_path, canvas)
print(f"\nSaved: {out_path}")
print("Open the image and note which index numbers correspond to the ear features you need.")
print("\nQuick shape reference:")
print("  Label 0 → pts  0-19  (outer helix, 20 pts)")
print("  Label 1 → pts 20-34  (antihelix / inner, 15 pts)")
print("  Label 2 → pts 35-49  (inner fold, 15 pts)")
print("  Label 3 → pts 50-54  (lobe + tragus, 5 pts)")
