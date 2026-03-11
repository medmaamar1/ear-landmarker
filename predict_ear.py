import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import json

def load_ground_truth(json_path):
    """
    Parses LabelMe JSON to get landmark coordinates.
    """
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path) as f:
            data = json.load(f)
            
        if 'shapes' in data:
            # Sort shapes by label to ensure consistent order
            def get_label_id(s):
                label = s.get('label', '')
                try: return int(label)
                except: return 0
            shapes = sorted(data['shapes'], key=get_label_id)
            all_pts = []
            for s in shapes:
                all_pts.extend(s['points'])
            return np.array(all_pts).reshape(-1, 2)
        else:
            pts = data.get('landmarks') or data.get('pts') or data.get('points')
            if pts:
                return np.array(pts).reshape(-1, 2)
    except Exception as e:
        print(f"Error loading JSON: {e}")
    return None

def draw_landmarks(image, landmarks, color_scheme='ai'):
    """
    Draws landmarks on an image.
    color_scheme: 'ai' (Green/Red) or 'truth' (Blue/Yellow)
    """
    vis = image.copy()
    h, w, _ = vis.shape
    
    is_normalized = np.max(landmarks) <= 1.2 # heuristic to check if [0, 1] range
    
    for i, pt in enumerate(landmarks):
        px = int(pt[0] * w) if is_normalized else int(pt[0])
        py = int(pt[1] * h) if is_normalized else int(pt[1])
        
        if color_scheme == 'ai':
            color = (0, 0, 255) if i == 48 else (0, 255, 0) # Red for lobe, Green for others
        else:
            color = (0, 255, 255) if i == 48 else (255, 0, 0) # Purple/Blue for truth
            
        cv2.circle(vis, (px, py), max(2, w // 64), color, -1)
    return vis

def predict_ear(image_input, model_path=None, output_path_prefix=None):
    """
    Loads model, predicts landmarks, and compares with JSON if available.
    """
    # 1. Path Setup
    if model_path is None or model_path == 'ear_landmarker_final.keras':
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'ear_landmarker_final.keras')

    if not os.path.exists(model_path):
        if os.path.exists('ear_landmarker_final.keras'):
            model_path = 'ear_landmarker_final.keras'
        else:
            print(f"Error: Model not found at {model_path}.")
            return None

    # 2. Input Handling
    json_data = None
    name = "result"
    
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            print(f"Error: Image not found at {image_input}")
            return None
        img = cv2.imread(image_input)
        base = os.path.basename(image_input)
        name, _ = os.path.splitext(base)
        
        # Check for matching JSON
        json_path = os.path.join(os.path.dirname(image_input), name + ".json")
        json_data = load_ground_truth(json_path)
    else:
        img = image_input.copy()

    if img is None:
        print("Error: Could not decode image.")
        return None

    # 3. Predict
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    img_size = 128
    h, w, _ = img.shape
    
    input_img = cv2.resize(img, (img_size, img_size))
    input_img_norm = input_img.astype(np.float32)
    input_tensor = np.expand_dims(input_img_norm, axis=0)

    print("Running inference...")
    preds = model.predict(input_tensor)[0]
    landmarks_ai = preds.reshape(-1, 2)

    # 4. Generate Visuals
    ai_vis = draw_landmarks(img, landmarks_ai, color_scheme='ai')
    
    output_files = []
    
    # Save AI Result
    ai_out = f"pred_{name}.jpg"
    cv2.imwrite(ai_out, ai_vis)
    output_files.append(ai_out)
    
    # Save Truth Result if available
    if json_data is not None:
        truth_vis = draw_landmarks(img, json_data, color_scheme='truth')
        truth_out = f"truth_{name}.jpg"
        cv2.imwrite(truth_out, truth_vis)
        output_files.append(truth_out)
        print(f"Ground truth found! Saved comparison: {truth_out}")
        
        # Optional: Combine them side-by-side
        combined = np.hstack((truth_vis, ai_vis))
        comb_out = f"compare_{name}.jpg"
        cv2.imwrite(comb_out, combined)
        output_files.append(comb_out)
        print(f"Saved side-by-side: {comb_out}")

    print(f"Success! Prediction images saved: {', '.join(output_files)}")
    return ai_vis

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_ear.py <path_to_image> [path_to_model]")
    else:
        m_path = sys.argv[2] if len(sys.argv) > 2 else None
        predict_ear(sys.argv[1], model_path=m_path)
