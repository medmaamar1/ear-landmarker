import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import json

def parse_json_landmarks(json_path):
    """
    Parses a LabelMe JSON file into a numpy array of landmarks.
    """
    try:
        if not os.path.exists(json_path):
            return None
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
                return np.array(all_pts).reshape(-1, 2)[:55]
            else:
                pts = data.get('landmarks') or data.get('pts') or data.get('points')
                if pts:
                    return np.array(pts).reshape(-1, 2)[:55]
    except Exception as e:
        print(f"Warning: Failed to parse JSON at {json_path}: {e}")
    return None

def draw_landmarks(img, landmarks, color=(0, 255, 0), lobe_color=(0, 0, 255)):
    """
    Draws landmarks on an image.
    """
    vis_img = img.copy()
    h, w, _ = vis_img.shape
    for i, pt in enumerate(landmarks):
        # We assume coordinates are absolute pixel values
        px, py = int(pt[0]), int(pt[1])
        # If they look normalized (between 0 and 1), we scale them
        if np.max(landmarks) <= 1.01:
            px, py = int(pt[0] * w), int(pt[1] * h)
            
        c = lobe_color if i == 48 else color
        cv2.circle(vis_img, (px, py), max(2, w // 64), c, -1)
    return vis_img

def predict_ear(image_input, model_path=None, output_path=None, show_comparison=True):
    """
    Loads the trained Keras model, predicts landmarks on a single image (path or numpy array).
    If a matching JSON exists, creates a side-by-side comparison.
    Returns the processed image.
    """
    # Use relative path from script location if model_path not provided or default used
    if model_path is None or model_path == 'ear_landmarker_final.keras':
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'ear_landmarker_final.keras')

    if not os.path.exists(model_path):
        # Check current directory as fallback
        if os.path.exists('ear_landmarker_final.keras'):
            model_path = 'ear_landmarker_final.keras'
        else:
            print(f"Error: Model not found at {model_path}. Did you download it or finish training?")
            return None

    # Handle input type and ground truth detection
    ground_truth = None
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            print(f"Error: Image not found at {image_input}")
            return None
        print(f"Reading image: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            print("Error: Could not decode image.")
            return None
        
        # Look for matching JSON
        base = os.path.basename(image_input)
        name, ext = os.path.splitext(base)
        json_path = os.path.join(os.path.dirname(image_input), name + ".json")
        ground_truth = parse_json_landmarks(json_path)

        # Default output name based on input path
        if output_path is None:
            output_path = f"cmp_{name}.jpg" if show_comparison and ground_truth is not None else f"pred_{name}.jpg"
    else:
        # Assume it's a numpy array (image object)
        img = image_input.copy()
        if output_path is None:
            output_path = "prediction_result.jpg"

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    img_size = 128
    h, w, _ = img.shape
    
    # Preprocess
    input_img = cv2.resize(img, (img_size, img_size))
    input_img_norm = input_img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_img_norm, axis=0)

    # Predict
    print("Running inference...")
    preds = model.predict(input_tensor)[0]
    landmarks_pred = preds.reshape(-1, 2)

    # Draw prediction
    pred_vis = draw_landmarks(img, landmarks_pred)

    final_vis = pred_vis
    if show_comparison and ground_truth is not None:
        print("Matching JSON found! Creating comparison...")
        gt_vis = draw_landmarks(img, ground_truth, color=(255, 0, 0), lobe_color=(0, 0, 255)) # Blue for GT
        
        # Add labels
        cv2.putText(gt_vis, "GROUND TRUTH (JSON)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(pred_vis, "AI PREDICTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Concatenate side-by-side
        final_vis = np.hstack((gt_vis, pred_vis))

    if output_path:
        cv2.imwrite(output_path, final_vis)
        print(f"Success! Result saved to: {output_path}")
    
    return final_vis

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_ear.py <path_to_image> [path_to_model]")
    else:
        m_path = sys.argv[2] if len(sys.argv) > 2 else None
        predict_ear(sys.argv[1], model_path=m_path)
