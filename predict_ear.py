import os
import cv2
import numpy as np
import tensorflow as tf
import sys

def predict_ear(image_input, model_path=None, output_path=None):
    """
    Loads the trained Keras model, predicts landmarks on a single image (path or numpy array).
    Returns the image with landmarks drawn.
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

    # Handle input type
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            print(f"Error: Image not found at {image_input}")
            return None
        print(f"Reading image: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            print("Error: Could not decode image.")
            return None
        
        # Default output name based on input path
        if output_path is None:
            base = os.path.basename(image_input)
            name, ext = os.path.splitext(base)
            output_path = f"pred_{name}.jpg"
    else:
        # Assume it's a numpy array (image object)
        img = image_input.copy()
        if output_path is None:
            output_path = "prediction_result.jpg"

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    img_size = 128
    
    h, w, _ = img.shape
    
    # Preprocess: Resize to 128x128 and normalize to [0, 1]
    input_img = cv2.resize(img, (img_size, img_size))
    input_img_norm = input_img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_img_norm, axis=0)

    # Predict
    print("Running inference...")
    preds = model.predict(input_tensor)[0]
    landmarks = preds.reshape(-1, 2)

    # Draw on the output image (using the original dimensions)
    vis_img = img.copy()
    for i, pt in enumerate(landmarks):
        # Coordinates are normalized [0, 1] relative to the input box
        px = int(pt[0] * w)
        py = int(pt[1] * h)
        
        # Draw point (Green for landmarks, Red for lobe point 48)
        color = (0, 0, 255) if i == 48 else (0, 255, 0)
        cv2.circle(vis_img, (px, py), max(2, w // 64), color, -1)
        
    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"Success! Prediction saved to: {output_path}")
    
    return vis_img

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_ear.py <path_to_image> [path_to_model]")
    else:
        m_path = sys.argv[2] if len(sys.argv) > 2 else None
        predict_ear(sys.argv[1], model_path=m_path)
