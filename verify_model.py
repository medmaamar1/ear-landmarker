import os
import cv2
import numpy as np
import tensorflow as tf

def verify(model_path='ear_landmarker_final.keras', data_dir='./data_audioear'):
    """
    Loads the trained model and visualizes predictions on a few images.
    """
    model = tf.keras.models.load_model(model_path)
    img_size = 128
    
    # Get some sample images
    img_dir = os.path.join(data_dir, 'images')
    files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))][:10]
    
    os.makedirs('results', exist_ok=True)
    
    for f in files:
        img = cv2.imread(os.path.join(img_dir, f))
        h, w, _ = img.shape
        
        # Simple center crop for verification (mimics ROI)
        # Note: in real use, we use FaceMesh ROI. This is just for a quick check.
        min_dim = min(h, w)
        startY, startX = (h - min_dim) // 2, (w - min_dim) // 2
        crop = img[startY:startY+min_dim, startX:startX+min_dim]
        
        # Preprocess
        input_img = cv2.resize(crop, (img_size, img_size)) / 255.0
        input_tensor = np.expand_dims(input_img, axis=0)
        
        # Predict
        preds = model.predict(input_tensor)[0].reshape(-1, 2)
        
        # Draw on crop
        for pt in preds:
            px = int(pt[0] * min_dim)
            py = int(pt[1] * min_dim)
            cv2.circle(crop, (px, py), 3, (0, 255, 0), -1)
            
        cv2.imwrite(f'results/pred_{f}', crop)
        print(f"Saved result: results/pred_{f}")

if __name__ == "__main__":
    verify()
