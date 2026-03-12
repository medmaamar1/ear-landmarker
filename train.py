import os
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from dataset import EarDataset, TARGET_INDICES, NUM_LANDMARKS
import numpy as np
import cv2
import argparse

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

@tf.keras.utils.register_keras_serializable()
def wing_loss(y_true, y_pred, w=0.05, epsilon=0.01):
    """
    Wing Loss: designed for robust landmark regression.
    It is more sensitive to small errors than MSE/L2.
    """
    delta = tf.abs(y_true - y_pred)
    C = w - w * tf.math.log(1.0 + w / epsilon)
    
    loss = tf.where(
        delta < w,
        w * tf.math.log(1.0 + delta / epsilon),
        delta - C
    )
    return tf.reduce_mean(loss, axis=-1)

def build_heatmap_model(input_shape=(224, 224, 3), num_landmarks=NUM_LANDMARKS):
    """
    Builds a Tiny U-Net heatmap regressor using MobileNetV3 as the encoder.
    Outputs a tensor of shape (56, 56, NUM_LANDMARKS).
    """
    inputs = Input(shape=input_shape)
    
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Pass inputs through the encoder (ignoring Batch Norm updates)
    x = base_model(inputs, training=False)
    # MobileNetV3 output shape for 224x224 input is roughly (7, 7, 576)
    
    # 2. DECODER (UpSampling)
    # 7x7 -> 14x14
    x = layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # 14x14 -> 28x28
    x = layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # 28x28 -> 56x56
    x = layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # 3. OUTPUT HEAD (55 Heatmaps)
    outputs = layers.Conv2D(num_landmarks, kernel_size=(1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    return model

def train(data_dir=None, epochs=100, batch_size=32):
    # Configuration
    IMG_SIZE = 224
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    
    # Priority: 1. Argument, 2. Env Var, 3. Default
    DATA_DIR = data_dir or os.environ.get('DATA_DIR') or './data_audioear' 

    print(f"Starting training process...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")

    # Load dataset
    dataset = EarDataset(DATA_DIR, img_size=IMG_SIZE, heatmap_size=56)
    train_gen, val_gen = dataset.get_generators(batch_size=BATCH_SIZE)

    if len(dataset.filenames) == 0:
        print("ERROR: No valid data found. Check your DATA_DIR and file extensions.")
        return

    # Build and train
    model = build_heatmap_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    checkpoint_path = "checkpoints/ear_landmarker_{epoch:02d}.keras"
    os.makedirs("checkpoints", exist_ok=True)
    
    # Callbacks for robust training
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Stage 1: Warm-up (Frozen Base) — 50 epochs
    print("\nStage 1: Warming up the decoder head (Base Frozen, 50 epochs)...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=min(50, EPOCHS),
        callbacks=callbacks
    )

    # Stage 2: Fine-tuning (Unfreeze Base)
    if EPOCHS > 50:
        print("\nStage 2: Unfreezing base model for fine-tuning...")
        for layer in model.layers:
            if 'mobilenet' in layer.name:
                 layer.trainable = True
            
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='mse',
            metrics=['mae']
        )

        print("Starting full fine-tuning...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=max(0, EPOCHS - 30),
            callbacks=callbacks
        )

    # Save final model
    model.save('ear_landmarker_final.keras')
    print("\nTraining complete. Model saved as 'ear_landmarker_final.keras'.")

    # --- VISUAL VERIFICATION STEP ---
    print("\n--- Visual Verification ---")
    os.makedirs('results', exist_ok=True)
    
    val_images, val_heatmaps = next(iter(val_gen))
    preds = model.predict(val_images)
    
    def extract_coords(heatmaps):
        coords = []
        for i in range(NUM_LANDMARKS):
            hmap = heatmaps[:, :, i]
            idx = np.unravel_index(np.argmax(hmap), hmap.shape)
            x_norm = idx[1] / hmap.shape[1]
            y_norm = idx[0] / hmap.shape[0]
            coords.append([x_norm, y_norm])
        return np.array(coords)

    for i in range(min(5, len(val_images))):
        img = val_images[i].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        pred_coords = extract_coords(preds[i])
        true_coords = extract_coords(val_heatmaps[i])

        for j, pt in enumerate(pred_coords):
            px, py = int(pt[0] * IMG_SIZE), int(pt[1] * IMG_SIZE)
            color = (0, 0, 255) if j == 48 else (0, 255, 0)
            cv2.circle(img, (px, py), 2, color, -1)
            
        cv2.imwrite(f'results/v_test_{i}.jpg', img)
        print(f"Saved validation sample: results/v_test_{i}.jpg")
    
    print("\nNext step: tensorflowjs_converter --input_format keras ear_landmarker_final.keras ./tfjs_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Ear Landmarker Model')
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
