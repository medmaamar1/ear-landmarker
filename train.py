import os
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from dataset import EarDataset
import numpy as np
import cv2
import argparse

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_model(input_shape=(128, 128, 3), num_landmarks=55):
    """
    Builds a lightweight MobileNetV3-Small based coordinate regressor.
    """
    inputs = Input(shape=input_shape)
    
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base model initially
    base_model.trainable = False
    
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(num_landmarks * 2, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    return model

def train(data_dir=None, epochs=100, batch_size=32):
    # Configuration
    IMG_SIZE = 128
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    
    # Priority: 1. Argument, 2. Env Var, 3. Default
    DATA_DIR = data_dir or os.environ.get('DATA_DIR') or './data_audioear' 

    print(f"Starting training process...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")

    # Load dataset
    dataset = EarDataset(DATA_DIR, img_size=IMG_SIZE)
    train_gen, val_gen = dataset.get_generators(batch_size=BATCH_SIZE)

    if len(dataset.filenames) == 0:
        print("ERROR: No valid data found. Check your DATA_DIR and file extensions.")
        return

    # Build and train
    model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
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

    # Stage 1: Warm-up (Frozen Base)
    print("\nStage 1: Warming up the head (Base Frozen)...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=min(30, EPOCHS),
        callbacks=callbacks
    )

    # Stage 2: Fine-tuning (Unfreeze Base)
    if EPOCHS > 30:
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
    
    val_images, val_lms = next(iter(val_gen))
    preds = model.predict(val_images)
    
    for i in range(min(5, len(val_images))):
        img = val_images[i].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        lms = preds[i].reshape(-1, 2)
        for j, pt in enumerate(lms):
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
