import os
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from dataset import EarDataset
import numpy as np

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
    x = layers.Dropout(0.5)(x) # Increased dropout
    outputs = layers.Dense(num_landmarks * 2, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    return model

def train(data_dir=None):
    # Configuration
    IMG_SIZE = 128
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # Priority: 1. Argument, 2. Env Var, 3. Default
    DATA_DIR = data_dir or os.environ.get('DATA_DIR') or './data_audioear' 

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
    print("Stage 1: Warming up the head (Base Frozen)...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10, # Fixed short warm-up
        callbacks=callbacks
    )

    # Stage 2: Fine-tuning (Unfreeze Base)
    print("Stage 2: Unfreezing base model for fine-tuning...")
    # Unfreeze the base model
    for layer in model.layers:
        if layer.name == 'mobilenetv3_small': # The name of the base model layer
             layer.trainable = True
        
    # Re-compile with a MUCH smaller learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Very slow
        loss='mse',
        metrics=['mae']
    )

    print("Starting full fine-tuning...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS - 10,
        callbacks=callbacks
    )

    # Save final model
    model.save('ear_landmarker_final.keras')
    print("Training complete. Model saved as 'ear_landmarker_final.keras'.")
    print("Next step: tensorflowjs_converter --input_format keras ear_landmarker_final.keras ./tfjs_model")

if __name__ == "__main__":
    train()
