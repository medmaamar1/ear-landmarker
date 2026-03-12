"""
train_2pt.py
============
Direct Coordinate Regression model — predicts ONLY 2 landmarks (indices 15 & 19).
Because the output space is now just 4 numbers (x1, y1, x2, y2) instead of 110,
the math is simple enough to break past the 0.03 MAE ceiling.

Architecture:
  MobileNetV3Small (frozen) → Conv2D(128) → Flatten → Dense(256) → Dense(4) + sigmoid

Saved model: ear_landmarker_2pt_final.keras
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from dataset_2pt import EarDataset2Pt, TARGET_INDICES

np.random.seed(42)
tf.random.set_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Wing Loss scaled for normalized [0, 1] coordinates
# ─────────────────────────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
def wing_loss(y_true, y_pred, w=0.05, epsilon=0.01):
    delta = tf.abs(y_true - y_pred)
    C = w - w * tf.math.log(1.0 + w / epsilon)
    loss = tf.where(
        delta < w,
        w * tf.math.log(1.0 + delta / epsilon),
        delta - C
    )
    return tf.reduce_mean(loss, axis=-1)


def build_2pt_model(input_shape=(224, 224, 3), num_points=2):
    """
    Lightweight direct-regression head.
    Only 4 output neurons: (x15, y15, x19, y19).
    """
    inputs = Input(shape=input_shape)

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model(inputs, training=False)

    # Reduce channels first to avoid parameter explosion
    x = layers.Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_points * 2, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name='ear_2pt_regressor')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=wing_loss,
        metrics=['mae']
    )
    return model


def train(data_dir=None, epochs=100, batch_size=16):
    IMG_SIZE   = 224
    DATA_DIR   = data_dir or os.environ.get('DATA_DIR') or '.'
    MODEL_NAME = 'ear_landmarker_2pt_final.keras'

    print(f"[2-Pt] Data dir : {DATA_DIR}")
    print(f"[2-Pt] Epochs   : {epochs}  |  Batch: {batch_size}")
    print(f"[2-Pt] Predicting landmark indices: {TARGET_INDICES}")

    dataset = EarDataset2Pt(DATA_DIR, img_size=IMG_SIZE, test_split=0.2)
    train_gen, val_gen = dataset.get_generators(batch_size=batch_size)

    model = build_2pt_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model.summary()

    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/ear_2pt_{epoch:02d}.keras",
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
            min_lr=1e-7,
            verbose=1
        )
    ]

    # ── Stage 1: Frozen base warm-up (50 epochs) ──────────────────────────────
    print("\nStage 1: Warm-up (Frozen Base, 50 epochs)...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=min(50, epochs),
        callbacks=callbacks
    )

    # ── Stage 2: Full fine-tuning (remaining epochs) ───────────────────────────
    if epochs > 50:
        print("\nStage 2: Unfreezing MobileNet for fine-tuning...")
        for layer in model.layers:
            if 'mobilenet' in layer.name.lower():
                layer.trainable = True
                # Keep internal BN in inference mode
                for sub in layer.layers:
                    if isinstance(sub, tf.keras.layers.BatchNormalization):
                        sub.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
            loss=wing_loss,
            metrics=['mae']
        )
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs - 50,
            callbacks=callbacks
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(MODEL_NAME)
    print(f"\nTraining complete. Saved: {MODEL_NAME}")

    # ── Visual Verification ────────────────────────────────────────────────────
    print("\n--- Visual Verification ---")
    os.makedirs('results', exist_ok=True)
    val_gen.augment = False

    for i in range(min(5, len(val_gen))):
        X_batch, Y_batch = val_gen[i]
        if len(X_batch) == 0:
            continue
        pred = model.predict(X_batch[:1], verbose=0)[0]   # (4,)
        img  = X_batch[0].astype(np.uint8)
        img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # Draw predicted points (Green)
        for j in range(len(TARGET_INDICES)):
            px = int(pred[j * 2]     * w)
            py = int(pred[j * 2 + 1] * h)
            cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(img, str(TARGET_INDICES[j]), (px + 6, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw ground truth (Blue)
        truth = Y_batch[0]
        for j in range(len(TARGET_INDICES)):
            px = int(truth[j * 2]     * w)
            py = int(truth[j * 2 + 1] * h)
            cv2.circle(img, (px, py), 5, (255, 80, 0), -1)

        out = f"results/2pt_val_{i}.jpg"
        cv2.imwrite(out, img)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2-Point Ear Landmark Model")
    parser.add_argument('--data_dir',   type=str, required=True)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
