"""
Echo Module — Final Model Architecture
Model   : MobileNetV2 (Transfer Learning)
Dataset : EchoNet-Dynamic
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model


def build_echo_model(weights_path=None):
    """
    Builds MobileNetV2-based echo classification model.

    Parameters
    ----------
    weights_path : str, optional
        Path to saved .keras weights file.
        If None, builds fresh model with ImageNet weights.

    Returns
    -------
    model : tf.keras.Model
        Compiled model ready for training or inference.
    """
    base_model = MobileNetV2(
        weights     = 'imagenet',
        include_top = False,
        input_shape = (224, 224, 3)
    )
    base_model.trainable = False

    inputs  = tf.keras.Input(shape=(224, 224, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(512, activation='relu')(x)
    x       = layers.Dropout(0.5)(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='softmax')(x)

    model = Model(inputs, outputs)

    if weights_path:
        model.load_weights(weights_path)
        print(f"Weights loaded from: {weights_path}")
    else:
        print("Fresh model built with ImageNet weights")

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )

    return model


# EF-based label mapping
# ─────────────────────────────────────────────
# EF > 55%   →  0  Normal
# EF 40-55%  →  1  Mild
# EF 30-40%  →  2  Moderate
# EF < 30%   →  3  Severe
LABEL_MAP = {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}

# Risk level mapping for fusion module
# ─────────────────────────────────────────────
# Normal + Mild   → Low
# Moderate        → Medium
# Severe          → High
RISK_MAP = {0: 'Low', 1: 'Low', 2: 'Medium', 3: 'High'}
