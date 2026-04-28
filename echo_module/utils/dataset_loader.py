"""
Echo Module — Dataset Loader & Generator
Dataset : EchoNet-Dynamic
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
NUM_FRAMES = 16


def get_label(ef):
    """Convert EF value to 4-class label."""
    if ef > 55:   return 0   # Normal
    elif ef > 40: return 1   # Mild
    elif ef > 30: return 2   # Moderate
    else:         return 3   # Severe


def load_dataframes(csv_path, video_dir):
    """
    Load FileList.csv and prepare train/val/test DataFrames.

    Parameters
    ----------
    csv_path  : str — path to FileList.csv
    video_dir : str — path to folder containing .avi videos

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    class_weights             : dict
    """
    df = pd.read_csv(csv_path)
    df['label'] = df['EF'].apply(get_label)

    def get_video_path(filename):
        path = os.path.join(video_dir, filename + '.avi')
        return path if os.path.exists(path) else None

    df['video_path'] = df['FileName'].apply(get_video_path)
    df = df[df['video_path'].notna()].reset_index(drop=True)

    train_df = df[df['Split'] == 'TRAIN'].reset_index(drop=True)
    val_df   = df[df['Split'] == 'VAL'].reset_index(drop=True)
    test_df  = df[df['Split'] == 'TEST'].reset_index(drop=True)

    weights = compute_class_weight(
        class_weight = 'balanced',
        classes      = np.unique(train_df['label'].values),
        y            = train_df['label'].values
    )
    class_weights = dict(enumerate(weights))

    print(f"Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")
    return train_df, val_df, test_df, class_weights


def extract_frames(video_path, num_frames=NUM_FRAMES):
    """
    Evenly samples frames from echocardiography video.

    Parameters
    ----------
    video_path : str — path to .avi video file
    num_frames : int — number of frames to extract

    Returns
    -------
    list of np.array — RGB frames at 224x224
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < num_frames:
        cap.release()
        return []

    indices = set(np.linspace(0, total - 1, num_frames, dtype=int))
    frames, current = [], 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if current in indices:
            frame = cv2.resize(frame, IMG_SIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        current += 1

    cap.release()
    return frames


class EchoDataGenerator(tf.keras.utils.Sequence):
    """
    Keras data generator for EchoNet videos.
    Loads videos in batches — memory efficient for large datasets.

    Parameters
    ----------
    df         : pd.DataFrame with columns [video_path, label]
    batch_size : int
    num_frames : int — frames per video
    augment    : bool — apply augmentation (train only)
    """

    def __init__(self, df, batch_size=8, num_frames=NUM_FRAMES, augment=False):
        self.df         = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.augment    = augment

    def __len__(self):
        return len(self.df) // self.batch_size

    def __augment_frame(self, frame):
        """Apply random augmentations to a single frame."""
        if np.random.rand() > 0.5:
            frame = cv2.flip(frame, 1)
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            frame  = np.clip(frame * factor, 0, 255).astype(np.uint8)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w  = frame.shape[:2]
            M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
        return frame

    def __getitem__(self, idx):
        batch  = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []

        for _, row in batch.iterrows():
            frames = extract_frames(row['video_path'], self.num_frames)
            if len(frames) == 0:
                continue
            for frame in frames:
                if self.augment:
                    frame = self.__augment_frame(frame)
                frame = preprocess_input(frame.astype(np.float32))
                images.append(frame)
                labels.append(row['label'])

        return np.array(images), tf.keras.utils.to_categorical(labels, num_classes=4)

    def on_epoch_end(self):
        if self.augment:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
