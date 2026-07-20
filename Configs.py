import os
from dataclasses import dataclass
import torch
import kagglehub

PATH = kagglehub.dataset_download("msambare/fer2013")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE:  int = 128
IMG_CHANNELS: int = 3
IMG_HEIGHT:  int = 224
IMG_WIDTH:   int = 224
NUM_CLASSES: int = 7
NUM_WORKERS: int = 0
EPOCHS:         int   = 15
LEARNING_RATE:  float = 0.00001
LOG_DIR:        str = "FER_LOGS"
CHECKPOINT_DIR: str = "CKPT_inter"
MODEL_DIR: str = "MODEL_DIR"
WEIGHT_DECAY: float = 0.5
MOMENTUM: float = 0.9

@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = NUM_CLASSES
    IMG_HEIGHT:  int = IMG_HEIGHT
    IMG_WIDTH:   int = IMG_WIDTH
    BATCH_SIZE:  int = BATCH_SIZE
    NUM_WORKERS: int = NUM_WORKERS
    path = os.path.join(PATH, "train")
    DATA_ROOT:  str = path


@dataclass(frozen=True)
class TrainingConfig:
    EPOCHS:         int   = EPOCHS
    LEARNING_RATE:  float = LEARNING_RATE
    LOG_DIR:        str = LOG_DIR
    CHECKPOINT_DIR: str = CHECKPOINT_DIR
    MODEL_DIR: str = MODEL_DIR


emotion_mapping = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Sad": 4,
    "Surprise": 5,
    "Neutral": 6,
}


input_size = (
    BATCH_SIZE,
    IMG_CHANNELS,
    IMG_HEIGHT,
    IMG_WIDTH
)