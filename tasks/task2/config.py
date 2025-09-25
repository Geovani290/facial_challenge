import torch
import os
from pathlib import Path

class UltimateConfig:
    """Configuration optimisée pour performance maximale"""
    
    # Chemins 
    PROJECT_ROOT = Path("/content/facial_challenge")
    DATA_DIR = PROJECT_ROOT / "data" / "task2"
    TRAIN_DIR = DATA_DIR / "train" 
    TEST_DIR = DATA_DIR / "test"
    
    # Hyperparamètres ULTIMES
    BACKBONE = "resnet50"
    PRETRAINED = True
    INPUT_SIZE = (224, 224)
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    MAX_AGE_VALUE = 80  # 🔥 Adapté à nos données (16-77)
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE = 1e-3  # 🔥 Plus agressif
    DROPOUT_RATE = 0.3
    
    # Configuration entraînement
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    PIN_MEMORY = True if torch.cuda.is_available() else False
    SEED = 42
    
    # Sauvegarde
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task2_ultimate"
    MODEL_SAVE_PATH = OUTPUT_DIR / "best_model.pth"
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = UltimateConfig()
