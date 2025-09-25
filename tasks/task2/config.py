import torch
import os
from pathlib import Path

class AdvancedConfig:
    """Configuration optimisée pour MAE < 4.0"""
    
    # Chemins (seront adaptés pour Colab)
    PROJECT_ROOT = Path("/content/facial_challenge")
    DATA_DIR = PROJECT_ROOT / "data" / "task2"
    TRAIN_DIR = DATA_DIR / "train"
    TEST_DIR = DATA_DIR / "test"
    
    # Hyperparamètres OPTIMISÉS
    BACKBONE = "resnet50"  # ResNet101 trop lourd pour 50 epochs
    PRETRAINED = True
    INPUT_SIZE = (224, 224)
    BATCH_SIZE = 64  # Plus grand sur GPU
    NUM_EPOCHS = 50  # ⬅️ 50 EPOCHS COMME DEMANDÉ
    MAX_AGE_VALUE = 116 # Âge maximal du dataset UTKFace [2, 3]
    EARLY_STOPPING_PATIENCE = 15
    LEARNING_RATE = 1e-4
    DROPOUT_RATE = 0.3
    
    # Configuration entraînement
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # Seed pour reproductibilité
    SEED = 42
    
    # Sauvegarde
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task2_advanced"
    MODEL_SAVE_PATH = OUTPUT_DIR / "best_model.pth"
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Instance globale
cfg = AdvancedConfig()
