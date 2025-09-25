import os
import cv2
import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from config import cfg

class AgeDataset(Dataset):
    """Dataset avec parsing robuste et data augmentation"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.ages = []
        
        self._load_data()
    
    def _parse_filename(self, filename):
        """
        Parser robuste pour la convention XXXXXX_YZWW et ses variantes
        - XXXXXX_YZWW
        - XXXXX_AYZWW
        - XXXX_AYZWW
        - XXXXX_YZWW
        - XXXX_YZWW
        """
        # Ex: 123456_0M25.jpg
        # ID_NUMPHOTO(A ou Y)SEXEAGE.jpg
        
        # Regex qui capture tous les formats
        pattern = r'^\d+_(?:[A-Z]?)(\d+)([MF])(\d{2})\.(?:jpg|jpeg|png)$'
        match = re.search(pattern, filename, re.IGNORECASE)
        
        if match:
            try:
                # L'âge est dans le groupe de capture 3
                age = int(match.group(3))
                if 0 <= age <= cfg.MAX_AGE_VALUE:
                    return age
            except (ValueError, IndexError):
                return None
        return None
    
    def _load_data(self):
        """Chargement des données avec validation"""
        if not os.path.exists(self.data_dir):
            print(f"❌ Dossier introuvable: {self.data_dir}")
            return
        
        image_files = [f for f in os.listdir(self.data_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        valid_files = 0
        for img_file in image_files:
            age = self._parse_filename(img_file)
            if age is not None:
                self.image_files.append(img_file)
                self.ages.append(age)
                valid_files += 1
        
        print(f"📁 {self.data_dir}: {valid_files}/{len(image_files)} fichiers valides")
        if valid_files > 0:
            print(f"📊 Âges: {min(self.ages)}-{max(self.ages)} ans (moyenne: {np.mean(self.ages):.1f})")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(self.ages[idx], dtype=torch.float32)

def get_advanced_transforms():
    """Data augmentation aggressive pour meilleure généralisation"""
    train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    
    return train_transforms, test_transforms

def create_data_loaders():
    """Création des DataLoaders optimisés avec un split de validation"""
    train_transforms, val_transforms = get_advanced_transforms()
    
    full_train_dataset = AgeDataset(cfg.TRAIN_DIR, transform=train_transforms)
    test_dataset = AgeDataset(cfg.TEST_DIR, transform=val_transforms)
    
    # Split en 90% pour l'entraînement et 10% pour la validation
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader
