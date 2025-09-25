import os
import cv2
import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from.config import cfg

class AgeDataset(Dataset):
    """Dataset avec parsing robuste et data augmentation"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files =
        self.ages =
        
        self._load_data()
    
    def _parse_filename(self, filename):
        """Parser robuste pour le format standard UTKFace"""
        # Pattern pour le format standard UTKFace: [age]_[gender]_[race]_[date&time].jpg
        # Gère aussi le suffixe.chip.jpg
        # Ex: 1_0_0_20170112213500903.jpg
        # Ex: 1_0_0_20170112213500903.jpg.chip.jpg
        pattern = r'^(\d+)_(\d)_(\d)_(\d+)\.(?:jpg|jpeg|png)(?:\.chip\.jpg)?$'
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            try:
                age = int(match.group(1))
                if 0 <= age <= 116: # Plage d'âge pour UTKFace
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
    train_transforms = transforms.Compose(, std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose(, std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, test_transforms

def create_data_loaders():
    """Création des DataLoaders optimisés"""
    train_transforms, test_transforms = get_advanced_transforms()
    
    train_dataset = AgeDataset(cfg.TRAIN_DIR, transform=train_transforms)
    test_dataset = AgeDataset(cfg.TEST_DIR, transform=test_transforms)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    return train_loader, test_loader
