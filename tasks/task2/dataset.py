import os
import cv2
import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from .config import cfg

class AgeDataset(Dataset):
    """Dataset avec parsing robuste et data augmentation"""
    
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.image_files = []
        self.ages = []
        self.genders = []
        
        self._load_data()
    
    def _parse_filename(self, filename):
        """Parser robuste pour tous les formats"""
        patterns = [
            r'^(\d+)_(\d+)([A-Z])(\d{2})\.',
            r'^(\d+)_(\d+)([A-Z])_(\d{2})\.',
            r'(\d{2})\.(jpg|jpeg|png|JPG|JPEG|PNG)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                if len(groups) >= 4:  # Pattern complet
                    try:
                        age = int(groups[3])
                        if 1 <= age <= 100:
                            return groups[0], groups[1], groups[2], age
                    except:
                        continue
                elif len(groups) >= 1:  # Pattern âge seul
                    try:
                        age = int(groups[0])
                        if 1 <= age <= 100:
                            return "unknown", "0", "U", age
                    except:
                        continue
        return None, None, None, None
    
    def _load_data(self):
        """Chargement des données avec validation"""
        if not os.path.exists(self.data_dir):
            print(f"❌ Dossier introuvable: {self.data_dir}")
            return
        
        image_files = [f for f in os.listdir(self.data_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        valid_files = 0
        for img_file in image_files:
            _, _, _, age = self._parse_filename(img_file)
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
        transforms.ToPILImage(),
        transforms.Resize((240, 240)),  # Redimensionner plus grand
        transforms.RandomCrop((224, 224)),  # Crop aléatoire
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, test_transforms

def create_data_loaders():
    """Création des DataLoaders optimisés"""
    train_transforms, test_transforms = get_advanced_transforms()
    
    train_dataset = AgeDataset(cfg.TRAIN_DIR, transform=train_transforms, is_train=True)
    test_dataset = AgeDataset(cfg.TEST_DIR, transform=test_transforms, is_train=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    return train_loader, test_loader