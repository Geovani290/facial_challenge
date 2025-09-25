import os
import cv2
import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment
from config import cfg

class AgeDataset(Dataset):
    """Dataset avec parsing robuste et data augmentation aggressive"""
    
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.image_files = []
        self.ages = []
        
        self._load_data()
    
    def _parse_filename(self, filename):
        """Parser ULTRA robuste pour tous les formats"""
        # Patterns multiples pour couvrir toutes les variantes
        patterns = [
            r'^\d+_(\d+)([MF])(\d{2})\.',  # Standard: XXXXXX_YZWW
            r'^\d+_([A-Z]?\d+)([MF])(\d{2})\.',  # Avec lettre: XXXXXX_AYZWW
            r'(\d{2})\.(jpg|jpeg|png)$'  # Fallback: cherche 2 chiffres avant extension
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) >= 3:  # Pattern complet
                        age_str = match.group(3) if match.lastindex >= 3 else match.group(1)
                    else:  # Fallback pattern
                        age_str = match.group(1)
                    
                    age = int(age_str)
                    if 1 <= age <= cfg.MAX_AGE_VALUE:
                        return age
                except (ValueError, IndexError):
                    continue
        return None
    
    def _load_data(self):
        """Chargement des données avec validation robuste"""
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
        
        print(f"📁 {os.path.basename(self.data_dir)}: {valid_files}/{len(image_files)} fichiers valides")
        if valid_files > 0:
            print(f"📊 Âges: {min(self.ages)}-{max(self.ages)} ans (moyenne: {np.mean(self.ages):.1f})")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        
        # Chargement image avec gestion d'erreur
        image = cv2.imread(img_path)
        if image is None:
            # Image corrompue - générer une image grise
            image = np.ones((224, 224, 3), dtype=np.uint8) * 128
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Redimensionnement intelligent
            if image.shape[0] < 224 or image.shape[1] < 224:
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Application des transformations
        if self.transform:
            image = self.transform(image)
        
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        return image, age

def get_ultimate_transforms():
    """Data augmentation aggressive avec AutoAugment moderne"""
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Larger resize for better augmentation
        transforms.RandomCrop(cfg.INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        autoaugment.TrivialAugmentWide(),  # 🔥 AutoAugment moderne
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # 🔥 Random Erasing pour robustesse
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(cfg.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, test_transforms

def create_data_loaders():
    """Création des DataLoaders optimisés avec validation split"""
    train_transforms, test_transforms = get_ultimate_transforms()
    
    # Chargement des datasets
    full_train_dataset = AgeDataset(cfg.TRAIN_DIR, transform=train_transforms, is_train=True)
    test_dataset = AgeDataset(cfg.TEST_DIR, transform=test_transforms, is_train=False)
    
    print(f"📊 Dataset complet: {len(full_train_dataset)} images")
    
    # Split train/val (90%/10%)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Application des transforms de validation au val_dataset
    val_dataset.dataset.transform = test_transforms
    
    # DataLoaders optimisés
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True,
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True  # 🔥 Évite les batches incomplets
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False,
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=cfg.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False,
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=cfg.PIN_MEMORY
    )
    
    print(f"✅ DataLoaders créés:")
    print(f"   Train: {len(train_loader)} batches ({len(train_dataset)} images)")
    print(f"   Val: {len(val_loader)} batches ({len(val_dataset)} images)")
    print(f"   Test: {len(test_loader)} batches ({len(test_dataset)} images)")
    
    return train_loader, val_loader, test_loader
