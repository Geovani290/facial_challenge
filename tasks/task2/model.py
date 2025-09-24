import torch
import torch.nn as nn
import torchvision.models as models
from .config import cfg

class AdvancedAgeModel(nn.Module):
    """Modèle optimisé pour l'estimation d'âge - MAE < 4.0"""
    
    def __init__(self, backbone=cfg.BACKBONE, pretrained=cfg.PRETRAINED):
        super().__init__()
        
        # Backbone pré-entraîné
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Supprimer classification
        else:
            raise ValueError(f"Backbone {backbone} non supporté")
        
        # Tête de regression OPTIMISÉE
        self.regressor = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # Sortie: âge
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation avancée des poids"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        age = self.regressor(features)
        return age.squeeze()

def create_advanced_model():
    """Factory function pour le modèle optimisé"""
    model = AdvancedAgeModel()
    print(f"✅ Modèle optimisé créé: {cfg.BACKBONE}")
    print(f"📊 Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 Objectif: MAE < 4.0 années sur {cfg.NUM_EPOCHS} epochs")
    return model