import torch
import torch.nn as nn
import torchvision.models as models
from .config import cfg

class OrdinalAgeModel(nn.Module):
    """Modèle optimisé pour l'estimation d'âge en utilisant la régression ordinale."""
    
    def __init__(self, backbone=cfg.BACKBONE, pretrained=cfg.PRETRAINED, num_classes=cfg.MAX_AGE_VALUE):
        super().__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Supprimer classification
        else:
            raise ValueError(f"Backbone {backbone} non supporté")
        
        # Tête de régression ordinale
        self.ordinal_regressor = nn.Sequential(
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
            nn.Linear(128, num_classes + 1) # Sortie: num_classes + 1 logits
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
        
        logits = self.ordinal_regressor(features)
        
        return logits.squeeze()

def create_ordinal_model():
    """Factory function pour le modèle de régression ordinale"""
    model = OrdinalAgeModel()
    print(f"✅ Modèle de régression ordinale créé: {cfg.BACKBONE}")
    print(f"📊 Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    return model
