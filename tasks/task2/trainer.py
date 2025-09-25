import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from.config import cfg
import os
import sys

# Ajout pour la régression ordinale
class OrdinalLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets_ordinal = self.age_to_ordinal_target(targets)
        return self.criterion(logits, targets_ordinal)
    
    def age_to_ordinal_target(self, ages):
        """Convertit les âges cibles en cibles ordinales (one-hot cumulatif)"""
        targets = torch.zeros(ages.size(0), self.num_classes).to(cfg.DEVICE)
        for i, age in enumerate(ages):
            # Assure que l'âge est un entier et est dans la plage valide
            age_int = min(int(age.item()), self.num_classes)
            if age_int > 0:
                targets[i, :age_int] = 1
        return targets

class AdvancedTrainer:
    """Trainer optimisé avec techniques avancées"""
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(cfg.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        self.criterion_ordinal = OrdinalLoss(num_classes=cfg.MAX_AGE_VALUE)
        self.criterion_mae = nn.L1Loss() # Pour la métrique
        
        # Tracking
        self.train_losses =
        self.val_losses =
        self.mae_scores =
        self.best_mae = float('inf')
        self.patience_counter = 0

    def train_epoch(self, freeze_backbone):
        """Entraînement d'une epoch avec gradient clipping"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Entraînement")

        # Geler/dégeler le backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze_backbone

        # Initialiser l'optimiseur et le scheduler pour l'étape actuelle
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.LEARNING_RATE,
            weight_decay=1e-4
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.NUM_EPOCHS
        )

        for images, targets in progress_bar:
            images, targets = images.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion_ordinal(logits, targets)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validation avec métriques détaillées"""
        self.model.eval()
        total_loss = 0
        all_predictions =
        all_targets =
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(cfg.DEVICE), targets.to(cfg.DEVICE)
                logits = self.model(images)
                loss = self.criterion_ordinal(logits, targets)
                
                total_loss += loss.item()
                # Calcul de l'âge à partir des logits (moyenne des probabilités)
                probs = torch.sigmoid(logits)
                predictions = torch.sum(probs > 0.5, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calcul MAE
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
        return total_loss / len(self.val_loader), mae
    
    def train(self, epochs=cfg.NUM_EPOCHS):
        """Boucle d'entraînement complète"""
        print(f"🚀 Début entraînement optimisé - {epochs} epochs - Device: {cfg.DEVICE}")

        print("\n--- ÉTAPE 1: Entraînement de la tête de régression (Backbone figé) ---")
        for epoch in range(10):  # Ex: 10 époques pour la première étape
            train_loss = self.train_epoch(freeze_backbone=True)
            val_loss, mae = self.validate()
            print(f'Epoch {epoch+1}/10: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {mae:.2f}')
            
        print("\n--- ÉTAPE 2: Fine-tuning du modèle complet (Backbone dégelé) ---")
        # Réinitialise les compteurs pour le suivi
        self.train_losses =
        self.val_losses =
        self.mae_scores =
        self.best_mae = float('inf')
        self.patience_counter = 0

        for epoch in range(epochs):
            # Entraînement
            train_loss = self.train_epoch(freeze_backbone=False)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, mae = self.validate()
            self.val_losses.append(val_loss)
            self.mae_scores.append(mae)
            
            # Sauvegarde meilleur modèle et early stopping
            best_msg = ""
            if mae < self.best_mae:
                self.best_mae = mae
                self.patience_counter = 0 # Réinitialise le compteur
                self._save_checkpoint(epoch, mae)
                best_msg = "⭐ NOUVEAU MEILLEUR"
            else:
                self.patience_counter += 1
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'  MAE: {mae:.2f} années {best_msg}')
            
            # Condition d'arrêt précoce basée sur la patience
            if self.patience_counter > cfg.EARLY_STOPPING_PATIENCE:
                print(f"🛑 Arrêt précoce: MAE non améliorée pendant {cfg.EARLY_STOPPING_PATIENCE} époques.")
                break
        
        self._plot_results()
        return self.best_mae
    
    def _save_checkpoint(self, epoch, mae):
        """Sauvegarde du modèle"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'mae': mae,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, cfg.MODEL_SAVE_PATH)
    
    def _plot_results(self):
        """Visualisation des résultats"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.mae_scores, label='MAE', color='red')
        plt.legend()
        plt.title('MAE (années)')
        
        plt.tight_layout()
        plt.savefig(cfg.OUTPUT_DIR / 'training_results.png', dpi=300)
        plt.show()

def create_advanced_trainer(model, train_loader, val_loader):
    return AdvancedTrainer(model, train_loader, val_loader)
