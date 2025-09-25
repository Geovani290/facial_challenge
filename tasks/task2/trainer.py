import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .config import cfg

class OrdinalLoss(nn.Module):
    """Loss pour la régression ordinale avec gestion des âges continus"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mae_loss = nn.L1Loss()  # Backup loss

    def forward(self, logits, targets):
        # Conversion âges continus → cibles ordinales
        targets_ordinal = self._continuous_age_to_ordinal(targets)
        ordinal_loss = self.bce_loss(logits, targets_ordinal)
        
        # Loss MAE de backup pour stabilité
        probs = torch.sigmoid(logits)
        predictions = torch.sum(probs > 0.5, dim=1).float()
        mae_loss = self.mae_loss(predictions, targets)
        
        # Combinaison pondérée
        return 0.7 * ordinal_loss + 0.3 * mae_loss
    
    def _continuous_age_to_ordinal(self, ages):
        """Convertit les âges continus en cibles ordinales avec soft boundaries"""
        batch_size = ages.size(0)
        targets = torch.zeros(batch_size, self.num_classes + 1).to(ages.device)
        
        for i, age in enumerate(ages):
            age_float = age.item()
            age_int = int(age_float)
            fraction = age_float - age_int
            
            # Soft assignment pour les âges non entiers
            if age_int > 0:
                targets[i, :age_int] = 1.0
            if age_int < self.num_classes:
                targets[i, age_int] = 1.0 - fraction
                if age_int + 1 <= self.num_classes:
                    targets[i, age_int + 1] = fraction
        
        return targets

class UltimateTrainer:
    """Trainer ultime avec toutes les techniques modernes"""
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(cfg.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer agressif
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Scheduler OneCycleLR pour convergence rapide
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=cfg.LEARNING_RATE,
            epochs=cfg.NUM_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
        # Loss functions
        self.criterion_ordinal = OrdinalLoss(num_classes=cfg.MAX_AGE_VALUE)
        self.criterion_mae = nn.L1Loss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.mae_scores = []
        self.rmse_scores = []
        self.learning_rates = []
        
        self.best_mae = float('inf')
        self.best_rmse = float('inf')
        self.patience_counter = 0
        
        print(f"✅ Trainer initialisé sur {cfg.DEVICE}")
        print(f"📊 Données: {len(train_loader)} batches train, {len(val_loader)} batches val")

    def train_epoch(self, epoch):
        """Entraînement d'une epoch avec gradient accumulation"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion_ordinal(logits, targets)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            current_lr = self.scheduler.get_last_lr()[0]
            self.learning_rates.append(current_lr)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        return total_loss / len(self.train_loader)

    def validate(self):
        """Validation avec métriques complètes"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(cfg.DEVICE), targets.to(cfg.DEVICE)
                logits = self.model(images)
                loss = self.criterion_ordinal(logits, targets)
                total_loss += loss.item()

                # Conversion logits → âges prédits
                probs = torch.sigmoid(logits)
                predictions = torch.sum(probs > 0.5, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calcul métriques
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        
        return total_loss / len(self.val_loader), mae, rmse

    def train(self, epochs=cfg.NUM_EPOCHS):
        """Boucle d'entraînement complète avec two-phase learning"""
        print("🚀 DÉBUT ENTRAÎNEMENT ULTIME")
        print("=" * 60)
        
        # PHASE 1: Warmup - Tête seulement
        print("\n--- PHASE 1: Warmup (10 époques) ---")
        self._freeze_backbone()
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.LEARNING_RATE * 0.1,  # LR plus bas pour warmup
            weight_decay=1e-4
        )
        
        for epoch in range(10):
            train_loss = self.train_epoch(epoch)
            val_loss, mae, rmse = self.validate()
            
            print(f'Warmup {epoch+1}/10 | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {mae:.2f}')
        
        # PHASE 2: Fine-tuning complet
        print("\n--- PHASE 2: Fine-tuning complet ---")
        self._unfreeze_backbone()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=1e-4
        )
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, mae, rmse = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.mae_scores.append(mae)
            self.rmse_scores.append(rmse)
            
            # Sauvegarde meilleur modèle
            best_msg = ""
            if mae < self.best_mae:
                self.best_mae = mae
                self.best_rmse = rmse
                self.patience_counter = 0
                self._save_checkpoint(epoch, mae, rmse)
                best_msg = "⭐ NOUVEAU MEILLEUR"
            else:
                self.patience_counter += 1
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'  MAE: {mae:.2f} ans | RMSE: {rmse:.2f} ans {best_msg}')
            print(f'  LR: {self.learning_rates[-1]:.2e}')
            
            # Early stopping
            if self.patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"🛑 Early stopping à l'epoch {epoch+1}")
                break
        
        self._plot_training_history()
        return self.best_mae, self.best_rmse

    def _freeze_backbone(self):
        """Gèle le backbone pour warmup"""
        for name, param in self.model.backbone.named_parameters():
            param.requires_grad = False
        print("🔒 Backbone gelé pour warmup")

    def _unfreeze_backbone(self):
        """Dégèle le backbone pour fine-tuning"""
        for name, param in self.model.backbone.named_parameters():
            param.requires_grad = True
        print("🔓 Backbone dégelé pour fine-tuning")

    def _save_checkpoint(self, epoch, mae, rmse):
        """Sauvegarde du modèle"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'mae': mae,
            'rmse': rmse,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, cfg.MODEL_SAVE_PATH)
        print(f"💾 Modèle sauvegardé (MAE: {mae:.2f}, RMSE: {rmse:.2f})")

    def _plot_training_history(self):
        """Visualisation complète de l'entraînement"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Losses
        ax1.plot(self.train_losses, label='Train Loss', alpha=0.7)
        ax1.plot(self.val_losses, label='Val Loss', alpha=0.7)
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.plot(self.mae_scores, label='MAE', color='red', alpha=0.7)
        ax2.set_title('MAE (années)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # RMSE
        ax3.plot(self.rmse_scores, label='RMSE', color='green', alpha=0.7)
        ax3.set_title('RMSE (années)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning Rate
        ax4.plot(self.learning_rates, label='LR', color='purple', alpha=0.7)
        ax4.set_title('Learning Rate')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(cfg.OUTPUT_DIR / 'ultimate_training.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_ultimate_trainer(model, train_loader, val_loader):
    return UltimateTrainer(model, train_loader, val_loader)
