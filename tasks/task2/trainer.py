import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from .config import cfg

class AdvancedTrainer:
    """Trainer optimisé avec techniques avancées"""
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(cfg.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimiseur avancé
        self.optimizer = AdamW(
            model.parameters(), 
            lr=cfg.LEARNING_RATE, 
            weight_decay=1e-4
        )
        
        # Scheduler cosine
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS
        )
        
        # Loss function
        self.criterion = nn.L1Loss()  # MAE loss
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.mae_scores = []
        self.best_mae = float('inf')
    
    def train_epoch(self):
        """Entraînement d'une epoch avec gradient clipping"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Entraînement")
        
        for images, targets in progress_bar:
            images, targets = images.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Gradient clipping pour stabilité
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        self.scheduler.step()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validation avec métriques détaillées"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(cfg.DEVICE), targets.to(cfg.DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calcul MAE
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
        return total_loss / len(self.val_loader), mae
    
    def train(self, epochs=cfg.NUM_EPOCHS):
        """Boucle d'entraînement complète"""
        print(f"🚀 Début entraînement optimisé - {epochs} epochs - Device: {cfg.DEVICE}")
        
        for epoch in range(epochs):
            # Entraînement
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, mae = self.validate()
            self.val_losses.append(val_loss)
            self.mae_scores.append(mae)
            
            # Sauvegarde meilleur modèle
            if mae < self.best_mae:
                self.best_mae = mae
                self._save_checkpoint(epoch, mae)
                best_msg = "⭐ NOUVEAU MEILLEUR"
            else:
                best_msg = ""
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'  MAE: {mae:.2f} années {best_msg}')
            
            # Early stopping condition
            if epoch > 20 and mae > np.mean(self.mae_scores[-5:]):
                print("🛑 Convergence atteinte - early stopping")
                break
        
        self._plot_results()
        return self.best_mae
    
    def _save_checkpoint(self, epoch, mae):
        """Sauvegarde du modèle"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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