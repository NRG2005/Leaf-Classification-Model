import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import time
import multiprocessing as mp
from collections import Counter

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print("DEVICE SETUP")
print("=" * 60)
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("âœ“ RTX 3060 detected - training will be FAST!")
else:
    print("âš  No GPU detected - training will be slow")
print("=" * 60 + "\n")


class LeafRecognitionModel:
    def __init__(self, num_classes, learning_rate=0.0001):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = device
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_top5_accs = []
        self.val_top5_accs = []
        
    def build_model(self, model_name='resnet50', freeze_backbone=True):
        """
        Build model following paper's best practices
        Paper achieved 97.8% with ResNet101, but ResNet50 also got 97.6%
        """
        print(f"Building model: {model_name}")
        
        if model_name == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V2')
            num_features = self.model.fc.in_features
            # Replace final layer
            self.model.fc = nn.Linear(num_features, self.num_classes)
            
        elif model_name == 'resnet101':
            self.model = models.resnet101(weights='IMAGENET1K_V2')
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
            
        elif model_name == 'resnet152':
            self.model = models.resnet152(weights='IMAGENET1K_V2')
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        
        self.model = self.model.to(self.device)
        
        # Freeze backbone (as per paper - only train classifier)
        if freeze_backbone:
            print("Freezing backbone layers (transfer learning mode)...")
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # Don't freeze final layer
                    param.requires_grad = False
            print("âœ“ Only final classifier layer is trainable")
        
        # Loss and optimizer (paper used Adam with lower LR for fine-tuning)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.0001  # L2 regularization
        )
        
        # Scheduler - reduce LR on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        print(f"âœ“ Model built with {self.num_classes} classes")
        print(f"âœ“ Learning rate: {self.learning_rate}")
        
        return self.model
    
    def get_data_loaders(self, train_dir, val_dir, batch_size=64, num_workers=4, 
                        use_weighted_sampling=False):
        """
        Create data loaders with paper's preprocessing
        Paper used: 256x256 resize, normalization with ImageNet stats
        """
        
   
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Add augmentation for improved results (not in paper but helps)
        train_transform_augmented = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets (use augmentation)
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform_augmented)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        
        # Weighted sampling for class imbalance (optional)
        sampler = None
        if use_weighted_sampling:
            class_counts = Counter([label for _, label in train_dataset.samples])
            class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for _, label in train_dataset.samples]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            print("âœ“ Using weighted sampling for class imbalance")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        self.class_names = train_dataset.classes
        
        print(f"\nâœ“ Training samples: {len(train_dataset)}")
        print(f"âœ“ Validation samples: {len(val_dataset)}")
        print(f"âœ“ Number of classes: {len(self.class_names)}")
        print(f"âœ“ Batch size: {batch_size}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch with top-5 accuracy tracking"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        correct_top5 = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'top5': f'{100.*correct_top5/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        epoch_top5_acc = 100. * correct_top5 / total
        
        return epoch_loss, epoch_acc, epoch_top5_acc
    
    def validate(self, val_loader):
        """Validate with top-5 accuracy"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, True, True)
                correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'top5': f'{100.*correct_top5/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        epoch_top5_acc = 100. * correct_top5 / total
        
        return epoch_loss, epoch_acc, epoch_top5_acc
    
    def train(self, train_loader, val_loader, epochs=25, save_best=True):
        """Complete training loop"""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 10
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc, train_top5 = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_top5 = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.train_top5_accs.append(train_top5)
            self.val_top5_accs.append(val_top5)
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Top-5: {train_top5:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Top-5: {val_top5:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_top5_acc': val_top5,
                    'class_names': self.class_names,
                    'num_classes': self.num_classes
                }, 'best_leaf_model.pth')
                print(f"  âœ“ Best model saved! (Val Acc: {val_acc:.2f}%, Top-5: {val_top5:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("âœ… TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    def plot_results(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Top-1 Accuracy
        axes[1].plot(epochs, self.train_accs, 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.val_accs, 'r-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        axes[1].set_title('Top-1 Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[2].plot(epochs, self.train_top5_accs, 'b-', label='Train', linewidth=2)
        axes[2].plot(epochs, self.val_top5_accs, 'r-', label='Validation', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Top-5 Accuracy (%)', fontsize=12)
        axes[2].set_title('Top-5 Accuracy', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        print("\nâœ“ Training plots saved as 'training_results.png'")
        plt.show()
    
    def predict_image(self, image_path, top_k=5):
        """
        Predict a single leaf image - MAIN INFERENCE FUNCTION
        """
        self.model.eval()
        
       
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        from PIL import Image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, None
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
      
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
      
        top_probs, top_indices = probabilities.topk(top_k)
        
        print(f"\nðŸƒ Predictions for {Path(image_path).name}:")
        print("=" * 60)
        results = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
            species = self.class_names[idx]
            confidence = prob.item() * 100
            print(f"  {i}. {species:40s} {confidence:6.2f}%")
            results.append((species, confidence))
        print("=" * 60)
        
        return results[0][0], results[0][1]
    
    def save_model(self, filepath='leaf_model_final.pth'):
        """Save complete model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'train_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accs': self.train_accs,
                'val_accs': self.val_accs,
                'train_top5_accs': self.train_top5_accs,
                'val_top5_accs': self.val_top5_accs
            }
        }, filepath)
        
      
        with open('class_names.json', 'w') as f:
            json.dump(self.class_names, f, indent=2)
        
        print(f"\nâœ“ Model saved to {filepath}")
        print("âœ“ Class names saved to class_names.json")
    
    def load_model(self, filepath, model_name='resnet50'):
        """Load saved model for inference"""
        print(f"Loading model from {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.num_classes = checkpoint['num_classes']
        self.class_names = checkpoint['class_names']
        
    
        self.build_model(model_name, freeze_backbone=False)
        
    
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
       
        if 'train_history' in checkpoint:
            history = checkpoint['train_history']
            self.train_losses = history.get('train_losses', [])
            self.val_losses = history.get('val_losses', [])
            self.train_accs = history.get('train_accs', [])
            self.val_accs = history.get('val_accs', [])
        
        print(f"âœ“ Model loaded successfully")
        print(f"âœ“ Classes: {self.num_classes}")
        
        return self.model


def train_model():
    """Main training function"""
    print("\n" + "ðŸŒ³" * 20)
    print("PYTORCH LEAF RECOGNITION TRAINING")
    print("Based on Stanford's 'Stop and Scan the Trees' paper")
    print("ðŸŒ³" * 20 + "\n")
    
    # Configuration (optimized based on paper)
    TRAIN_DIR = "leafsnap-dataset/train"
    VAL_DIR = "leafsnap-dataset/val"
    BATCH_SIZE = 64  # Paper used similar batch size
    EPOCHS = 50  # More epochs with early stopping
    LEARNING_RATE = 0.0001  # Lower LR for transfer learning
    MODEL_NAME = 'resnet50'  # Paper showed ResNet50 gets 97.6%
    NUM_WORKERS = min(mp.cpu_count(), 8)
    

    num_classes = len(list(Path(TRAIN_DIR).iterdir()))
    print(f"Detected {num_classes} classes\n")
    
   
    model = LeafRecognitionModel(num_classes, learning_rate=LEARNING_RATE)
    
 
    model.build_model(model_name=MODEL_NAME, freeze_backbone=True)
    

    train_loader, val_loader = model.get_data_loaders(
        TRAIN_DIR, VAL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_weighted_sampling=False
    )
    
  
    model.train(train_loader, val_loader, epochs=EPOCHS)
    
    
    model.plot_results()
    
    
    model.save_model('leaf_model_final.pth')
    
    print("\nâœ… Training complete! Model ready for inference.")
    print("\nTo classify a leaf image, run:")
    print("  python train_leafsnap.py --predict path/to/leaf.jpg")



def predict_leaf(image_path, model_path='best_leaf_model.pth', model_name='resnet50'):
    """Predict a single leaf image"""
    print("\n" + "ðŸ”" * 20)
    print("LEAF CLASSIFICATION")
    print("ðŸ”" * 20 + "\n")
    
   
    model = LeafRecognitionModel(num_classes=185)  # Will be updated from checkpoint
    model.load_model(model_path, model_name=model_name)
    
 
    species, confidence = model.predict_image(image_path, top_k=5)
    
    print(f"\nâœ… Prediction complete!")
    print(f"Most likely species: {species}")
    print(f"Confidence: {confidence:.2f}%")
    
    return species, confidence



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--predict':
            if len(sys.argv) < 3:
                print("Usage: python train_leafsnap.py --predict <image_path>")
                sys.exit(1)
            
            image_path = sys.argv[2]
            model_path = sys.argv[3] if len(sys.argv) > 3 else 'best_leaf_model.pth'
            
            predict_leaf(image_path, model_path)
        else:
            print("Unknown argument. Use --predict <image_path> for inference")
    else:
        train_model()