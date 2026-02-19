import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from pathlib import Path
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    # Paths
    DATA_DIR = "output_img"  
    MODEL_SAVE_DIR = "models"
    RESULTS_DIR = "results"
    
    MODEL_TYPE = "resnet18"  
    NUM_CLASSES = 10 
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    IMG_SIZE = 224  
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    PATIENCE = 10  

    SAVE_BEST_ONLY = True

config = Config()

# Create output directories
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("Music Genre Classification - CNN Training")
print("=" * 60)
print(f"Device: {config.DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model: {config.MODEL_TYPE}")
print(f"Dataset: {config.DATA_DIR}")
print("=" * 60)


# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

def get_data_transforms():
    """
    Define image transformations for training and validation/test.
    """
    # Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),  # Mild augmentation
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Validation/Test transforms (no augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms


def load_dataset():
    print("\nLoading dataset...")
    
    full_dataset = datasets.ImageFolder(root=config.DATA_DIR)
    
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Total images: {len(full_dataset)}")
    
    total_size = len(full_dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = int(config.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset split:")
    print(f"  Training: {train_size} images ({config.TRAIN_SPLIT*100:.0f}%)")
    print(f"  Validation: {val_size} images ({config.VAL_SPLIT*100:.0f}%)")
    print(f"  Test: {test_size} images ({config.TEST_SPLIT*100:.0f}%)")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_transforms, val_test_transforms = get_data_transforms()
    
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_test_transforms
    test_dataset.dataset.transform = val_test_transforms
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, class_names


# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

class CustomCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_type, num_classes, pretrained=True):

    print(f"\n Building model: {model_type}")
    
    if model_type == "custom":
        model = CustomCNN(num_classes=num_classes)
        print("Custom CNN architecture created")
        
    elif model_type == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        print(f"ResNet18 loaded (pretrained={pretrained})")
        
    elif model_type == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        print(f"ResNet34 loaded (pretrained={pretrained})")
        
    elif model_type == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        print(f"EfficientNet-B0 loaded (pretrained={pretrained})")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%', end='\r')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    print("\nStarting training...")
    print("=" * 60)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print('-' * 60)
        
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f'\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f} | Time: {epoch_time:.2f}s')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            if config.SAVE_BEST_ONLY:
                model_path = os.path.join(config.MODEL_SAVE_DIR, f'best_model_{config.MODEL_TYPE}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, model_path)
                print(f'  Best model saved! (Val Acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            print(f'Best validation accuracy: {best_val_acc:.2f}%')
            break
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f' Training completed in {total_time/60:.2f} minutes')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print("=" * 60)
    
    return history, best_val_acc


# ============================================================================
# 4. EVALUATION AND VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """
    Plot training and validation metrics.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n  Training history plot saved to {save_path}')
    plt.close()


def evaluate_model(model, test_loader, class_names, device):
    """
    Evaluate model on test set and generate metrics.
    """
    print("\n🧪 Evaluating on test set...")
    
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * correct / total
    
    print(f'\n✓ Test Accuracy: {test_acc:.2f}%')
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.2f}%')
    plt.tight_layout()
    
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f'\nConfusion matrix saved to {cm_path}')
    plt.close()
    
    return test_acc, cm


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    train_loader, val_loader, test_loader, class_names = load_dataset()
    
    # Create model
    model = get_model(config.MODEL_TYPE, config.NUM_CLASSES, pretrained=True)
    model = model.to(config.DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train model
    history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        config.DEVICE, config.NUM_EPOCHS
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    best_model_path = os.path.join(config.MODEL_SAVE_DIR, f'best_model_{config.MODEL_TYPE}.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'\n✓ Loaded best model from epoch {checkpoint["epoch"]+1}')
    
    test_acc, cm = evaluate_model(model, test_loader, class_names, config.DEVICE)
    
    results = {
        'model_type': config.MODEL_TYPE,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'num_epochs_trained': len(history['train_loss']),
        'class_names': class_names
    }
    
    import json
    results_path = os.path.join(config.RESULTS_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f'\n✓ Results saved to {results_path}')
    print("\n" + "=" * 60)
    print("🎉 Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()