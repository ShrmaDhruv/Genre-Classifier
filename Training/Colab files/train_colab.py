import torch
print("=" * 60)
print("SYSTEM CHECK")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: GPU not available! Enable it in Runtime → Change runtime type")
print("=" * 60)
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/gtzan_spectrograms"
print(f"\nData directory set to: {DATA_DIR}")

import os

print("\n" + "=" * 60)
print("DATA VERIFICATION")
print("=" * 60)

if os.path.exists(DATA_DIR):
    genres = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"Found {len(genres)} genres: {genres}")

    total_images = 0
    for genre in genres:
        genre_path = os.path.join(DATA_DIR, genre)
        num_images = len([f for f in os.listdir(genre_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {genre}: {num_images} images")
        total_images += num_images

    print(f"\n Total images: {total_images}")
    print("=" * 60)
else:
    print(f"ERROR: Data directory not found: {DATA_DIR}")
    print("\nPlease check:")
    print("1. Did you mount Google Drive?")
    print("2. Is the path correct?")
    print("3. Did you upload the spectrograms folder?")
    print("=" * 60)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import json

torch.manual_seed(42)
np.random.seed(42)

print("  All libraries imported successfully!")
class Config:
    DATA_DIR = DATA_DIR

    MODEL_TYPE = "resnet18"
    NUM_CLASSES = 10

    BATCH_SIZE = 64         
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    IMG_SIZE = 224           

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PATIENCE = 10

    SAVE_TO_DRIVE = True
    DRIVE_SAVE_PATH = "/content/drive/MyDrive/music_genre_models"

config = Config()

if config.SAVE_TO_DRIVE:
    os.makedirs(config.DRIVE_SAVE_PATH, exist_ok=True)
    MODEL_SAVE_DIR = config.DRIVE_SAVE_PATH
    RESULTS_DIR = config.DRIVE_SAVE_PATH
    print(f"  Results will be saved to Google Drive: {config.DRIVE_SAVE_PATH}")
else:
    MODEL_SAVE_DIR = "/content/models"
    RESULTS_DIR = "/content/results"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"  Results will be saved locally (will be lost when session ends)")

print("\n" + "=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Device: {config.DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Model: {config.MODEL_TYPE}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Image size: {config.IMG_SIZE}x{config.IMG_SIZE}")
print(f"Learning rate: {config.LEARNING_RATE}")
print(f"Max epochs: {config.NUM_EPOCHS}")
print("=" * 60)

def get_data_transforms():
    """
    Define image transformations for training and validation/test.
    Training uses data augmentation to improve generalization.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_test_transforms


def load_dataset():
    """
    Load the spectrogram dataset and split into train/val/test sets.
    """
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

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_transforms, val_test_transforms = get_data_transforms()
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_test_transforms
    test_dataset.dataset.transform = val_test_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"\n  Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_names

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
    """
    Get model based on configuration.

    Args:
        model_type: "custom", "resnet18", or "resnet34"
        num_classes: number of output classes (10 for GTZAN)
        pretrained: use ImageNet pretrained weights
    """
    print(f"\nBuilding model: {model_type}")

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

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch with progress bar.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc='Training', leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model with progress bar.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc='Validation', leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, save_dir):
    """
    Complete training loop with early stopping and model checkpointing.
    """
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

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f} | Epoch Time: {epoch_time:.2f}s')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            model_path = os.path.join(save_dir, f'best_model_{config.MODEL_TYPE}.pth')
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
            print(f'Patience: {patience_counter}/{config.PATIENCE}')

        if patience_counter >= config.PATIENCE:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            print(f'Best validation accuracy: {best_val_acc:.2f}%')
            break

        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f'  Training completed in {total_time/60:.2f} minutes')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print("=" * 60)

    return history, best_val_acc


def plot_training_history(history, save_path):
    """
    Plot and save training history (loss and accuracy curves).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'  Training history plot saved to {save_path}')


def evaluate_and_plot_confusion_matrix(model, test_loader, class_names, device, save_dir):
    """
    Evaluate model on test set and create confusion matrix.
    """
    print("\nEvaluating on test set...")

    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    pbar = tqdm(test_loader, desc='Testing')

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})

    test_acc = 100. * correct / total

    print(f'\nTest Accuracy: {test_acc:.2f}%')
    print(f'Test Samples: {total}')

    print("\nClassification Report:")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    print(report)

    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Total Samples: {total}\n\n")
        f.write(report)
    print(f"  Classification report saved to {report_path}")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.xlabel('Predicted Genre', fontsize=12)
    plt.ylabel('True Genre', fontsize=12)
    plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.2f}%',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'  Confusion matrix saved to {cm_path}')

    print("\nPer-Class Accuracy:")
    print("=" * 60)
    for i, genre in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        class_acc = 100. * class_correct / class_total if class_total > 0 else 0
        print(f"{genre:12s}: {class_acc:6.2f}% ({class_correct}/{class_total})")

    return test_acc, cm

print("\n" + "=" * 60)
print("STARTING TRAINING PIPELINE")
print("=" * 60)

train_loader, val_loader, test_loader, class_names = load_dataset()

model = get_model(config.MODEL_TYPE, config.NUM_CLASSES, pretrained=True)
model = model.to(config.DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print("\nTraining setup complete")
print(f"Optimizer: Adam (lr={config.LEARNING_RATE})")
print(f"Loss function: CrossEntropyLoss")
print(f"Scheduler: ReduceLROnPlateau")

history, best_val_acc = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    config.DEVICE, config.NUM_EPOCHS, MODEL_SAVE_DIR
)

print("\nGenerating training history plots...")
plot_training_history(history, os.path.join(RESULTS_DIR, 'training_history.png'))

best_model_path = os.path.join(MODEL_SAVE_DIR, f'best_model_{config.MODEL_TYPE}.pth')
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'\n  Loaded best model from epoch {checkpoint["epoch"]+1}')
print(f'  Validation accuracy: {checkpoint["val_acc"]:.2f}%')

test_acc, cm = evaluate_and_plot_confusion_matrix(model, test_loader, class_names, config.DEVICE, RESULTS_DIR)

results = {
    'model_type': config.MODEL_TYPE,
    'best_val_acc': float(best_val_acc),
    'test_acc': float(test_acc),
    'num_epochs_trained': len(history['train_loss']),
    'best_epoch': int(checkpoint['epoch']) + 1,
    'class_names': class_names,
    'batch_size': config.BATCH_SIZE,
    'image_size': config.IMG_SIZE,
    'learning_rate': config.LEARNING_RATE,
    'train_samples': len(train_loader.dataset),
    'val_samples': len(val_loader.dataset),
    'test_samples': len(test_loader.dataset),
}

results_path = os.path.join(RESULTS_DIR, 'results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f'\nResults saved to {results_path}')

print("\n" + "=" * 60)
print("TRAINING PIPELINE COMPLETED!")
print("=" * 60)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"\nFiles saved to: {RESULTS_DIR}")
print(f"best_model_{config.MODEL_TYPE}.pth")
print(f"training_history.png")
print(f"confusion_matrix.png")
print(f"classification_report.txt")
print(f"results.json")
print("=" * 60)

if torch.cuda.is_available():
    print(f"\n GPU Memory Summary:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")



from google.colab import files

print("\nDownloading results to your computer...")

model_file = os.path.join(MODEL_SAVE_DIR, f'best_model_{config.MODEL_TYPE}.pth')
if os.path.exists(model_file):
    files.download(model_file)
    print(f"Downloaded: {model_file}")

history_file = os.path.join(RESULTS_DIR, 'training_history.png')
if os.path.exists(history_file):
    files.download(history_file)
    print(f"Downloaded: {history_file}")

cm_file = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
if os.path.exists(cm_file):
    files.download(cm_file)
    print(f"Downloaded: {cm_file}")

results_file = os.path.join(RESULTS_DIR, 'results.json')
if os.path.exists(results_file):
    files.download(results_file)
    print(f"Downloaded: {results_file}")

report_file = os.path.join(RESULTS_DIR, 'classification_report.txt')
if os.path.exists(report_file):
    files.download(report_file)
    print(f"Downloaded: {report_file}")

print("\nAll files downloaded successfully!")


print("\nAll done Check your Google Drive for saved models and results.")