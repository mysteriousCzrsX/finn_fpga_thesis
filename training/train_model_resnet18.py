import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ResNet expects 3-channel input
    transforms.Resize((224, 224)),  # ResNet18 default
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet normalization
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("data/train", transform=transform)
val_dataset = datasets.ImageFolder("data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


resnet = models.resnet18(pretrained=True)

# Freeze all layers (optional for feature extraction)
for param in resnet.parameters():
    param.requires_grad = False


# Replace the classifier
resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.15),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)


criterion = nn.BCELoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.00015)


num_epochs = 40

# Lists to track metrics over epochs
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    num_batches = len(train_loader)

    # Training phase
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Calculate training accuracy
        preds_train = (outputs > 0.5).squeeze()
        labels_train = labels.squeeze()
        correct_train += (preds_train == labels_train).sum().item()
        total_train += labels.size(0)
        
        # Print batch progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{num_batches}], "
              f"Loss: {loss.item():.4f}")

    # Calculate average training loss and accuracy for this epoch
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Validation phase
    resnet.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels_float = labels.float().unsqueeze(1)
            outputs = resnet(inputs)
            loss = criterion(outputs, labels_float)
            val_loss += loss.item()
            
            preds = (outputs > 0.5).squeeze().long()
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    # Print epoch summary
    print(f"\n{'='*60}")
    print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
    print(f"  Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    print(f"  Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    print(f"{'='*60}\n")

# Create visualization graphs
plt.figure(figsize=(15, 5))

# Plot 1: Loss over epochs
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(range(1, num_epochs + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(range(1, num_epochs + 1))

# Plot 2: Accuracy over epochs
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
plt.plot(range(1, num_epochs + 1), val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Training and Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(range(1, num_epochs + 1))

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print(f"\nTraining graphs saved to 'training_history.png'")
plt.show()

# Print final summary
print(f"\n{'='*60}")
print("FINAL TRAINING SUMMARY")
print(f"{'='*60}")
print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
print(f"Best Validation Accuracy: {max(val_accuracies):.2f}% (Epoch {val_accuracies.index(max(val_accuracies)) + 1})")
print(f"{'='*60}")

# Save the trained model
model_save_path = 'resnet18_jamming_detector.pth'
torch.save({
    'model_state_dict': resnet.state_dict(),
    'model': resnet,  # Save full model for easier loading
    'num_epochs': num_epochs,
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'final_train_accuracy': train_accuracies[-1],
    'final_val_accuracy': val_accuracies[-1],
    'best_val_accuracy': max(val_accuracies),
    'best_epoch': val_accuracies.index(max(val_accuracies)) + 1,
    'model_architecture': {
        'base': 'resnet18',
        'classifier': str(resnet.fc),
        'input_size': (224, 224),
        'num_classes': 1
    }
}, model_save_path)

print(f"\n{'='*60}")
print("MODEL SAVED")
print(f"{'='*60}")
print(f"Model saved to: {model_save_path}")
print(f"\nTo load the model later, use:")
print(f"  checkpoint = torch.load('{model_save_path}')")
print(f"  model = checkpoint['model']  # Load full model")
print(f"  # OR")
print(f"  model = models.resnet18(pretrained=True)")
print(f"  model.fc = nn.Sequential(...)  # Recreate architecture")
print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
print(f"{'='*60}")
