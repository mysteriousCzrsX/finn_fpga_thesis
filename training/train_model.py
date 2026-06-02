import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import brevitas.nn as qnn
from brevitas.export import export_qonnx



#Dataloaders
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = ImageFolder("data/train", transform=transform)
val_dataset = ImageFolder("data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Classes: {train_dataset.classes}")  # ['jamming', 'no_jamming']


#CNN Model
class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # grayscale input
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # binary classification jamming or no_jamming
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class QuantSpectrogramCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            qnn.QuantConv2d(
                1,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
                weight_bit_width=8
            ),

            nn.BatchNorm2d(16),
            qnn.QuantReLU(bit_width=8),
            nn.MaxPool2d(kernel_size=2, stride=2),

            qnn.QuantConv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
                weight_bit_width=8
            ),

            nn.BatchNorm2d(32),
            qnn.QuantReLU(bit_width=8),
            nn.MaxPool2d(kernel_size=2, stride=2),

            qnn.QuantConv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
                weight_bit_width=8
            ),

            nn.BatchNorm2d(64),
            qnn.QuantReLU(bit_width=8),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(
                64,
                128,
                bias=True,
                weight_bit_width=8
            ),
            qnn.QuantReLU(bit_width=8),
            nn.Dropout(0.3),
            qnn.QuantLinear(
                128,
                1,
                bias=True,
                weight_bit_width=8
            )
        )
        self.input_quant = qnn.QuantIdentity(
            bit_width=8,
            return_quant_tensor=True
        )
    def forward(self, x):
        x = self.input_quant(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = QuantSpectrogramCNN().to(device)
model = SpectrogramCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 3e-4)



# training loop, validation
num_epochs = 1
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        preds = (outputs > 0).squeeze().long()
        correct_train += (preds == labels.squeeze().long()).sum().item()
        total_train += labels.size(0)

    train_loss = running_train_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()
            preds = (outputs > 0.5).squeeze().long()
            correct_val += (preds == labels.squeeze().long()).sum().item()
            total_val += labels.size(0)

    val_loss = running_val_loss / len(val_loader)
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Final validation accuracy (from last epoch)
print(f"\nFinal Validation Accuracy: {val_accuracies[-1]*100:.2f}%")

# Optional: Best validation accuracy achieved during training
best_val_acc = max(val_accuracies)
best_epoch = val_accuracies.index(best_val_acc) + 1
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% at Epoch {best_epoch}")

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

example_inputs = torch.rand(1, 1, 256, 256)
# export_qonnx(
#     model,
#     torch.rand(1, 1, 256, 256),
#     export_path="quant_spectrogram_cnn.onnx"
# )
onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
onnx_program.save("spectrogram_cnn.onnx")