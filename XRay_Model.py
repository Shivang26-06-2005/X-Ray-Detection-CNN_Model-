import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt

# ----------------------------
# 1. Dataset
# ----------------------------
class CovidDataset(Dataset):
    def __init__(self, root_dir, classes=None, limit_per_class=None):
        self.root_dir = root_dir
        self.classes = classes or ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
        self.files = []

        for label, cls in enumerate(self.classes):
            img_folder = os.path.join(root_dir, cls, "images")
            mask_folder = os.path.join(root_dir, cls, "masks")
            all_files = sorted(os.listdir(img_folder))
            if limit_per_class:
                all_files = all_files[:limit_per_class]
            for fname in all_files:
                img_path = os.path.join(img_folder, fname)
                mask_path = os.path.join(mask_folder, fname)
                if os.path.exists(mask_path):
                    self.files.append({"image": img_path, "mask": mask_path, "label": label})

    def __len__(self):
        return len(self.files)

    def random_augment(self, image, mask, strong=False):
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if strong and random.random() < 0.3:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        if strong and random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w = image.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if strong and random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            image = np.clip(alpha*image + beta, 0, 255).astype(np.uint8)
        return image, mask

    def __getitem__(self, idx):
        entry = self.files[idx]
        image = cv2.imread(entry["image"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(entry["mask"], cv2.IMREAD_GRAYSCALE)

        if entry["label"] in [0,3]:
            image, mask = self.random_augment(image, mask, strong=True)
        else:
            image, mask = self.random_augment(image, mask, strong=False)

        # Resize both to same size
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # Normalize
        image = image / 255.0
        mask = mask / 255.0

        # Stack channels
        combined = np.stack([image, mask], axis=-1)
        combined = torch.tensor(combined, dtype=torch.float32).permute(2,0,1)
        return combined, entry["label"]

# ----------------------------
# 2. CNN Model
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.adapt_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.adapt_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

# ----------------------------
# 3. Training Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = CovidDataset(r"D:\XRAY DETECTION\COVID-19_Radiography_Dataset\train", limit_per_class=1500)
val_dataset = CovidDataset(r"D:\XRAY DETECTION\COVID-19_Radiography_Dataset\val")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = SimpleCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

# For plotting
train_losses, val_losses = [], []
train_accs, val_accs = [], []
val_f1s = []
best_val_acc = 0.0

# ----------------------------
# 4. Training Loop
# ----------------------------
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss, correct, total = 0.0,0,0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs,1)
        total += targets.size(0)
        correct += (predicted==targets).sum().item()
    train_loss = running_loss/len(train_loader)
    train_acc = correct/total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0,0,0
    all_preds, all_labels = [],[]
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs,1)
            val_total += targets.size(0)
            val_correct += (predicted==targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    val_loss /= len(val_loader)
    val_acc = val_correct/val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # F1-score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    val_f1s.append(f1)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),"best_model.pth")

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {f1:.4f}")

# ----------------------------
# 5. Confusion Matrix
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)
classes = val_dataset.classes
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# ----------------------------
# 6. Plot Training Curves
# ----------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.plot(val_f1s, label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Accuracy & F1 Curve")
plt.legend()
plt.show()

print("Training complete. Best model saved as 'best_model.pth'")
