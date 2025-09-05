import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

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
                    self.files.append({"image": img_path, "mask": mask_path, "label": label, "class_name": cls, "filename": fname})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        entry = self.files[idx]
        image = cv2.imread(entry["image"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(entry["mask"], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        image = image / 255.0
        mask = mask / 255.0
        combined = np.stack([image, mask], axis=-1)
        combined = torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1)
        return combined, entry["label"], entry["filename"], entry["class_name"]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.adapt_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_dataset = CovidDataset(r"D:\XRAY DETECTION\COVID-19_Radiography_Dataset\val")
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = SimpleCNN(num_classes=4).to(device)
model.load_state_dict(torch.load(r"D:\XRAY DETECTION\X Ray Detection\XrayDetection.pth", map_location=device))
model.eval()

all_labels = []
all_preds = []
all_filenames = []
all_true_names = []
all_pred_names = []

class_names = val_dataset.classes

with torch.no_grad():
    for inputs, targets, filenames, true_names in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        all_filenames.extend(filenames)
        all_true_names.extend(true_names)
        all_pred_names.extend([class_names[p.item()] for p in predicted.cpu()])

# Print predictions
print("Sample Predictions (filename | true class | predicted class):")
for fname, true_cls, pred_cls in zip(all_filenames, all_true_names, all_pred_names):
    print(f"{fname} | {true_cls} | {pred_cls}")

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\nValidation Accuracy: {accuracy:.4f}")
print(f"Validation F1-score: {f1:.4f}")
