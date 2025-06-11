import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / 'data' / 'ProcessedResizedNorm'
TRAIN_DIR  = DATA_DIR / 'train'
VAL_DIR    = DATA_DIR / 'val'
CATEGORIES = ['Cat', 'Dog']
SEED       = 42
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
PATIENCE   = 3
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():  
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class NPYDataset(Dataset):
    def __init__(self, root_dir: Path, augment: bool = False):
        self.paths = []
        self.labels = []
        self.augment = augment

        for idx, label in enumerate(CATEGORIES):
            for f in (root_dir / label).glob("*.npy"):
                self.paths.append(f)
                self.labels.append(idx)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]) if augment else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx]).astype(np.float32)
        img = torch.tensor(arr).permute(2, 0, 1)
        if self.augment and self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

train_ds = NPYDataset(TRAIN_DIR, augment=True)
val_ds   = NPYDataset(VAL_DIR, augment=False)

if len(train_ds) == 0 or len(val_ds) == 0:
    raise ValueError("No data found! Check your directory structure.")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

model = BasicCNN().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_accuracies = []
val_f1_scores = []
best_f1 = 0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred_labels)
            y_true.extend(yb.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    train_losses.append(avg_loss)
    val_accuracies.append(acc)
    val_f1_scores.append(f1)

    print(f"Epoch {epoch+1:02d}: Loss={avg_loss:.4f} | Accuracy={acc:.3f} | F1={f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (No F1 imporvement in {PATIENCE} epochs)")
            break

print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Average Training Loss: {np.mean(train_losses):.3f} ± {np.std(train_losses):.3f}")
print(f"Average Validation Accuracy: {np.mean(val_accuracies):.3f} ± {np.std(val_accuracies):.3f}")
print(f"Average Validation F1-Score: {np.mean(val_f1_scores):.3f} ± {np.std(val_f1_scores):.3f}")

MODEL_PATH = SCRIPT_DIR / "models" / "cnn_model.pth"
MODEL_PATH.parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)


"""GradCAM"""
for i in range(3):
    random_idx = random.randint(0, len(val_ds) - 1)
    val_img, val_label = val_ds[random_idx]
    input_tensor = val_img.unsqueeze(0).to(DEVICE)

    target_layer = model.features[-3]

    cam = GradCAM(model=model, target_layers=[target_layer])

    model.eval()
    pred = model(input_tensor)
    pred_class = pred.argmax(dim=1).item()

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]

    rgb_img = val_img.permute(1, 2, 0).cpu().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.title(f"Sample #{random_idx} - Predicted: {CATEGORIES[pred_class]}, Actual: {CATEGORIES[val_label]}")
    plt.axis('off')
    plt.show()
