import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.dataset import BrainMRIDataset
from models.classifier import TumorClassifier


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# =========================
# CLASSIFICATION DATASET
# =========================

class ClassificationDataset(Dataset):
    def __init__(self, root_dir):
        self.base_dataset = BrainMRIDataset(
            root_dir=root_dir,
            target_size=(64, 64, 64)
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]

        # Temporary fake labels
        # 0 = Glioblastoma
        # 1 = Astrocytoma
        # 2 = Meningioma

        label = torch.tensor(idx % 3, dtype=torch.long)

        return image, label


# =========================
# DATASET
# =========================

dataset = ClassificationDataset(
    "data/BraTS2024/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

# =========================
# MODEL
# =========================

model = TumorClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

os.makedirs("checkpoints", exist_ok=True)

# =========================
# TRAINING LOOP
# =========================

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    loop = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{num_epochs}"
    )

    for img, label in loop:

        img = img.to(device)
        label = label.to(device)

        # Forward pass
        pred = model(img)

        loss = criterion(pred, label)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        loop.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = running_loss / len(loader)

    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    save_path = f"checkpoints/classifier_epoch_{epoch + 1}.pth"

    torch.save(
        model.state_dict(),
        save_path
    )

    print(f"Saved checkpoint: {save_path}")

# Save final model
torch.save(
    model.state_dict(),
    "classifier.pth"
)

print("Classifier training completed successfully.")