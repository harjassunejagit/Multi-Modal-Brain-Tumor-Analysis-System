import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import BrainMRIDataset
from models.unet3d import UNet3D
from utils.loss import dice_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset
dataset = BrainMRIDataset(
    "data/BraTS2024/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    target_size=(64, 64, 64)
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

# Model
model = UNet3D().to(device)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# Number of epochs
num_epochs = 10

# =========================
# TRAINING LOOP
# =========================

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    loop = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{num_epochs}"
    )

    for img, mask in loop:

        img = img.to(device)
        mask = mask.to(device)

        # Forward pass
        pred = model(img)

        # Resize mask if needed
        if pred.shape != mask.shape:
            mask = F.interpolate(
                mask,
                size=pred.shape[2:],
                mode="nearest"
            )

        # Compute loss
        loss = dice_loss(pred, mask)

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

    # Save checkpoint
    checkpoint_path = f"checkpoints/unet3d_epoch_{epoch + 1}.pth"

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(f"Saved checkpoint: {checkpoint_path}")

# Save final model
torch.save(
    model.state_dict(),
    "unet3d.pth"
)

print("Training completed successfully.")
print("Final model saved as unet3d.pth")


