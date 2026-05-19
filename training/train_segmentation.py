import torch
from torch.utils.data import DataLoader
from data.dataset import BrainMRIDataset
from models.unet3d import UNet3D
from utils.loss import dice_loss
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = BrainMRIDataset("data/raw")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNet3D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    loop = tqdm(loader)

    for img, mask in loop:
        img, mask = img.to(device), mask.to(device)

        pred = model(img)
        loss = dice_loss(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "unet3d.pth")