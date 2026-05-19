import torch
from models.unet3d import UNet3D

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet3D().to(device)
model.load_state_dict(torch.load("unet3d.pth"))
model.eval()

def predict(volume):
    with torch.no_grad():
        volume = volume.to(device)
        pred = model(volume)
    return pred