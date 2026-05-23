import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.unet3d import UNet3D


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# =========================
# Load trained model
# =========================
model = UNet3D().to(device)

model_path = "weights/unet3d.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found at {model_path}"
    )

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()

print("Segmentation model loaded successfully.")


# =========================
# Utility functions
# =========================
def load_nifti(path):
    img = nib.load(path).get_fdata().astype(np.float32)

    img = (img - img.mean()) / (img.std() + 1e-8)

    return img


def resize_volume(volume, target_size=(64, 64, 64)):
    tensor = torch.tensor(
        volume,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=target_size,
        mode="trilinear",
        align_corners=False
    )

    return tensor.squeeze().numpy()


def find_file(folder, suffix):
    for file in os.listdir(folder):
        if file.endswith(suffix):
            return os.path.join(folder, file)

    raise FileNotFoundError(
        f"Could not find file ending with '{suffix}'"
    )


# =========================
# Load MRI sample
# =========================
# =========================
# Dynamic patient selection
# =========================

base_folder = "test_sample"

if not os.path.exists(base_folder):
    raise FileNotFoundError(
        f"Folder not found: {base_folder}"
    )

patients = [
    p for p in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, p))
]

if len(patients) == 0:
    raise ValueError(
        "No patient folders found inside test_sample/"
    )

print("\nAvailable Patient Folders:\n")

for idx, patient in enumerate(patients):
    print(f"{idx + 1}. {patient}")

choice = int(input("\nEnter patient number: "))

sample_folder = os.path.join(
    base_folder,
    patients[choice - 1]
)

print(f"\nSelected patient: {patients[choice - 1]}")

print("Loading MRI modalities...")

t1_path = find_file(sample_folder, "-t1n.nii.gz")
t2_path = find_file(sample_folder, "-t2w.nii.gz")
flair_path = find_file(sample_folder, "-t2f.nii.gz")

t1 = resize_volume(load_nifti(t1_path))
t2 = resize_volume(load_nifti(t2_path))
flair = resize_volume(load_nifti(flair_path))

# Stack modalities
image = np.stack([t1, t2, flair], axis=0)

# Convert to tensor
image_tensor = torch.tensor(
    image,
    dtype=torch.float32
).unsqueeze(0).to(device)

print("Running segmentation inference...")


# =========================
# Prediction
# =========================
with torch.no_grad():
    prediction = model(image_tensor)

prediction = torch.sigmoid(prediction)

prediction = (
    prediction > 0.5
).float()

prediction = prediction.squeeze().cpu().numpy()

print("Inference completed.")


# =========================
# Save outputs
# =========================
os.makedirs("outputs", exist_ok=True)

middle_slice = prediction.shape[0] // 2

plt.figure(figsize=(10, 5))

# FLAIR slice
plt.subplot(1, 2, 1)
plt.imshow(flair[middle_slice], cmap="gray")
plt.title("FLAIR MRI Slice")
plt.axis("off")

# Predicted mask
plt.subplot(1, 2, 2)
plt.imshow(prediction[middle_slice], cmap="hot")
plt.title("Predicted Tumor Mask")
plt.axis("off")

output_path = "outputs/prediction.png"

plt.savefig(output_path)

print(f"Prediction saved at: {output_path}")

plt.show()