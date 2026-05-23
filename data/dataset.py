import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


class BrainMRIDataset(Dataset):
    def __init__(
        self,
        root_dir="data/BraTS2024",
        target_size=(128,128,128)
    ):
        self.root_dir = root_dir
        self.target_size = target_size

        self.patients = [
            p for p in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, p))
        ]

        if len(self.patients) == 0:
            raise ValueError(f"No patient folders found in {root_dir}")

        print(f"Found {len(self.patients)} patient folders.")

    def load_nifti(self, path):
        img = nib.load(path).get_fdata().astype(np.float32)

        # Normalize intensity
        img = (img - img.mean()) / (img.std() + 1e-8)

        return img

    def resize_volume(self, volume, is_mask=False):
        tensor = torch.tensor(
            volume,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        if is_mask:
            tensor = F.interpolate(
                tensor,
                size=self.target_size,
                mode="nearest"
            )
        else:
            tensor = F.interpolate(
                tensor,
                size=self.target_size,
                mode="trilinear",
                align_corners=False
            )

        return tensor.squeeze(0).squeeze(0).numpy()

    def find_file(self, folder, suffix):
        for file in os.listdir(folder):
            if file.endswith(suffix):
                return os.path.join(folder, file)

        raise FileNotFoundError(
            f"Could not find file ending with '{suffix}' in {folder}"
        )

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        folder = os.path.join(self.root_dir, patient)

        # BraTS naming conventions
        t1_path = self.find_file(folder, "-t1n.nii.gz")
        t2_path = self.find_file(folder, "-t2w.nii.gz")
        flair_path = self.find_file(folder, "-t2f.nii.gz")
        seg_path = self.find_file(folder, "-seg.nii.gz")

        # Load volumes
        t1 = self.load_nifti(t1_path)
        t2 = self.load_nifti(t2_path)
        flair = self.load_nifti(flair_path)
        mask = self.load_nifti(seg_path)

        # Resize to fixed dimensions
        t1 = self.resize_volume(t1, is_mask=False)
        t2 = self.resize_volume(t2, is_mask=False)
        flair = self.resize_volume(flair, is_mask=False)
        mask = self.resize_volume(mask, is_mask=True)

        # Convert segmentation labels to binary mask
        mask = (mask > 0).astype(np.float32)

        # Stack modalities -> (3, D, H, W)
        image = np.stack([t1, t2, flair], axis=0)

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask