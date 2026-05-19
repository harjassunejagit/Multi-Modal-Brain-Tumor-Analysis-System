import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.patients = os.listdir(root_dir)

    def load_nifti(self, path):
        img = nib.load(path).get_fdata()
        img = (img - img.mean()) / (img.std() + 1e-8)
        return img.astype(np.float32)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        folder = os.path.join(self.root_dir, p)

        t1 = self.load_nifti(os.path.join(folder, "T1.nii.gz"))
        t2 = self.load_nifti(os.path.join(folder, "T2.nii.gz"))
        flair = self.load_nifti(os.path.join(folder, "FLAIR.nii.gz"))
        mask = self.load_nifti(os.path.join(folder, "seg.nii.gz"))

        image = np.stack([t1, t2, flair], axis=0)

        return torch.tensor(image), torch.tensor(mask).unsqueeze(0)