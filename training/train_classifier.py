import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from data.dataset import BrainMRIDataset
from models.classifier import TumorClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClassificationDataset(BrainMRIDataset):
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        tumor = image * mask

        label = self.get_label(self.patients[idx])

        return tumor, torch.tensor(label, dtype=torch.long)

    def get_label(self, patient_name):
        if "glioma" in patient_name.lower():
            return 0
        elif "meningioma" in patient_name.lower():
            return 1
        else:
            return 2


dataset = ClassificationDataset("data/raw")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = TumorClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10

for epoch in range(epochs):
    model.train()
    loop = tqdm(loader)

    for img, label in loop:
        img = img.to(device)
        label = label.to(device)

        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "classifier.pth")