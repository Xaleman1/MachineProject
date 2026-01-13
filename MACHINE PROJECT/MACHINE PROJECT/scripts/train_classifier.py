# train_classifier.py
import torch, os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim

DATA_DIR = "dataset"
BATCH = 32
EPOCHS = 10
IMG = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor()
])

# dataset espera estructura dataset/<class_name>/<subfolders...>
# We'll build a classifier dataset with just the three class folders
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3 clases
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    print(f"Epoch {epoch+1}, loss {running/len(train_loader):.4f}")

torch.save(model.state_dict(), "modelos/classifier_resnet18.pth")
print("Classifier saved")
''