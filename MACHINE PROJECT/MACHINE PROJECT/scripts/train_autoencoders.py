# train_autoencoders.py
import os, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim
from tqdm import tqdm

DATA_DIR = "dataset"
IMG = 128
BATCH = 16
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "modelos"
os.makedirs(MODEL_DIR, exist_ok=True)

transform = transforms.Compose([transforms.Resize((IMG,IMG)), transforms.ToTensor()])

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(True),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,2,1,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,16,3,2,1,1), nn.ReLU(True),
            nn.ConvTranspose2d(16,3,3,2,1,1), nn.Sigmoid()
        )
    def forward(self,x): return self.dec(self.enc(x))

for clase in ["leche","detergente","fideos"]:
    print("Training AE for", clase)
    folder = os.path.join(DATA_DIR, clase, "buenos")
    ds = datasets.ImageFolder(os.path.join(DATA_DIR, clase), transform=transform)  # expects subfolder 'buenos'
    # If structure is dataset/leche/buenos/*, ImageFolder needs top folder, so adjust:
    # Alternative: use datasets.ImageFolder(folder) with dummy label; for simplicity assume dataset/<clase>/buenos as single dir
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model = ConvAE().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.MSELoss()
    for ep in range(EPOCHS):
        model.train()
        running = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            loss = lossfn(out, imgs)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"  ep {ep+1}/{EPOCHS} loss {running/len(loader):.6f}")
    fname = os.path.join(MODEL_DIR, f"auto_{clase}.pth")
    torch.save(model.state_dict(), fname)
    print("Saved", fname)
