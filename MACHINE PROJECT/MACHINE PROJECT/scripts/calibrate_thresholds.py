# calibrate_thresholds.py
import os, torch, numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_autoencoders import ConvAE  # o define ConvAE aquí
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG=128
transform = transforms.Compose([transforms.Resize((IMG,IMG)), transforms.ToTensor()])

results = {}
for clase in ["leche","detergente","fideos"]:
    model = ConvAE().to(DEVICE)
    model.load_state_dict(torch.load(f"modelos/auto_{clase}.pth", map_location=DEVICE))
    model.eval()
    folder = os.path.join("dataset", clase, "valid")  # valid contains buenos images
    ds = datasets.ImageFolder(os.path.join("dataset", clase), transform=transform)  # adapt as needed
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    errors = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            mse = ((imgs - out)**2).mean(dim=[1,2,3]).cpu().numpy()
            errors.extend(mse.tolist())
    errors = np.array(errors)
    mean,std = errors.mean(), errors.std()
    p95 = np.percentile(errors,95)
    results[clase] = {"mean":float(mean),"std":float(std),"p95":float(p95)}
    print(clase, results[clase])
np.save("modelos/thresholds.npy", results)
