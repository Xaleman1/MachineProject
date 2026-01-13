# inspeccion_online.py
import time, serial, cv2, torch, numpy as np
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from collections import Counter

# ---------- CONFIG ----------
COM_PORT = "COM7"      # cambiar por tu puerto
BAUD = 115200
CAM_INDEX = 0
WINDOW_SECONDS = 1.2   # ventana para muestrear
IMG = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "modelos"
# --------------------------

# ---------- TRANSFORMACIONES ----------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG,IMG)),
    transforms.ToTensor()
])

# ---------- CLASIFICADOR (ResNet) ----------
clf = models.resnet18(pretrained=False)
clf.fc = nn.Linear(clf.fc.in_features, 3)
clf.load_state_dict(torch.load(f"{MODEL_DIR}/classifier_resnet18.pth", map_location=DEVICE))
clf = clf.to(DEVICE)
clf.eval()
classnames = ['leche','detergente','fideos']

# ---------- AUTOENCODER (misma arquitectura que en entrenamiento) ----------
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(True),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,2,1,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,16,3,2,1,1), nn.ReLU(True),
            nn.ConvTranspose2d(16,3,3,2,1,1), nn.Sigmoid()
        )
    def forward(self,x): return self.dec(self.enc(x))

# Cargar thresholds guardados (diccionario)
import json, os
thresholds_path = os.path.join(MODEL_DIR,"thresholds.npy")
if os.path.exists(thresholds_path):
    thresholds = np.load(thresholds_path, allow_pickle=True).item()
else:
    # umbrales por defecto (ajustar)
    thresholds = { 'leche':{'mean':0.002,'std':0.001}, 'detergente':{'mean':0.002,'std':0.001}, 'fideos':{'mean':0.002,'std':0.001} }

# ---------- SERIAL ----------
try:
    ser = serial.Serial(COM_PORT, BAUD, timeout=0.1)
    time.sleep(2)
    print("Serial conectado a", COM_PORT)
except Exception as e:
    print("No se pudo abrir serial:", e)
    ser = None

# ---------- CAMARA ----------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened(): raise RuntimeError("No se puede abrir cámara")

print("Esperando 'S' desde Arduino... pulsar ESC para salir")

while True:
    ret, frame = cap.read()
    if not ret: break

    if ser and ser.in_waiting > 0:
        line = ser.readline().decode().strip()
        if line.upper() == 'S':
            print("S recibido: muestreando ventana...")
            start = time.time()
            frames = []
            while time.time() - start < WINDOW_SECONDS:
                ret, f = cap.read()
                if not ret: break
                frames.append(f)
            if len(frames)==0:
                print("No frames")
                continue

            # 1) Determinar clase por mayoría en la ventana
            votes = []
            for f in frames:
                img = transform(f).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = clf(img)
                    pred = torch.argmax(out, dim=1).item()
                    votes.append(classnames[pred])
            pred_class = Counter(votes).most_common(1)[0][0]
            print("Clase detectada:", pred_class)

            # 2) Cargar autoencoder de esa clase
            ae = ConvAE().to(DEVICE)
            ae_path = f"{MODEL_DIR}/auto_{pred_class}.pth"
            ae.load_state_dict(torch.load(ae_path, map_location=DEVICE))
            ae.eval()

            # 3) Calcular error promedio en la ventana
            errs = []
            for f in frames:
                timg = transform(f).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    recon = ae(timg)
                mse = float(((timg - recon)**2).mean().cpu().numpy())
                errs.append(mse)
            avg_err = float(np.mean(errs))
            print("Avg MSE:", avg_err)

            # 4) Decisión con umbral
            th_info = thresholds.get(pred_class, None)
            if th_info:
                th = th_info.get("mean",0) + 3*th_info.get("std",0)
            else:
                th = 0.005
            decision = "BAD" if avg_err > th else "OK"
            print("Decision:", decision, "umbral:", th)

            # 5) Enviar a Arduino
            if ser:
                send = "DEFECTO\n" if decision=="BAD" else "BUENO\n"
                ser.write(send.encode())
                print("Enviado:", send.strip())

    # Mostrar frame y esperar ESC
    cv2.imshow("Inspeccion", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
if ser: ser.close()
