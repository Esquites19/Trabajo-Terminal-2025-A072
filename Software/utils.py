import os
import json
import cv2
import colorsys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from models.mobilenet_model import MobileNetV2Classifier
from models.resnet_model import ResNet50Classifier
from ultralytics import YOLO

# Transformaciones y utilidades de ML
transform_class = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def cargar_modelo_clasificacion(path, device):
    """Carga un modelo de clasificación y su mapeo de clases."""
    model_dir = os.path.dirname(path)
    cmap_file = os.path.join(model_dir, 'class_map.json')
    with open(cmap_file, 'r') as f:
        class_map = json.load(f)
    idx_to_class = {int(v): k for k, v in class_map.items()}
    num_classes = len(idx_to_class)
    if 'mobilenet' in path.lower():
        model = MobileNetV2Classifier(num_classes=num_classes, pretrained=False)
    elif 'resnet' in path.lower():
        model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model, idx_to_class


def clasificar_frame(model, idx_to_class, frame, device):
    """Clasifica un frame (BGR) y retorna 'Etiqueta: Confianza'."""
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform_class(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = F.softmax(out, dim=1)
        p, idx = probs.max(1)
    label = idx_to_class[idx.item()]
    return f"{label}: {p.item()*100:.2f}%"


def cargar_detector(path):
    """Carga modelo YOLO para detección."""
    det = YOLO(path)
    det.fuse()
    return det


def generar_mapa_colores(class_ids):
    """Genera colores HSV equidistantes para etiquetas."""
    colors = {}
    unique_ids = sorted(set(class_ids))
    total = len(unique_ids)
    for i, cid in enumerate(unique_ids):
        hue = i / max(total, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors[cid] = (int(r*255), int(g*255), int(b*255))
    return colors


def detectar_frame(detector, frame, threshold=0.5, device=None):
    """Detecta objetos con YOLO y devuelve boxes y colores."""
    res = detector(frame, device=0 if device is None and torch.cuda.is_available() else device)[0]
    boxes = [b for b in res.boxes if float(b.conf) > threshold]
    ids = [int(b.cls) for b in boxes]
    colors = generar_mapa_colores(ids)
    return boxes, colors

def generar_reporte_pdf(counts: dict, class_history: list, output_path: str, last_classification: str):
    """
    Genera un PDF con:
      – Conteo acumulado de detecciones (counts)
      – Última clasificación (last_classification)
      – Historial de clasificaciones nuevas (class_history)
    """
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Reporte de SightBerry – v0.1.0")

    # Sección de detecciones
    c.setFont("Helvetica", 12)
    y = height - 80
    c.drawString(50, y, "Detecciones acumuladas:")
    y -= 20
    for label, cnt in counts.items():
        c.drawString(70, y, f"{label}: {cnt}")
        y -= 15

    # Última clasificación
    y -= 10
    c.drawString(50, y, f"Última clasificación: {last_classification}")
    y -= 20

    # Historial de clasificaciones nuevas
    c.drawString(50, y, "Historial de clasificaciones nuevas:")
    y -= 20
    for cls in class_history:
        c.drawString(70, y, f"- {cls}")
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
