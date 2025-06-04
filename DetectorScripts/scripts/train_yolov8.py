# kfold_yolov8_trainer.py

import os
import shutil
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO
import torch

# CONFIGURACIÓN
DATASET_IMAGES = Path("data_d/images/train")
DATASET_LABELS = Path("data_d/labels/train")
KFOLD_OUTPUT = Path("2_Detector\output\yolov8_kfold")
NUM_FOLDS = 5
NC = 5  # Número de clases
CLASS_NAMES = {
    0: "Hojas", # "Hoja" en español
    1: "Fresa No Madura", # "Fresa" en español
    2: "Flores",
    3: "Fresa Madura",
    4: "Defectos",
}

if __name__ == "__main__":
    print("🧠 Dispositivo activo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Recolectar imágenes
    all_images = sorted([p for p in DATASET_IMAGES.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    assert len(all_images) > 0, "No se encontraron imágenes para k-fold."

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\nFold {fold + 1}/{NUM_FOLDS}")

        # Preparar carpetas
        fold_dir = KFOLD_OUTPUT / f"fold{fold}_run"
        images_train = fold_dir / "images/train"
        images_val = fold_dir / "images/val"
        labels_train = fold_dir / "labels/train"
        labels_val = fold_dir / "labels/val"

        for d in [images_train, images_val, labels_train, labels_val]:
            d.mkdir(parents=True, exist_ok=True)

        # Copiar archivos
        for idx in train_idx:
            img = all_images[idx]
            lbl = DATASET_LABELS / f"{img.stem}.txt"
            shutil.copy2(img, images_train / img.name)
            if lbl.exists():
                shutil.copy2(lbl, labels_train / lbl.name)
            else:
                # Si no existe, crea un archivo vacío en la carpeta de destino
                print(f"Advertencia: {lbl} no existe, se creará vacío.")
                (labels_train / lbl.name).touch()

        for idx in val_idx:
            img = all_images[idx]
            lbl = DATASET_LABELS / f"{img.stem}.txt"
            shutil.copy2(img, images_val / img.name)
            if lbl.exists():
                shutil.copy2(lbl, labels_val / lbl.name)
            else:
                print(f"Advertencia: {lbl} no existe, se creará vacío.")
                (labels_val / lbl.name).touch()

        # Crear data.yaml
        data_yaml = fold_dir / "data.yaml"
        with open(data_yaml, "w") as f:
            f.write(f"train: {images_train.resolve().as_posix()}\n")
            f.write(f"val: {images_val.resolve().as_posix()}\n")
            f.write(f"nc: {NC}\n")
            f.write(f"names: {CLASS_NAMES}\n")

        # Entrenar modelo YOLOv8
        print(f"Entrenando modelo para fold {fold}...")
        model = YOLO("yolov8s.pt")
        model.train(
            data=str(data_yaml),
            epochs=10,  # Número de épocas
            imgsz=400,  # Tamaño de la imagen
            batch=16,    # Tamaño del lote
            amp=True,   # Aceleración de precisión mixta
            device=0,   # Forzar uso de GPU si está disponible
            project=str(KFOLD_OUTPUT),
            name=f"fold{fold}_run",
            exist_ok=True
        )

    print("\n✅ Entrenamiento K-Fold completado con YOLOv8.")
