# scripts/evaluate_model.py
import os
import sys
import numpy as np
import collections
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar ruta absoluta del proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from utils.preprocess import ImageFolderDataset
from utils.augmentation import get_data_transforms
from models.mobilenet_model import MobileNetV2Classifier
from models.resnet_model     import ResNet50Classifier


def evaluate_stratified_kfold(model_name, data_dir, output_dir, n_splits=5, batch_size=16):
    """
    Evaluación con StratifiedKFold:
    - Divide el dataset en folds.
    - Carga checkpoints best_{model}_fold_i.pth del directorio de salida.
    - Evalúa precisión y pérdida por fold, imprime distribuciones.
    - Calcula promedio general.
    - Guarda en 'eval_outputs':
      * classification_report_{model}.txt
      * confusion_matrix_global_{model}.png
      * folds_report_{model}.txt con Fold, Precisión y Pérdida, y promedio
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Dispositivo: {device}\n")

    # Crear carpeta eval_outputs dentro de output_dir
    eval_out = os.path.join(output_dir, 'eval_outputs')
    os.makedirs(eval_out, exist_ok=True)

    # Preparar dataset con transformaciones
    _, val_transform = get_data_transforms()
    ds = ImageFolderDataset(data_dir, transform=val_transform)
    labels = ds.targets if hasattr(ds, 'targets') else [s[1] for s in ds.samples]
    class_names = [k for k,_ in sorted(ds.class_to_idx.items(), key=lambda x: x[1])]

    # Recolectar checkpoints
    all_ckpts = glob(os.path.join(output_dir, '*.pth'))
    model_ckpts = sorted([p for p in all_ckpts if model_name in os.path.basename(p) and 'fold' in os.path.basename(p)])
    if not model_ckpts:
        print(f"[WARN] No se encontraron checkpoints para '{model_name}', usando todos los .pth")
        model_ckpts = sorted(all_ckpts)
    print(f"[INFO] Usando {len(model_ckpts)} checkpoints: {model_ckpts}\n")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    loss_fn = nn.CrossEntropyLoss()

    results, all_preds, all_trues, accuracies, losses = [], [], [], [], []

    print("📊 Resultados por fold:")
    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if fold_idx >= len(model_ckpts):
            break
        ckpt = model_ckpts[fold_idx]
        fold_name = os.path.basename(ckpt).replace('.pth', '')
        print(f"[INFO] Fold {fold_idx+1}: cargando {fold_name}")

        ds_val = Subset(ds, val_idx)
        loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

        # Carga modelo
        ModelCls = MobileNetV2Classifier if model_name=='mobilenet' else ResNet50Classifier
        model = ModelCls(len(ds.class_to_idx), pretrained=False).to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device))

        # Evalúa
        correct = total = 0
        fold_losses, preds_fold, trues_fold = [], [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = loss_fn(out, lbls).item(); fold_losses.append(loss)
                pred = out.argmax(1)
                correct += (pred==lbls).sum().item(); total += lbls.size(0)
                preds_fold.extend(pred.cpu().numpy()); trues_fold.extend(lbls.cpu().numpy())

        acc = correct/total; mean_loss = np.mean(fold_losses)
        accuracies.append(acc); losses.append(mean_loss)
        all_preds.append(np.array(preds_fold)); all_trues.append(np.array(trues_fold))
        results.append((fold_name, acc, mean_loss))

        print(f"[{fold_name}] Precisión={acc:.4f} | Pérdida={mean_loss:.4f}")
        print(f"  Verdadero: {collections.Counter(trues_fold)}")
        print(f"  Predicho:  {collections.Counter(preds_fold)}\n")

    # Promedio general
    avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
    print(f"📌 Promedio general: Precisión={avg_acc:.4f} | Pérdida={avg_loss:.4f}\n")

    # Reporte global
    all_pred = np.concatenate(all_preds)
    all_true = np.concatenate(all_trues)
    report = classification_report(all_true, all_pred, target_names=class_names, digits=4)
    print("[INFO] Reporte global:")
    print(report)
    report_txt = os.path.join(eval_out, f"classification_report_{model_name}.txt")
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[📄] Reporte guardado en: {report_txt}\n")

    # Matriz de confusión global
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión Global')
    plt.tight_layout()
    cm_png = os.path.join(eval_out, f"confusion_matrix_global_{model_name}.png")
    plt.savefig(cm_png)
    plt.close()
    print(f"[🖼️] Matriz guardada en: {cm_png}\n")

    # Guardar reporte de folds detallado (precisión, pérdida y promedio)
    folds_txt = os.path.join(eval_out, f"folds_report_{model_name}.txt")
    with open(folds_txt, 'w', encoding='utf-8') as f:
        f.write(f"{'Fold':<30}{'Precisión':<12}{'Pérdida':<12}{'Promedio':<12}\n")
        f.write('-'*66 + '\n')
        for name, a, l in results:
            f.write(f"{name:<30}{a:<12.4f}{l:<12.4f}\n") 
        f.write('-'*66 + '\n')
        f.write(f"{'PROMEDIO':<30}{avg_acc:<12.4f}{avg_loss:<12.4f}{((avg_acc + (1-avg_loss))/2):<12.4f}\n")
    print(f"[📄] Reporte folds guardado en: {folds_txt}\n")


def menu():
    while True:
        print("""
========================================
      Menú de Evaluación K-Fold
========================================
1) Evaluar MobileNetV2
2) Evaluar ResNet50
3) Salir
""")
        ch = input("Elige [1-3]: ").strip()
        if ch == '3':
            print("Saliendo…")
            break
        if ch not in ['1','2']:
            print("Opción inválida.")
            continue
        mdl = 'mobilenet' if ch=='1' else 'resnet'
        dd = input("Ruta al directorio de imágenes: ").strip()
        od = input("Ruta al directorio de salidas: ").strip()
        bs = input("Batch size [16]: ").strip() or '16'
        ns = input("Número de folds [5]: ").strip() or '5'
        try:
            bs = int(bs)
            ns = int(ns)
        except ValueError:
            print("Valores inválidos, usando por defecto.")
            bs, ns = 16, 5
        evaluate_stratified_kfold(mdl, dd, od, n_splits=ns, batch_size=bs)

if __name__ == '__main__':
    menu()
