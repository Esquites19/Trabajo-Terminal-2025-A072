import sys, os
# Asegura imports desde la raíz del proyecto

top = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if top not in sys.path:
    sys.path.insert(0, top)

import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from models.mobilenet_model import MobileNetV2Classifier
from models.resnet_model     import ResNet50Classifier
from utils.preprocess        import ImageFolderDataset
from utils.augmentation      import get_data_transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


def train(model_name, data_dir, output_dir, num_epochs=10, batch_size=16):
    # Selección de dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n==> Usando dispositivo: {device}')
    if device.type == 'cuda':
        print(f'    GPUs disponibles: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'      GPU {i}: {torch.cuda.get_device_name(i)}')
    torch.backends.cudnn.benchmark = True

    eval_out = os.path.join(output_dir, 'eval_outputs')

    # Comprobación de rutas
    if not os.path.isdir(data_dir):
        print(f"Error: no existe carpeta de datos '{data_dir}'")
        return
    os.makedirs(output_dir, exist_ok=True)

    # Preparar datos y transformaciones
    train_transform, val_transform = get_data_transforms()
    dataset = ImageFolderDataset(data_dir, transform=None)
    with open(os.path.join(output_dir, 'class_map.json'), 'w') as f:
        json.dump(dataset.class_to_idx, f)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    num_classes = len(dataset.class_to_idx)

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset))), 1):
        print(f'\n========== FOLD {fold}/{kf.n_splits} ==========' )
        fold_start = time.time()

        # Crear DataLoaders
        train_ds = Subset(dataset, train_idx)
        train_ds.dataset.transform = train_transform
        val_ds   = Subset(dataset, val_idx)
        val_ds.dataset.transform   = val_transform
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Instanciar modelo
        cls = MobileNetV2Classifier if model_name=='mobilenet' else ResNet50Classifier
        model = cls(num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)
        best_val_loss = float('inf')

        # Listas para métricas
        train_losses, val_losses = [], []
        train_accs, val_accs     = [], []

        # Bucle de épocas
        for epoch in range(1, num_epochs+1):
            epoch_start = time.time()
            # Entrenamiento
            model.train()
            train_loss = train_correct = 0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
                train_loss   += loss.item() * imgs.size(0)
                train_correct+= (outputs.argmax(1)==lbls).sum().item()
            avg_train_loss = train_loss / len(train_loader.dataset)
            train_acc = 100 * train_correct / len(train_loader.dataset)

            # Validación
            model.eval()
            val_loss = val_correct = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                    val_loss   += loss.item() * imgs.size(0)
                    val_correct+= (outputs.argmax(1)==lbls).sum().item()
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_acc = 100 * val_correct / len(val_loader.dataset)

            # Scheduler y log de época
            scheduler.step(avg_val_loss)
            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch}/{num_epochs} | '
                  f'TrainAcc: {train_acc:.2f}% | ValAcc: {val_acc:.2f}% | Tiempo: {epoch_time:.1f}s')

            # Guardar métricas
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Guardar mejor modelo
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ckpt = os.path.join(output_dir, f"{model_name}_fold{fold}.pth")
                torch.save(model.state_dict(), ckpt)
                print(f'   [INFO] Mejor modelo guardado en: {ckpt}')

        # Generar gráfica de pérdidas y guardar en PNG
        plt.figure()
        epochs_range = list(range(1, num_epochs+1))
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curves Fold {fold}')
        plt.legend()
        plt.tight_layout()
        loss_png = os.path.join(eval_out, f"loss_{model_name}_fold{fold}.png")
        plt.savefig(loss_png)
        plt.close()
        print(f'   [INFO] Gráfica de pérdida guardada en: {loss_png}')

        # Generar gráfica de precisión y guardar en PNG
        plt.figure()
        plt.plot(epochs_range, train_accs, label='Train Acc')
        plt.plot(epochs_range, val_accs, label='Validation Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy Curves Fold {fold}')
        plt.legend()
        plt.tight_layout()
        acc_png = os.path.join(eval_out, f"accuracy_{model_name}_fold{fold}.png")
        plt.savefig(acc_png)
        plt.close()
        print(f'   [INFO] Gráfica de precisión guardada en: {acc_png}')

        fold_time = time.time() - fold_start
        print(f'=== Fin Fold {fold} en {fold_time:.1f}s ===')

def menu():
    while True:
        print("""
====================================
    Menú de Entrenamiento de Modelo
====================================
1) Entrenar MobileNetV2
2) Entrenar ResNet50
3) Salir
""")
        opt = input("Selecciona una opción [1-3]: ").strip()
        if opt == '3':
            print("Saliendo…"); break
        if opt not in ['1','2']:
            print("Opción inválida."); continue

        model_name = 'mobilenet' if opt=='1' else 'resnet'
        data_dir   = input("Ruta al directorio de entrenamiento (e.g., 'data/train'): ").strip()
        output_dir = input("Ruta de salida para checkpoints [output]: ").strip() or os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')), 'output')
        bs         = input("Batch size [16]: ").strip() or "16"
        epochs     = input("Número de épocas [10]: ").strip() or "10"

        try:
            batch_size = int(bs)
            num_epochs = int(epochs)
        except ValueError:
            print("Batch size o número de épocas inválido, usando valores por defecto.")
            batch_size, num_epochs = 16, 10

        train(model_name, data_dir, output_dir, num_epochs, batch_size)

if __name__ == '__main__':
    menu()
