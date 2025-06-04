import os
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderDataset(Dataset):
    """
    Carga imágenes y etiquetas desde un directorio estilo ImageFolder para PyTorch.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Detecta subcarpetas como clases
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # Construye lista de (ruta_imagen, etiqueta)
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
