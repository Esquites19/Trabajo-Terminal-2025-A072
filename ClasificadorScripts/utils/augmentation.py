from torchvision import transforms

def get_data_transforms():
    """
    Retorna transformaciones para entrenamiento y validación.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=20, translate=(0.2,0.2), shear=0.15, scale=(0.8,1.2)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    return train_transform, val_transform
