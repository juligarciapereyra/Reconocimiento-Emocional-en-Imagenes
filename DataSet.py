import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

def load_data(directory, emotion_mapping):
    """
    Carga imágenes y etiquetas desde un directorio, asignando etiquetas numéricas según un mapeo de emociones.

    Args:
        directory (str): Ruta al directorio raíz que contiene subcarpetas por emoción.
        emotion_mapping (dict): Diccionario que asigna emociones (nombres de carpetas) a etiquetas numéricas.

    Returns:
        tuple: Arrays de datos (imágenes en blanco y negro) y etiquetas correspondientes.
    """
    data = []
    labels = []
    for emotion, label in emotion_mapping.items():
        emotion_folder = os.path.join(directory, emotion)
        for filename in os.listdir(emotion_folder):
            filepath = os.path.join(emotion_folder, filename)
            try:
                img = Image.open(filepath).convert("L")  # Convertir a blanco y negro
                data.append(np.array(img))
                labels.append(label)
            except Exception as e:
                print(f"Error al procesar {filepath}: {e}")

    return np.array(data), np.array(labels)


def create_checkpoint_dir(checkpoint_dir):
    """
    Crea un directorio para guardar checkpoints si no existe.

    Args:
        checkpoint_dir (str): Ruta al directorio donde se guardarán los checkpoints.

    Returns:
        str: Ruta del directorio creado o existente.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print(f"Checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir


def image_preprocess_transforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], img_size=(224, 224)):
    """
    Define transformaciones de preprocesamiento para imágenes.

    Args:
        mean (list): Lista de medias para normalización.
        std (list): Lista de desviaciones estándar para normalización.
        img_size (tuple): Tamaño al que se redimensionarán las imágenes (alto, ancho).

    Returns:
        torchvision.transforms.Compose: Transformaciones aplicadas a las imágenes.
    """
    preprocess = T.Compose([
        T.Resize(img_size, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return preprocess


def get_X_Y_arrays(path_to_fer, emotion_mapping, channels):
    """
    Obtiene los conjuntos de datos y etiquetas para entrenamiento y prueba.

    Args:
        path_to_fer (str): Ruta al directorio raíz que contiene las carpetas "train" y "test".
        channels (int): Número de canales deseados para las imágenes.

    Returns:
        tuple: Conjuntos de datos y etiquetas para entrenamiento y prueba.
    """
    train_data, train_labels = load_data(os.path.join(path_to_fer, "train"), emotion_mapping)
    test_data, test_labels = load_data(os.path.join(path_to_fer, "test"), emotion_mapping)

    X_train = np.stack([train_data] * channels, axis=-1)
    X_test = np.stack([test_data] * channels, axis=-1)

    Y_train = np.array(train_labels)
    Y_test = np.array(test_labels)
    return X_train, X_test, Y_train, Y_test


class CustomDataset(Dataset):
    """
    Dataset personalizado para admitir imágenes, etiquetas y transformaciones opcionales.

    Args:
        images (array-like): Conjunto de imágenes.
        labels (array-like): Etiquetas correspondientes.
        transform (callable, optional): Transformaciones a aplicar a cada imagen.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Obtiene un elemento del dataset.

        Args:
            idx (int): Índice del elemento a obtener.

        Returns:
            tuple: Imagen transformada y etiqueta correspondiente.
        """
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_datasets(X_train, Y_train, X_test, Y_test, preprocess_fn):
    """
    Crea los datasets de entrenamiento y prueba con transformaciones aplicadas.

    Args:
        X_train (array-like): Imágenes de entrenamiento.
        Y_train (array-like): Etiquetas de entrenamiento.
        X_test (array-like): Imágenes de prueba.
        Y_test (array-like): Etiquetas de prueba.
        preprocess_fn (callable): Función de transformación para las imágenes.

    Returns:
        tuple: Datasets de entrenamiento y prueba.
    """
    train_dataset = CustomDataset(X_train, Y_train, transform=preprocess_fn)
    test_dataset = CustomDataset(X_test, Y_test, transform=preprocess_fn)
    return train_dataset, test_dataset


def split_train_valid(dataset, labels, valid_split=0.2, random_state=42):
    """
    Divide el dataset de entrenamiento en conjuntos de entrenamiento y validación.

    Args:
        dataset (torch.utils.data.Dataset): Dataset de entrenamiento completo.
        labels (array-like): Etiquetas correspondientes al dataset.
        valid_split (float, optional): Proporción del conjunto de validación. Default es 0.2.
        random_state (int, optional): Semilla para reproducibilidad. Default es 42.

    Returns:
        tuple: Subsets de entrenamiento y validación, junto con los índices usados.
    """
    dataset_size = len(dataset)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_split, random_state=random_state)
    for train_indices, valid_indices in splitter.split(range(dataset_size), labels):
        train_data = Subset(dataset, train_indices)
        valid_data = Subset(dataset, valid_indices)
        return train_data, valid_data, train_indices, valid_indices


def create_data_loaders(train_data, valid_data, test_data, sampler, data_config):
    """
    Crea DataLoaders para los conjuntos de entrenamiento, validación y prueba.

    Args:
        train_data (torch.utils.data.Dataset): Dataset de entrenamiento.
        valid_data (torch.utils.data.Dataset): Dataset de validación.
        test_data (torch.utils.data.Dataset): Dataset de prueba.
        sampler (callable): Estrategia de muestreo para el DataLoader.
        data_config (object): Configuración de los DataLoaders (BATCH_SIZE y NUM_WORKERS).

    Returns:
        tuple: DataLoaders para entrenamiento, validación y prueba.
    """
    train_loader = DataLoader(train_data, batch_size=data_config.BATCH_SIZE, sampler=sampler, num_workers=data_config.NUM_WORKERS)
    valid_loader = DataLoader(valid_data, batch_size=data_config.BATCH_SIZE, shuffle=False, num_workers=data_config.NUM_WORKERS)
    testing_loader = DataLoader(test_data, batch_size=data_config.BATCH_SIZE, shuffle=False, num_workers=data_config.NUM_WORKERS)

    return train_loader, valid_loader, testing_loader


def create_data_loaders(train_data, valid_data, test_data, data_config):
    """
    Crea DataLoaders para los conjuntos de entrenamiento, validación y prueba.

    Args:
        train_data (torch.utils.data.Dataset): Dataset de entrenamiento.
        valid_data (torch.utils.data.Dataset): Dataset de validación.
        test_data (torch.utils.data.Dataset): Dataset de prueba.
        data_config (object): Configuración de los DataLoaders (BATCH_SIZE y NUM_WORKERS).

    Returns:
        tuple: DataLoaders para entrenamiento, validación y prueba.
    """
    train_loader = DataLoader(train_data, batch_size=data_config.BATCH_SIZE, num_workers=data_config.NUM_WORKERS)

    valid_loader = DataLoader(valid_data, batch_size=data_config.BATCH_SIZE, shuffle=False,num_workers=data_config.NUM_WORKERS)

    testing_loader = DataLoader(test_data, batch_size=data_config.BATCH_SIZE, shuffle=False,num_workers=data_config.NUM_WORKERS)

    return train_loader, valid_loader, testing_loader
