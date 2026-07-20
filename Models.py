from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights, vgg16, vgg19, resnet50, ResNet50_Weights
from torch import nn
import torch.nn.functional as F
import torch

def get_resnet50(num_classes=7):
    """
    Configura el modelo ResNet-50 para transfer learning.

    Args:
        num_classes (int): Número de clases en el problema actual.

    Returns:
        torch.nn.Module: Modelo ResNet-50 ajustado.
    """
    # Cargar el modelo ResNet-50 con pesos preentrenados de ImageNet
    weights = ResNet50_Weights.IMAGENET1K_V1  # Pesos preentrenados
    model = resnet50(weights=weights)

    # Congelar todas las capas excepto las últimas subcapas de `layer4` y `fc`
    for name, param in model.named_parameters():
        if "layer4.2" not in name and "fc" not in name:  # Solo la última subcapa de layer4 y fc quedan entrenables
            param.requires_grad = False

    # Modificar la capa totalmente conectada (`fc`) para que sea más liviana
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),  # Reducir dimensionalidad
        nn.ReLU(),
        nn.Dropout(0.5),               # Regularización
        nn.Linear(256, num_classes)    # Adaptar a num_classes
    )

    return model

def get_efficientnet_b5(num_classes=7):
    """
    Configura el modelo EfficientNet-B5 para transfer learning.

    Args:
        num_classes (int): Número de clases en el problema actual.

    Returns:
        torch.nn.Module: Modelo EfficientNet-B5 ajustado.
    """
    # Cargar EfficientNet-B5 con pesos preentrenados
    weights = EfficientNet_B5_Weights.IMAGENET1K_V1  # Pesos preentrenados de ImageNet
    model = efficientnet_b5(weights=weights)

    # Congelar todas las capas excepto las últimas capas (features[6:], classifier)
    for name, param in model.named_parameters():
        if "features.6" not in name and "features.7" not in name and "classifier" not in name:
            param.requires_grad = False

    # Modificar la capa final `classifier`
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),  # Mayor capacidad de representación
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),           # Otra capa intermedia
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)    # Ajustar al número de clases
    )

    return model

class ResidualBlock(nn.Module):
    """
    Define un bloque residual para redes convolucionales.

    Args:
        in_channels (int): Número de canales de entrada.
        out_channels (int): Número de canales de salida.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Propagación hacia adelante del bloque residual.

        Args:
            x (torch.Tensor): Entrada.

        Returns:
            torch.Tensor: Salida después de aplicar el bloque residual.
        """
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)  # Skip connection (residual connection)
        out = F.relu(out)
        return out

class CustomCNN(nn.Module):
    """
    Define una red convolucional simple con bloques residuales.

    Args:
        num_classes (int): Número de clases en el problema.
    """
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # Capa convolucional 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Capa convolucional 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Capa convolucional 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Bloques residuales
        self.res_block1 = ResidualBlock(128, 256)
        self.res_block2 = ResidualBlock(256, 512)

        # Capa totalmente conectada
        self.fc1 = nn.Linear(512 * 7 * 7, 256)  # Ajuste del tamaño de entrada
        self.fc2 = nn.Linear(256, num_classes)  # Salida con el número de clases

        # Dropout para evitar overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Propagación hacia adelante.

        Args:
            x (torch.Tensor): Entrada.

        Returns:
            torch.Tensor: Salida después de aplicar todas las capas.
        """
        # Bloque convolucional 1 + Pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Bloque convolucional 2 + Pooling
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Bloque convolucional 3 + Pooling
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Bloques residuales
        x = self.res_block1(x)
        x = self.pool(x)

        x = self.res_block2(x)
        x = self.pool(x)

        # Aplanar las características
        x = torch.flatten(x, start_dim=1)

        # Capas totalmente conectadas
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Aplicar dropout después de la primera capa FC
        x = self.fc2(x)  # Sin activación aquí porque usamos CrossEntropyLoss

        return x

  
def get_vgg(num_classes):
    model = vgg16(weights="DEFAULT")  # Modelo preentrenado en ImageNet

    # Congelar todas las capas convolucionales inicialmente
    for params in model.features.parameters():
        params.requires_grad = False

    # Descongelar las últimas capas convolucionales del bloque final
    for name, param in model.features.named_parameters():
        if "28" in name:
            param.requires_grad = True

    # Modificar el clasificador para hacerlo más liviano
    in_features = model.classifier[0].in_features  # Salida de AdaptiveAvgPool2d
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),  # Reducir el tamaño a 512
        nn.ReLU(),
        nn.Dropout(0.5),               # Regularización
        nn.Linear(256, num_classes)   # Salida final para las clases
    )

    return model
