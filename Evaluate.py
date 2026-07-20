import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from Utils import plot_confusion_matrix

def evaluate_model(model, test_loader, device, class_names):
    """
    Evalúa el rendimiento del modelo en el conjunto de prueba y genera la matriz de confusión.

    Args:
        model (torch.nn.Module): El modelo a evaluar.
        test_loader (DataLoader): El DataLoader para el conjunto de prueba.
        device (torch.device): El dispositivo (CPU o GPU) en el que realizar la evaluación.
        class_names (list): Lista con los nombres de las clases.

    Returns:
        float: Precisión del modelo en el conjunto de prueba.
    """
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # Concatenar todos los resultados
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Calcular precisión
    accuracy = accuracy_score(all_labels, all_preds)

    # Calcular matriz de confusión
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Graficar la matriz de confusión
    plot_confusion_matrix(conf_matrix, class_names, title="Confusion Matrix")

    return accuracy
