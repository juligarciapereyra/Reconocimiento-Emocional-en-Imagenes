import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def log_metrics(train_loss, train_acc, val_loss, val_accuracy, conf_matrix, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc, epoch_cm):
    """
    Almacena métricas de pérdida y precisión por época.

    Args:
        train_loss (float): La pérdida del modelo en el conjunto de entrenamiento.
        train_acc (float): La precisión del modelo en el conjunto de entrenamiento.
        val_loss (float): La pérdida del modelo en el conjunto de validación.
        val_accuracy (float): La precisión del modelo en el conjunto de validación.
        conf_matrix (np.ndarray): La matriz de confusión calculada para el modelo.
        epoch_train_loss (list): Lista que almacena las pérdidas de entrenamiento por época.
        epoch_train_acc (list): Lista que almacena las precisiones de entrenamiento por época.
        epoch_test_loss (list): Lista que almacena las pérdidas de validación por época.
        epoch_test_acc (list): Lista que almacena las precisiones de validación por época.
        epoch_cm (list): Lista que almacena las matrices de confusión por época.

    Retorna:
        tuple: Se retorna una tupla con las listas actualizadas:
            - `epoch_train_loss`: Pérdida de entrenamiento por época.
            - `epoch_train_acc`: Precisión de entrenamiento por época.
            - `epoch_test_loss`: Pérdida de validación por época.
            - `epoch_test_acc`: Precisión de validación por época.
    Descripción:
        Esta función tiene como objetivo registrar las métricas clave durante el entrenamiento y la validación
        del modelo. Las métricas incluyen:
            - Pérdida y precisión en el conjunto de entrenamiento.
            - Pérdida y precisión en el conjunto de validación.
            - La matriz de confusión que muestra el desempeño del modelo en términos de clasificación de las clases.

        Estas métricas se almacenan por época y se devuelven como listas actualizadas para su análisis posterior,
        como la visualización de curvas de desempeño o el ajuste del modelo.
    """
    # Almacenar las métricas de la época actual en sus respectivas listas
    epoch_train_loss.append(train_loss)
    epoch_train_acc.append(train_acc)
    epoch_test_loss.append(val_loss)
    epoch_test_acc.append(val_accuracy)
    epoch_cm.append(conf_matrix)

    return epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: list, title: str = "Confusion Matrix") -> None:
    """
    Grafica la matriz de confusión normalizada.

    Args:
        conf_matrix (np.ndarray): Matriz de confusión (2D).
        class_names (list): Lista con los nombres de las clases.
        title (str): Título del plot. Valor predeterminado es "Confusion Matrix".

    Descripción:
        Esta función grafica una matriz de confusión normalizada. La matriz de confusión es
        representada como un mapa de calor donde los valores se indican con anotaciones numéricas
        para mostrar el desempeño del modelo en términos de etiquetas verdaderas y predichas.

    La matriz de confusión se normaliza por filas, es decir, cada fila se divide por la suma de sus valores,
    lo que permite observar las proporciones de predicciones correctas e incorrectas.
    """
    # Normalizar por filas
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.show()


def plot_training_curves(train_loss, val_loss, train_acc, val_acc):
    """
    Grafica las curvas de pérdida y precisión durante el entrenamiento.

    Args:
        train_loss (list): Lista con la pérdida en entrenamiento por época.
        val_loss (list): Lista con la pérdida en validación por época.
        train_acc (list): Lista con la precisión en entrenamiento por época.
        val_acc (list): Lista con la precisión en validación por época.

    Descripción:
        Esta función visualiza las curvas de entrenamiento de pérdida y precisión para ambos conjuntos
        (entrenamiento y validación). Esto ayuda a monitorear el desempeño del modelo durante las épocas
        y a identificar posibles problemas como el sobreajuste (overfitting).

        La función crea dos gráficos:
        1. Pérdida a través de las épocas.
        2. Precisión a través de las épocas.
    """
    epochs = range(1, len(train_loss) + 1)

    # Pérdida
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', color='#D68F8F', linewidth=3)  # Rosa viejo
    plt.plot(epochs, val_loss, label='Validation Loss', color='#1565C0', linewidth=3)  # Azul
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Precisión
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy', color='#D68F8F', linewidth=3)  # Rosa viejo
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='#1565C0', linewidth=3)  # Azul
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_examples(samples, title, color, labels, mean=None, std=None):
    """
    Grafica ejemplos cualitativos con imágenes correctamente procesadas.

    Parámetros:
    - samples: lista de tuplas (imagen, etiqueta_real, etiqueta_predicha).
    - title: título del gráfico.
    - color: color del título.
    - labels: lista de nombres de las clases.
    - mean: lista con las medias usadas en la normalización (opcional).
    - std: lista con las desviaciones estándar usadas en la normalización (opcional).

    Descripción:
        Esta función permite visualizar imágenes que han sido clasificadas correctamente e incorrectamente.
        Cada imagen se visualiza junto con sus etiquetas reales y predichas. La normalización y desnormalización
        de las imágenes se realiza utilizando los parámetros `mean` y `std` si se proporcionan.

        La función es útil para realizar un análisis cualitativo de los ejemplos predichos por el modelo.
    """
    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=16, color=color)

    for i, (img, y_true, y_pred) in enumerate(samples):
        img = img.cpu().numpy().transpose(1, 2, 0)  # Convertir de tensor a numpy (C x H x W -> H x W x C)

        # Desnormalizar si se proporcionan mean y std
        if mean is not None and std is not None:
            img = img * std + mean
            img = np.clip(img, 0, 1)  # Limitar a valores válidos [0, 1]

        # Verificar si la imagen es escala de grises y convertir a RGB
        if img.shape[-1] == 1:  # Si solo hay un canal
            img = np.repeat(img, 3, axis=-1)  # Repetir el canal para convertirlo a RGB

        plt.subplot(1, len(samples), i + 1)
        plt.imshow(img)
        plt.title(f"Real: {labels[y_true]}\nPred: {labels[y_pred]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def show_qualitative_examples(model, dataloader, device, labels, n_examples=5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Muestra ejemplos cualitativos de imágenes clasificadas correctamente e incorrectamente.

    Parámetros:
    - model: modelo entrenado.
    - dataloader: DataLoader con datos de validación.
    - device: dispositivo (CPU o GPU).
    - labels: lista de nombres de las clases.
    - n_examples: cantidad de ejemplos a mostrar por categoría.
    - mean: lista con las medias usadas en la normalización (por canal).
    - std: lista con las desviaciones estándar usadas en la normalización (por canal).

    Descripción:
        Esta función recopila ejemplos correctamente e incorrectamente clasificados de las imágenes del conjunto
        de validación. Muestra las imágenes junto con las etiquetas reales y predichas para realizar un análisis
        visual de cómo el modelo está desempeñándose en las tareas de clasificación.
    """
    model.eval()
    correct_samples, incorrect_samples = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).argmax(dim=1)

            for i in range(len(y_batch)):
                if y_pred[i] == y_batch[i]:
                    correct_samples.append((X_batch[i], y_batch[i], y_pred[i]))
                else:
                    incorrect_samples.append((X_batch[i], y_batch[i], y_pred[i]))

                # Limitar el número de ejemplos a n_examples
                if len(correct_samples) >= n_examples and len(incorrect_samples) >= n_examples:
                    break

    # Mostrar ejemplos correctamente clasificados
    plot_examples(correct_samples[:n_examples], "Ejemplos Correctamente Clasificados", "green", labels, mean, std)

    # Mostrar ejemplos mal clasificados
    plot_examples(incorrect_samples[:n_examples], "Ejemplos Incorrectamente Clasificados", "red", labels, mean, std)

