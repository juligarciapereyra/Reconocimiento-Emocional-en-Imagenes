import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import confusion_matrix
import numpy as np
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
import time
from Utils import log_metrics, plot_training_curves, plot_confusion_matrix, show_qualitative_examples
from SaveModels import save_best_model, save_model_weights

bold = f"\033[1m"
reset = f"\033[0m"


def train(
    DEVICE: torch.device,
    dataset_config,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    total_epochs: int,
    class_weights: torch.Tensor
) -> Tuple[float, float]:
    """
    Entrena el modelo durante una época.

    Args:
        DEVICE (torch.device): Dispositivo de entrenamiento (CPU o CUDA).
        train_config (TrainingConfig): Configuración del entrenamiento.
        dataset_config (DatasetConfig): Configuración del dataset.
        model (nn.Module): El modelo a entrenar.
        optimizer (torch.optim.Optimizer): Optimizer para actualizar los pesos del modelo.
        train_loader (torch.utils.data.DataLoader): DataLoader con los datos de entrenamiento.
        epoch_idx (int): Índice de la época actual.
        total_epochs (int): Número total de épocas.
        class_weights (torch.Tensor): Pesos para cada clase en la función de pérdida (CrossEntropy).

    Retorna:
        Tuple[float, float]: Retorna la pérdida promedio y la precisión promedio de la época.
        
    Descripción:
        Esta función entrena el modelo durante una época. En cada iteración, realiza un pase hacia adelante, 
        calcula la pérdida usando la función `cross_entropy`, realiza el cálculo de gradientes, actualiza los 
        pesos del modelo y calcula la precisión para el lote actual. Al final de la época, retorna la pérdida y
        la precisión promedio de la época.
    """
    model.train()
    model.to(DEVICE)

    acc_metric = MulticlassAccuracy(num_classes=dataset_config.NUM_CLASSES, average="micro")
    mean_metric = MeanMetric()

    status = f"Train:\t{bold}Epoch: {epoch_idx}/{total_epochs}{reset}"

    prog_bar = tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    prog_bar.set_description(status)

    for data, target in prog_bar:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()

        output = model(data)

        loss = F.cross_entropy(output, target, weight=class_weights)

        loss.backward()

        optimizer.step()

        batch_loss = mean_metric(loss.item(), weight=data.shape[0])

        prob = F.softmax(output, dim=1)
        pred_idx = prob.detach().argmax(dim=1)

        batch_acc = acc_metric(pred_idx.cpu(), target.cpu())

        step_status = status + f"\tLoss: {mean_metric.compute():.4f}, Acc: {acc_metric.compute():.4f}"
        prog_bar.set_description(step_status)

    epoch_loss = mean_metric.compute()
    epoch_acc = acc_metric.compute()

    prog_bar.close()

    return epoch_loss, epoch_acc


def validate(
    DEVICE,
    dataset_config,
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    total_epochs: int
) -> Tuple[float, float, np.ndarray]:
    """
    Valida el modelo en el conjunto de prueba y calcula la matriz de confusión.

    Args:
        DEVICE (torch.device): Dispositivo de validación (CPU o CUDA).
        train_config (TrainingConfig): Configuración del entrenamiento.
        dataset_config (DatasetConfig): Configuración del dataset.
        model (nn.Module): El modelo a evaluar.
        test_loader (torch.utils.data.DataLoader): DataLoader con los datos de validación.
        epoch_idx (int): Índice de la época actual.
        total_epochs (int): Número total de épocas.

    Retorna:
        Tuple[float, float, np.ndarray]: Retorna la pérdida, precisión y la matriz de confusión de la validación.

    Descripción:
        Esta función evalúa el rendimiento del modelo en el conjunto de validación (o prueba) durante una época.
        Calcula la pérdida y precisión del modelo y también genera la matriz de confusión para evaluar el
        desempeño en cada clase. La función retorna la pérdida, precisión y matriz de confusión calculada a lo largo
        de todos los lotes del conjunto de validación.
    """
    model.eval()
    model.to(DEVICE)

    acc_metric = MulticlassAccuracy(num_classes=dataset_config.NUM_CLASSES, average="micro")
    mean_metric = MeanMetric()

    all_targets = []
    all_preds = []

    status = f"Valid:\t{bold}Epoch: {epoch_idx}/{total_epochs}{reset}"
    prog_bar = tqdm(test_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    prog_bar.set_description(status)

    for data, target in prog_bar:
        data, target = data.to(DEVICE), target.to(DEVICE)

        with torch.no_grad():
            output = model(data)

        test_loss = F.cross_entropy(output, target).item()
        prob = F.softmax(output, dim=1)
        pred_idx = prob.detach().argmax(dim=1)

        batch_loss = mean_metric(test_loss, weight=data.shape[0])
        batch_acc = acc_metric(pred_idx.cpu(), target.cpu())

        all_targets.extend(target.cpu().numpy())
        all_preds.extend(pred_idx.cpu().numpy())

        step_status = status + f"\tLoss: {mean_metric.compute():.4f}, Acc: {acc_metric.compute():.4f}"
        prog_bar.set_description(step_status)

    test_loss = mean_metric.compute()
    test_acc = acc_metric.compute()

    conf_matrix = confusion_matrix(all_targets, all_preds)

    prog_bar.close()

    return test_loss, test_acc, conf_matrix


def train_and_validate(
    DEVICE, train_config, data_config, model, optimizer, train_loader, test_loader, epoch, class_weights
):
    """
    Entrena y valida el modelo para una época.

    Args:
        DEVICE (torch.device): Dispositivo de entrenamiento (CPU o CUDA).
        train_config (TrainingConfig): Configuración del entrenamiento.
        data_config (DatasetConfig): Configuración del dataset.
        model (nn.Module): El modelo a entrenar y validar.
        optimizer (torch.optim.Optimizer): Optimizer usado para el entrenamiento.
        train_loader (torch.utils.data.DataLoader): DataLoader para los datos de entrenamiento.
        test_loader (torch.utils.data.DataLoader): DataLoader para los datos de validación.
        epoch (int): Índice de la época actual.
        class_weights (torch.Tensor): Pesos para cada clase en la función de pérdida.

    Retorna:
        Tuple[float, float, float, float, np.ndarray]: Retorna las métricas de la época (pérdida y precisión en entrenamiento y validación), 
        y la matriz de confusión calculada en la validación.

    Descripción:
        Esta función entrena el modelo durante una época y luego lo valida en el conjunto de prueba. 
        Primero se entrena el modelo, calculando la pérdida y precisión en el conjunto de entrenamiento, 
        luego se valida el modelo en el conjunto de prueba calculando la pérdida, precisión y la matriz de confusión.
    """
    train_loss, train_acc = train(
        DEVICE, data_config, model, optimizer, train_loader, epoch, train_config.EPOCHS, class_weights
    )
    val_loss, val_accuracy, conf_matrix = validate(
        DEVICE, data_config, model, test_loader, epoch, train_config.EPOCHS
    )
    return train_loss, train_acc, val_loss, val_accuracy, conf_matrix


def train_model(
    DEVICE, model, optimizer, train_loader, valid_loader, train_config, data_config, ckpt_dir, model_dir, model_name, class_weights
) -> dict:
    """
    Entrena el modelo durante varias épocas y devuelve el historial y la matriz de confusión final.

    Args:
        DEVICE (torch.device): Dispositivo de entrenamiento (CPU o CUDA).
        model (nn.Module): El modelo a entrenar.
        optimizer (torch.optim.Optimizer): Optimizer usado para el entrenamiento.
        train_loader (torch.utils.data.DataLoader): DataLoader para los datos de entrenamiento.
        valid_loader (torch.utils.data.DataLoader): DataLoader para los datos de validación.
        train_config (TrainingConfig): Configuración del entrenamiento.
        data_config (DatasetConfig): Configuración del dataset.
        ckpt_dir (str): Directorio donde se guardarán los pesos del modelo.
        model_dir (str): Directorio para guardar el modelo.
        model_name (str): Nombre del modelo.
        class_weights (torch.Tensor): Pesos de clase para la función de pérdida.

    Retorna:
        dict: Un diccionario con el historial de entrenamiento, precisión final, y la matriz de confusión final.

    Descripción:
        Esta función entrena el modelo durante varias épocas y guarda el mejor modelo basado en el rendimiento
        en el conjunto de validación. También guarda el historial de métricas (pérdida y precisión) y finalmente
        genera gráficos con las curvas de entrenamiento y la matriz de confusión del modelo.
    """
    best_loss = torch.tensor(np.inf)
    best_weights = None

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_test_loss = []
    epoch_test_acc = []
    epoch_cm = []

    t_begin = time.time()

    for epoch in range(train_config.EPOCHS):
        train_loss, train_acc, val_loss, val_accuracy, conf_matrix = train_and_validate(
            DEVICE, train_config, data_config, model, optimizer, train_loader, valid_loader, epoch + 1, class_weights
        )

        # Log de métricas
        epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = log_metrics(
            train_loss, train_acc, val_loss, val_accuracy, conf_matrix,
            epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc, epoch_cm
        )

        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        new_weights, best_loss = save_best_model(model, val_loss, best_loss, ckpt_dir, model_name)
        if new_weights:
            best_weights = new_weights

        print(f"{'='*72}\n")

    print(f"Total time: {(time.time() - t_begin):.2f}s, Best Loss: {best_loss:.3f}")

    # Cargar los mejores pesos
    model.load_state_dict(best_weights)
    final_conf_matrix = epoch_cm[-1]

    save_model_weights(model_dir, best_weights, model_name)

    # Graficar curvas de loss y accuracy
    plot_training_curves(epoch_train_loss, epoch_test_loss, epoch_train_acc, epoch_test_acc)

    # Graficar la matriz de confusión
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    plot_confusion_matrix(final_conf_matrix, class_names, title="Confusion Matrix")

    # Mostrar ejemplos cualitativos de clasificaciones
    show_qualitative_examples(
        model, valid_loader, DEVICE, class_names, n_examples=10
    )

    history = dict(
        model=model,
        train_loss=epoch_train_loss,
        train_acc=epoch_train_acc,
        valid_loss=epoch_test_loss,
        valid_acc=epoch_test_acc,
        train_config=train_config,
        data_config=data_config,
        confusion_matrix=final_conf_matrix,
    )

    return history
