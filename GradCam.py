import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import LayerGradCam
from torch.utils.data import DataLoader

def plot_gradcam_from_loader(model, target_layer, loader, device, mapping=None):
    """
    Genera heatmaps de Grad-CAM para todas las muestras de un DataLoader y los muestra en un único gráfico.

    Args:
        model (nn.Module): Modelo de PyTorch.
        target_layer (nn.Module): Capa objetivo para Grad-CAM.
        loader (DataLoader): DataLoader que proporciona las imágenes.
        device (torch.device): Dispositivo donde se moverán los tensores.
        mapping (dict): Diccionario para mapear índices a nombres de clases.

    Returns:
        None: Plotea todos los heatmaps y histogramas en un único gráfico.
    """
    if mapping is None:
        mapping = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "sad",
            5: "surprise",
            6: "neutral"
        }

    model.eval()
    grad_cam = LayerGradCam(model, target_layer)

    inputs, labels = next(iter(loader))  # Obtener el primer batch del DataLoader
    inputs, labels = inputs.to(device), labels.to(device)

    num_samples = len(inputs)
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))  # Crear subplots

    for idx in range(num_samples):
        # Seleccionar una imagen y su etiqueta
        input_img = inputs[idx].unsqueeze(0)  # Añadir dimensión de batch
        label = labels[idx].item()

        # Predicción del modelo
        output = model(input_img)
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze(0)  # Softmax para probabilidades
        pred_label = torch.argmax(probabilities).item()

        # Obtener las atribuciones de Grad-CAM
        attr = grad_cam.attribute(input_img, target=pred_label)

        # Extraer el tensor de entrada como imagen
        img = input_img[0].cpu().detach().numpy()  # Mover a CPU y eliminar batch -> (3, H, W)
        img = img.transpose(1, 2, 0)  # Convertir de (C, H, W) a (H, W, C)
        img = (img - img.min()) / (img.max() - img.min())  # Normalizar a [0, 1]

        # Reducir Grad-CAM a 2D
        attr = attr[0].cpu().detach().numpy()  # Mover a CPU y eliminar batch -> (C, H, W)
        attr = np.mean(attr, axis=0)  # Promediar sobre canales -> (H, W)

        # Redimensionar el heatmap a 224x224 (tamaño de la imagen)
        attr_resized = cv2.resize(attr, (img.shape[1], img.shape[0]))  # (W, H)

        # Plotear Grad-CAM
        axes[idx, 0].imshow(img)
        hm = axes[idx, 0].imshow(attr_resized, cmap='jet', alpha=0.2, extent=(0, img.shape[1], img.shape[0], 0))
        axes[idx, 0].set_title(f"Grad-CAM\nPredicted: {mapping[pred_label]} (True: {mapping[label]})")
        axes[idx, 0].axis('off')
        cbar = fig.colorbar(hm, ax=axes[idx, 0], fraction=0.046, pad=0.04)
        cbar.set_label("Activation Intensity")

        ## Plotear histograma de probabilidades
        axes[idx, 1].bar(range(len(probabilities)), probabilities.cpu().detach().numpy(), tick_label=list(mapping.values()))
        axes[idx, 1].set_xticklabels(list(mapping.values()), rotation=45, ha='right')
        axes[idx, 1].set_ylabel("Probability")
        axes[idx, 1].set_title("Prediction Probabilities")
        axes[idx, 1].set_ylim([0, 1])

    plt.tight_layout()
    plt.show()


def plot_avg_gradcam_by_emotion(model, target_layer, loader, device, thresholds, mapping=None):
    """
    Genera heatmaps promedio para cada emoción usando Grad-CAM basado en predicciones con diferentes umbrales.
    """
    if mapping is None:
        mapping = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "sad",
            5: "surprise",
            6: "neutral"
        }

    model.eval()
    grad_cam = LayerGradCam(model, target_layer)

    # Inicializar contenedores para los heatmaps y las imágenes
    emotion_heatmaps = {emotion: [] for emotion in mapping.values()}
    emotion_images = {emotion: [] for emotion in mapping.values()}

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Predicciones del modelo
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Probabilidades
            predictions = torch.argmax(probabilities, dim=1)

            for i in range(len(inputs)):
                # Obtener la probabilidad de la predicción
                pred_label = predictions[i].item()
                confidence = probabilities[i, pred_label].item()

                # Filtrar por umbral
                if confidence >= thresholds[0]:
                    # Calcular el heatmap Grad-CAM
                    attr = grad_cam.attribute(inputs[i].unsqueeze(0), target=pred_label)

                    # Procesar Grad-CAM a 2D
                    attr = attr[0].cpu().numpy()  # Extraer datos
                    attr = np.mean(attr, axis=0)  # Reducir canales -> (H, W)
                    attr_resized = cv2.resize(attr, (inputs[i].shape[2], inputs[i].shape[1]))  # Redimensionar

                    # Procesar imagen de entrada
                    img = inputs[i].cpu().numpy().transpose(1, 2, 0)  # Convertir a (H, W, C)
                    img = (img - img.min()) / (img.max() - img.min())  # Normalizar

                    # Guardar heatmap e imagen
                    emotion_name = mapping[pred_label]
                    emotion_heatmaps[emotion_name].append(attr_resized)
                    emotion_images[emotion_name].append(img)

    # Calcular promedios y varianzas
    fig, axes = plt.subplots(len(mapping), 2, figsize=(12, 4 * len(mapping)))  # Crear subplots

    for idx, emotion in enumerate(mapping.values()):
        if len(emotion_heatmaps[emotion]) > 0:
            avg_heatmap = np.mean(emotion_heatmaps[emotion], axis=0)
            avg_image = np.mean(emotion_images[emotion], axis=0)
            var_image = np.var(emotion_images[emotion], axis=0)
            print(emotion, len(emotion_images[emotion]))

            # Plotear heatmap promedio
            im = axes[idx, 0].imshow(avg_heatmap, cmap='Spectral_r', alpha=0.8)
            axes[idx, 0].set_title(f"{emotion} - Avg Grad-CAM")
            axes[idx, 0].axis('off')

            cbar = fig.colorbar(im, ax=axes[idx, 0], fraction=0.046, pad=0.04)
            cbar.set_label("Activation Intensity")

            # Plotear imagen promedio
            axes[idx, 1].imshow(avg_image)
            axes[idx, 1].set_title(f"{emotion} - Avg Image")
            axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

def print_gradcam_results(model, model_name, device, dataset, dataset_config, thresholds=[0.4]):
    """
    Genera visualizaciones Grad-CAM para un modelo dado.

    Args:
        model (torch.nn.Module): Modelo a evaluar.
        model_name (str): Nombre del modelo (e.g., 'ResNet', 'EfficientNet', 'VGG', 'SimpleCNN').
        device (torch.device): Dispositivo en el que se ejecutará el modelo.
        dataset (torch.utils.data.Dataset): Dataset de entrada.
        dataset_config (object): Configuración del dataset, debe incluir atributos `BATCH_SIZE` y `NUM_WORKERS`.

    Returns:
        None: Genera visualizaciones Grad-CAM.
    """
    gradcam_loader = DataLoader(dataset, batch_size=dataset_config.BATCH_SIZE, shuffle=False,num_workers=dataset_config.NUM_WORKERS)

    # Determinar la capa objetivo para Grad-CAM según el modelo
    if model_name == 'ResNet':
        target_layer = model.layer4[-1]  
    elif model_name == 'EfficientNet':
        target_layer = model.features[7] # CAMBIAR A 8 
    elif model_name == 'VGG':
        target_layer = model.features[29]   
    elif model_name == 'CustomCNN':
        target_layer = model.res_block2.conv2  
    else:
        print("Model not recognized. Supported models: ResNet, EfficientNet, VGG, CustomCNN.")
        return -1

    # Generar visualizaciones Grad-CAM
    plot_gradcam_from_loader(model, target_layer, gradcam_loader, device=device)
    plot_avg_gradcam_by_emotion(model, target_layer, gradcam_loader, device, thresholds)



