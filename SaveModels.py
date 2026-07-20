import os
import copy
import torch

def save_best_model(model, val_loss, best_loss, ckpt_dir, model_name):
    """
    Guarda el modelo si la pérdida de validación mejora.
    
    Esta función compara la pérdida de validación actual con la mejor pérdida registrada hasta ahora.
    Si la nueva pérdida es menor, se guarda el modelo en el directorio de puntos de control (checkpoint).
    Se guarda el modelo completo con sus pesos actuales.

    Parámetros:
    - model (torch.nn.Module): El modelo que se está entrenando.
    - val_loss (float): La pérdida de validación en la iteración actual.
    - best_loss (float): La mejor pérdida de validación registrada hasta ahora.
    - ckpt_dir (str): El directorio donde se guardará el modelo.
    - model_name (str): El nombre del archivo para guardar el modelo.

    Retorna:
    - best_weights (dict): El estado de los pesos del modelo si se guarda el mejor modelo. 
    - best_loss (float): El valor actualizado de la mejor pérdida.

    Si la pérdida de validación actual es menor que la mejor pérdida, guarda los pesos del modelo
    y actualiza el valor de `best_loss`.
    """
    if val_loss < best_loss:
        best_loss = val_loss
        print(f"\nModel Improved... Saving Model ... ", end="")
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), os.path.join(ckpt_dir, model_name))
        print("Done.\n")
        return best_weights, best_loss
    return None, best_loss



def save_model_weights(model_dir, best_weights, model_name):
    """
    Guarda los pesos del modelo.
    Si se proporcionan los mejores pesos, también guarda el mejor modelo.
    
    Esta función guarda los pesos del modelo en el directorio proporcionado. Si se proporcionan
    los mejores pesos (`best_weights`), guarda esos pesos con un nombre distinto para hacer
    referencia al modelo con la mejor precisión hasta el momento.

    Parámetros:
    - model (torch.nn.Module): El modelo que se está entrenando.
    - model_dir (str): El directorio donde se guardarán los pesos del modelo.
    - best_weights (dict): Los pesos del modelo que se guardarán si son los mejores.
    - model_name (str): El nombre del modelo para identificar el archivo guardado.

    Retorna:
    - None: No retorna ningún valor explícito, solo guarda los archivos.

    La función verifica si `best_weights` no es `None`, y si es el caso, guarda esos pesos
    como el modelo "mejor" en un archivo `.pth` en el directorio especificado.
    """
    # Crear subdirectorios si no existen
    os.makedirs(model_dir, exist_ok=True)

    # Guardar pesos del mejor modelo
    if best_weights is not None:
        best_model_path = os.path.join(model_dir, f"{model_name}_best_weights.pth")
        torch.save(best_weights, best_model_path)
        print(f"Best model weights saved at: {best_model_path}")

    return


def get_model_preloaded(model, model_dir, model_name, device):
    """
    Carga pesos preentrenados en un modelo de PyTorch y lo mueve al dispositivo especificado.

    Args:
        model (torch.nn.Module): El modelo de PyTorch al que se le cargarán los pesos.
        model_dir (str): La ruta del directorio donde se encuentran los archivos de pesos del modelo.
        model_name (str): El nombre del archivo que contiene los pesos del modelo.
        device (str): El dispositivo donde se ejecutará el modelo. Ejemplo: 'cpu' o 'cuda:0'.

    Returns:
        torch.nn.Module: El modelo con los pesos preentrenados cargados y movido al dispositivo especificado.

    Example:
        >>> import torch
        >>> from torchvision.models import resnet50
        >>> model = resnet50()
        >>> model = get_model_preloaded(model, "./weights", "resnet50_weights.pth", "cuda:0")
    """
    device = torch.device(device) 
    model.to(device)  
    model_weights = model_name  
    path = os.path.join(model_dir, model_weights)  
    model.load_state_dict(torch.load(path, map_location=device))  
    return model

