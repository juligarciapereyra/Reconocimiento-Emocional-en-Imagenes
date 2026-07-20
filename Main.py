from Seed import set_seed
from Train import train_model
from Evaluate import evaluate_model
from GradCam import print_gradcam_results
from typing import Union
from collections import Counter
from PIL import Image
from DataSet import *
import torch

def main(
    emotion_mapping,
    DEVICE: torch.device,
    model: torch.nn.Module,
    optimizer: Union[torch.optim.SGD, torch.optim.Adam],
    ckpt_dir: str,
    data_config,
    train_config,
    path_dataset,
    model_dir,
    model_name,
    seed: int=42, 
) -> dict:

    set_seed(seed)
    print(f"Using seed: {seed}")

    _, train_labels = load_data(os.path.join(path_dataset, "train"), emotion_mapping)

    X_train, X_test, Y_train, Y_test = get_X_Y_arrays(path_dataset, emotion_mapping,channels=3)
    X_train_pil = [Image.fromarray(img.astype('uint8')) for img in X_train]
    X_test_pil = [Image.fromarray(img.astype('uint8')) for img in X_test]

    preprocess = image_preprocess_transforms()

    training_dataset, testing_dataset = create_datasets(X_train_pil, Y_train, X_test_pil, Y_test, preprocess)

    train_data, valid_data, _, _ = split_train_valid(training_dataset, train_labels, valid_split=0.2)

    class_counts = Counter(train_labels)
    total_samples = len(train_labels)

    weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    train_loader, valid_loader, testing_loader = create_data_loaders(train_data, valid_data, testing_dataset, data_config)

    history = train_model(DEVICE, model, optimizer, train_loader, valid_loader, train_config, data_config, ckpt_dir, model_dir, model_name, weights)

    accuracy = evaluate_model(model, testing_loader, DEVICE, emotion_mapping)

    print(f"Accuracy on the test set: {accuracy:.4f}")

    print_gradcam_results(model, model_name, DEVICE, testing_dataset, data_config) 

    return history