import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_processing
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
import json

# Switch to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convolutional Neural Network class methods
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers and batch normalizations
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # The input images are 100x100
        x = torch.randn(100, 100).view(-1, 1, 100, 100)
        # Variabile to ensure the correct size of in_features passed from the last conv layer to the first fc layer
        self._to_linear = None
        self.convs(x)

        # Fully connected layers and dropout to reduce the overfitment risk
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.5)

    # Feed forward x through each conv layer, apply batch norm, ReLU activation and pooling to reduce dimensions
    def convs(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    # Feed forward x through each fc layer, apply ReLU activation, dropout and return the logits
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    # Predict the classification of images
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            # Apply softmax and choose the highest probability
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            print(f"Predictions: {predictions}")
        return predictions.tolist()

# Save training dataset tensors
def save_data(X_train, y_train, X_test, y_test, class_count, filename="dataset.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((X_train, y_train, X_test, y_test, class_count), f)
    print(f"Dataset saved to {filename}")

# Load training dataset tensors
def load_data(filename="dataset.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            X_train, y_train, X_test, y_test, class_count = pickle.load(f)
        print(f"Dataset loaded from {filename}")
        return X_train, y_train, X_test, y_test, class_count
    else:
        print(f"{filename} not found. You need to generate and save the dataset first.")
        return None, None, None, None, None

# Save model's state
def save_model(model, optimizer, scheduler, epoch, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, filename)
    print(f"Model saved to {filename}")

# Load the model
def load_model(model, optimizer, scheduler, filename, device=None):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Model loaded from {filename}, resuming from epoch {epoch}")
        return model, optimizer, scheduler, epoch
    else:
        print(f"{filename} not found. Starting training from scratch.")
        return model, optimizer, scheduler, 0

# Get training and validation data if dataset.pkl is missing
def get_dataset_tensors(X_train, y_train, X_test, y_test, class_count):
    training_data, class_count = data_processing.get_dataset('train')
    validation_data, _ = data_processing.get_dataset('test')

    # Shuffle and trim the datasets
    np.random.shuffle(training_data)
    np.random.shuffle(validation_data)
    training_data = training_data[:100000]
    validation_data = validation_data[:20000]
    np.random.shuffle(training_data)
    np.random.shuffle(validation_data)

    # Separate the image tensor and the one hot encoding
    X_train = torch.tensor([i[0] for i in training_data]).float().to(device)
    X_test = torch.tensor([i[0] for i in validation_data]).float().to(device)
    y_train = torch.tensor([i[1] for i in training_data]).float().to(device)
    y_test = torch.tensor([i[1] for i in validation_data]).float().to(device)

    # Normalize the image tensor
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test

# Get training and validation dataset from dataset.pkl
def prepare_data():
    X_train, y_train, X_test, y_test, class_count = load_data()

    # Get training and validation data if they weren't saved
    if X_train is None and y_train is None and X_test is None and y_test is None and class_count is None:
        get_dataset_tensors(X_train, y_train, X_test, y_test, class_count)
        save_data(X_train, y_train, X_test, y_test, class_count)

    # Compute the weight for each class for non-biased model results (unbalanced dataset)
    class_weights = []
    total_samples = np.sum(class_count)
    num_classes = len(class_count)
    class_weights = torch.tensor([total_samples / (num_classes * count) for count in class_count]).float().to(device)
    class_weights = class_weights / class_weights.sum()

    return X_train, y_train, X_test, y_test, class_weights

# Plot the functions
def plot_info(train_losses, val_losses, train_accuracies, val_accuracies, epochs_range):
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', color='red', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Trend')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='red', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trend')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def training_loop(continue_training):
    X_train, y_train, X_test, y_test, class_weights = prepare_data()
    class_weights = class_weights.to(device)
    neuralNet = NeuralNet().to(device)

    # Optimizer and loss function
    optimizer = optim.AdamW(neuralNet.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Schedular for a good convergence, decrease learning rate by 0.7 every 2 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

    # Load checkpoint at the start of training
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Chose the checkpoint model
    neuralNet, optimizer, scheduler, start_epoch = load_model(neuralNet, optimizer, scheduler, filename=f"{model_dir}/model_checkpoint.pth", device=device)

    if continue_training:
        # Tracking lists for loss and accuracy over epochs
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        batch_size = 64
        epochs = 5

        for epoch in range(start_epoch, epochs):
            neuralNet.train()
            epoch_train_loss = 0
            correct_train = 0
            total_train = 0

            # Training loop
            for i in tqdm(range(0, len(X_train), batch_size)):
                batch_X = X_train[i:i + batch_size].view(-1, 1, 100, 100).to(device)
                batch_y = y_train[i:i + batch_size].to(device)

                # Convert one-hot to class indices if necessary
                batch_y = batch_y.argmax(dim=1) if batch_y.ndim > 1 else batch_y
                batch_y = batch_y.long()

                optimizer.zero_grad()

                # Forward pass
                outputs = neuralNet(batch_X)
                loss = loss_fn(outputs, batch_y)
                epoch_train_loss += loss.item()

                # Calculate training accuracy
                predicted_classes = torch.argmax(outputs, dim=1)
                correct_train += (predicted_classes == batch_y).sum().item()
                total_train += batch_y.size(0)

                # Backward pass
                loss.backward()
                optimizer.step()

            train_losses.append(epoch_train_loss / (len(X_train) // batch_size))
            train_accuracies.append(correct_train / total_train)

            neuralNet.eval()
            correct_val = 0
            total_val = 0
            val_loss = 0

            # Validation loop
            with torch.no_grad():
                for i in tqdm(range(0, len(X_test), batch_size)):
                    batch_X = X_test[i:i + batch_size].view(-1, 1, 100, 100).to(device)
                    batch_y = y_test[i:i + batch_size].to(device)

                    batch_y = batch_y.argmax(dim=1) if batch_y.ndim > 1 else batch_y
                    batch_y = batch_y.long()

                    outputs = neuralNet(batch_X)
                    loss = loss_fn(outputs, batch_y)  # Compute validation loss
                    val_loss += loss.item()

                    predicted_classes = torch.argmax(outputs, dim=1)
                    correct_val += (predicted_classes == batch_y).sum().item()
                    total_val += batch_y.size(0)

            # Store losses
            val_losses.append(val_loss / (len(X_test) // batch_size))
            val_accuracies.append(correct_val / total_val)

            epoch_details = f"Epoch: {epoch + 1}, Train Loss: {epoch_train_loss / (len(X_train) // batch_size):.4f}, " \
                            f"Train Accuracy: {correct_train / total_train:.4f}, " \
                            f"Val Loss: {val_loss / (len(X_test) // batch_size):.4f}, " \
                            f"Val Accuracy: {correct_val / total_val:.4f}"

            log_file = f"{model_dir}/model_{epoch + 1}.txt"

            with open(log_file, "w") as f:
                f.write(epoch_details)

            # Save the checkpoint at the end of each epoch
            save_model(neuralNet, optimizer, scheduler, epoch + 1, filename=f"{model_dir}/model_{epoch + 1}.pth")

            # Print the epoch details
            print(epoch_details)
            scheduler.step()

        # Plot the functions
        epochs_range = range(1, epochs + 1)
        plot_info(train_losses, val_losses, train_accuracies, val_accuracies, epochs_range)

    return neuralNet

# Predictions for the prediction dataset
def get_prediction(neuralNet):
    filename = "predictionset.pkl"
    predictions_json = "predictions.json"

    # Load or preprocess the prediction set
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            logos, urls = pickle.load(f)
        print(f"Prediction set loaded from {filename}")
    else:
        logos_and_urls = data_processing.get_logos()

        # Extract images and URLs, ensuring only valid images are included
        logos = [logo[0] for logo in logos_and_urls if isinstance(logo[0], np.ndarray)]
        urls = [logo[1] for logo in logos_and_urls if isinstance(logo[0], np.ndarray)]

        # Save the preprocessed dataset
        with open(filename, "wb") as f:
            pickle.dump((logos, urls), f)

    # Define labels
    LABELS = {'Accessories': 0, 'Clothes': 1, 'Cosmetic': 2, 'Electronic': 3,
              'Food': 4, 'Institution': 5, 'Leisure': 6, 'Medical': 7,
              'Necessities': 8, 'Transportation': 9}
    label_names = {v: k for k, v in LABELS.items()}

    # Dictionary to store predictions
    results = {label_name: [] for label_name in label_names.values()}

    # Process each logo one by one
    for i, logo in enumerate(logos):
        image_tensor = torch.tensor(logo, dtype=torch.float32).unsqueeze(0).to(device)
        image_tensor /= 255.0

        # Get prediction
        predicted_label = neuralNet.predict(image_tensor)

        # Ensure predicted_label is a single integer
        if isinstance(predicted_label, torch.Tensor):
            predicted_label = predicted_label.squeeze()
            predicted_label = predicted_label.item()
        # If a list, get the first element
        elif isinstance(predicted_label, list):
            predicted_label = predicted_label[0]

        # Convert index to class name and store the URL
        label_name = label_names[predicted_label]
        results[label_name].append(urls[i])

    # Save the results as a JSON file
    with open(predictions_json, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Predictions saved to {predictions_json}")

prepare_data()
# Switch to False if not willing to resume training
neuralNet = training_loop(True)
get_prediction(neuralNet)
