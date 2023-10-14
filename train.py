import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from Network import Network as network

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")   # Use CPU

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default = 275)
parser.add_argument("-b", "--batch", type=int, help="Batch size", default = 1)

# Parse the arguments
args = parser.parse_args()
print(device)

num_classes = 50
batch_size = args.batch
num_epochs = args.epochs

print("classes: ", num_classes, "batch", batch_size)

# Define the dimensions of your images
image_width, image_height = 56, 56

# Define the total number of data points and images per data point
total_data_points = 2944
images_per_data_point = 36

preprocessed_data = np.load(f'preprocessed_train.npy')
# Convert the NumPy array to PyTorch tensors
data = torch.from_numpy(preprocessed_data).float()

class JigsawDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        puzzle = self.data[idx]
        label = self.labels[idx]
        return puzzle, label

# Load the labels
labels = np.loadtxt(f'train/label_train.txt')
labels = torch.from_numpy(labels).long()

# Define the dataset and dataloader
dataset = JigsawDataset(data, labels)

# Ensure shuffle = False when evaluating on validation and test
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Create the model
model = network(num_classes)
# Move the model to the device (CPU or GPU)
model.to(device)
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Define the criterion
criterion = nn.CrossEntropyLoss()

# Define the train_model function
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    # Set the model to training mode
    model.train()
    # Loop over the epochs
    for epoch in range(num_epochs):
        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_acc = 0.0
        # Loop over the batches
        for i, (puzzles, labels) in enumerate(train_loader):
            # Move the puzzles and labels to the device
            puzzles = puzzles.to(device)
            labels = labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(puzzles)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Compute the accuracy
            _, preds = torch.max(outputs, 1)
            acc = torch.sum(preds == labels).item() / batch_size
            # Update the running loss and accuracy
            running_loss += loss.item()
            running_acc += acc

        # Print the average loss and accuracy for each epoch
        print(f'Epoch {epoch + 1}, Average Loss: {running_loss / len(train_loader):.4f}, Average Accuracy: {(running_acc / len(train_loader))*100:.4f}%')
    # Return the trained model
    return model

# Call the train_model function
model = train_model(model, train_loader, optimizer, criterion, num_epochs)


# Define the JigsawValidationDataset class
class JigsawValidationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        puzzle = self.data[idx]
        return puzzle

# Define the evaluate_model function
def evaluate_model(model, val_loader, filename):
    # Set the model to evaluation mode
    model.eval()
    # Initialize an empty list to store the predictions
    predictions = []
    # Loop over the batches
    for puzzles in val_loader:
        # Move the puzzles to the device
        puzzles = puzzles.to(device)
        # Forward pass
        outputs = model(puzzles)
        # Get the predicted shuffle type
        _, preds = torch.max(outputs, 1)
        # Append the predictions to the list
        predictions.extend(preds.tolist())
    # Save the predictions to a txt file
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(str(pred) + '\n')

# Load the validation data from the preprocessed_validation.npy file
validation_data = np.load(f'preprocessed_validation.npy')
# Convert the NumPy array to PyTorch tensors
validation_data = torch.from_numpy(validation_data).float()

# Create a validation dataset and dataloader object
validation_dataset = JigsawValidationDataset(validation_data)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model and save the results to a txt file
evaluate_model(model, validation_loader, "1d_cnn_pre_b1_e275.txt")
