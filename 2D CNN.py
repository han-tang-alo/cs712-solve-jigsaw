import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from Layers import LRN

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")   # Use CPU

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default = 100)
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

# Define the jigsawModel class
class jigsawModel1D(nn.Module):
    def __init__(self, num_classes):
        super(jigsawModel1D, self).__init__()
        # Define the 1D convolutional layer
        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1',nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('relu1_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv.add_module('relu2_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('relu4_s1',nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('relu5_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1',nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(256*3*3, 1024))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(9*1024,4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7',nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8',nn.Linear(4096, num_classes))

    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.conv(x[i])
            z = self.fc6(z.view(B,-1))
            z = z.view([B,1,-1])
            x_list.append(z)

        x = cat(x_list,1)
        x = self.fc7(x.view(B,-1))
        x = self.classifier(x)

        return x



# Create the model
model = jigsawModel1D(num_classes)
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
evaluate_model(model, validation_loader, "1d_cnn_pre_b1_alo100.txt")
