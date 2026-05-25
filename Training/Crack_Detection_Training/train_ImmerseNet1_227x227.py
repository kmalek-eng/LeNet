# IMPORT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import os
import cv2
import random
from torch.utils.data import Dataset, DataLoader


# CLASSES AND FUNCTIONS
# Define the ImmerseNet1 model
class ImmerseNet1(nn.Module):
    def __init__(self):
        super(ImmerseNet1, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Convolutional layers
        # with torch.no_grad:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        # Flatten the output of the convolutional layers
        x = x.view(-1, 64 * 10 * 10)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Output layer
        return x


# Loading images from a folder
def load_images_from_folder(folder):
    imgs = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            imgs.append(img)
    return imgs


# Creating a dataset class
class CustomImageDataset(Dataset):
    def __init__(self, images, labs):
        self.images = images
        self.labels = labs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


# Function to show an image
def show_image(img):
    img = img.squeeze()
    plt.imshow(img, cmap='gray')
    plt.show()


# Function to format the weights to 4 significant digits
def format_weights(tensor, digits=4):
    tensor = tensor.data.cpu().numpy()
    formatted_tensor = np.round(tensor, digits)
    return formatted_tensor


# DATA
# Load and preprocess positive images
positive_images = load_images_from_folder(
    r'C:\!\Dessertation\Technical\AI\after_last\ImmerseNet\transfer\data\training\Positive')
pos_greys = [0.299/255 * positive_images[i][:, :, 0] + 0.587/255 * positive_images[i][:, :, 1] + 0.114/255 * positive_images[i][:, :, 2]
             for i in range(len(positive_images))]
pos_greys_tensors = [torch.reshape(torch.tensor(image), (1, 227, 227)) for image in pos_greys]
pos_labels = torch.ones(len(pos_greys_tensors))
# Load and preprocess negative images
negative_images = load_images_from_folder(
    r'C:\!\Dessertation\Technical\AI\after_last\ImmerseNet\transfer\data\training\Negative')
neg_greys = [0.299/255 * negative_images[i][:, :, 0] + 0.587/255 * negative_images[i][:, :, 1] + 0.114/255 * negative_images[i][:, :, 2]
             for i in range(len(negative_images))]
neg_greys_tensors = [torch.reshape(torch.tensor(image), (1, 227, 227)) for image in neg_greys]
neg_labels = torch.zeros(len(neg_greys_tensors))
# Combine positive and negative images and labels into a single list of tuples
combined_data = list(zip(pos_greys_tensors + neg_greys_tensors, pos_labels.tolist() + neg_labels.tolist()))
# Shuffle the combined list
random.shuffle(combined_data)
# Separate images and labels back into their respective lists
shuffled_images, shuffled_labels = zip(*combined_data)
# Hyperparameters
batch_size = 16
learning_rate = 0.002
num_epochs = 10
# Pooling layer parameters
pooling_kernel_size = 2
pooling_stride = 2
# Convert lists back to tensors
shuffled_images = list(shuffled_images)
shuffled_labels = torch.tensor(shuffled_labels)
# Create the dataset
dataset = CustomImageDataset(shuffled_images, shuffled_labels)
# Split the dataset into training and test sets
train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
print(test_size)
eval_size = len(dataset) - train_size - test_size
train_dataset, test_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, eval_size])
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)


# TRAINING
# Write hyperparameters to a file
with open('hyperparameters.txt', 'w') as f:
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Learning rate: {learning_rate}\n')
    f.write(f'Number of epochs: {num_epochs}\n')
    f.write(f'Pooling kernel size: {pooling_kernel_size}\n')
    f.write(f'Pooling stride: {pooling_stride}\n')
# Create the model
model = ImmerseNet1()
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Calculate total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of learnable parameters: {total_params}')
# Write total number of parameters to a file
with open('total_parameters.txt', 'w') as f:
    f.write(f'Total number of learnable parameters: {total_params}\n')

# Train the model
best_loss = np.inf
for epoch in range(num_epochs):
    running_loss = 0.0
    total_loss= 0.0
    model.train()
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.long()
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # print(type(outputs[0].numpy()))
        # Calculate the loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % 400 == 0:
            print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    # Print total loss for the epoch
    print('[Epoch %d] total loss: %.3f' % (epoch + 1, total_loss / len(train_loader)))
    # Evaluation
    ep_val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for j, batch in enumerate(eval_loader):
            inputs, labels = batch
            inputs = inputs.float()
            labels = labels.long()

            # Forward pass
            outputs = model(inputs)

            # Calculate validation loss
            val_loss = criterion(outputs, labels)
            ep_val_loss += val_loss.item()

    avg_ep_val_loss = ep_val_loss / len(eval_loader)
    print('[Epoch %d] validation loss: %.3f' % (epoch + 1, avg_ep_val_loss))

    # Save the model weights if validation loss is the best
    if avg_ep_val_loss < best_loss:
        best_loss = avg_ep_val_loss
        torch.save(model.state_dict(), 'ImmerseNet1_weights.pth')
        print('Model saved with validation loss: %.3f' % best_loss)


# TRANSFORMATION OF WEIGHTS
# Collect the model weights with 4 significant digits
weights_dict = {}
for name, param in model.named_parameters():
    formatted_param = format_weights(param)
    weights_dict[name] = formatted_param
# Create the C# script to save the weights and biases as arrays
csharp_code = """
using System;

public static class ImmerseNet1Weights
{
"""
for name, param in weights_dict.items():
    if 'conv1.weight' in name:
        for i, sub_param in enumerate(param):
            flat_param = sub_param.flatten()
            array_str = ', '.join(map(str, flat_param))
            csharp_code += f"""
    double[] {name.replace('.', '_')}_{i} = new double[]
    {{
        {array_str}
    }};
"""
    elif 'conv2.weight' in name:
        for i, sub_param in enumerate(param):
            flat_param = sub_param.flatten()
            array_str = ', '.join(map(str, flat_param))
            csharp_code += f"""
    double[] {name.replace('.', '_')}_{i} = new double[]
    {{
        {array_str}
    }};
"""
    elif 'fc1.weight' in name:
        for i, sub_param in enumerate(param):
            flat_param = sub_param.flatten()
            array_str = ', '.join(map(str, flat_param))
            csharp_code += f"""
    double[] {name.replace('.', '_')}_{i} = new double[]
    {{
        {array_str}
    }};
"""
    elif 'fc2.weight' in name:
        for i, sub_param in enumerate(param):
            flat_param = sub_param.flatten()
            array_str = ', '.join(map(str, flat_param))
            csharp_code += f"""
    double[] {name.replace('.', '_')}_{i} = new double[]
    {{
        {array_str}
    }};
"""
    elif 'fc3.weight' in name:
        for i, sub_param in enumerate(param):
            flat_param = sub_param.flatten()
            array_str = ', '.join(map(str, flat_param))
            csharp_code += f"""
    double[] {name.replace('.', '_')}_{i} = new double[]
    {{
        {array_str}
    }};
"""
    else:
        flat_param = param.flatten()
        array_str = ', '.join(map(str, flat_param))
        csharp_code += f"""
    double[] {name.replace('.', '_')} = new double[]
    {{
        {array_str}
    }};
"""

csharp_code += """
}
"""
# Save the C# script to a file
with open('ImmerseNet1Weights.cs', 'w') as f:
    f.write(csharp_code)


# TESTING
# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.long()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the %d test images: %d %%' % (test_size, 100 * correct / total))
# Collect predictions and true labels
all_labels = []
all_predictions = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.float()
        labels = labels.long()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels[4000:5000], all_predictions[4000:5000])

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.set(font='Garamond', font_scale=2)  # Set font to Garamond and increase font scale
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 24})
plt.xlabel('Predicted', fontsize=26, fontname='Garamond')
plt.ylabel('True', fontsize=26, fontname='Garamond')
plt.title('Confusion Matrix', fontsize=26, fontname='Garamond')
plt.show()

# Assuming num_classes is the number of classes
num_classes = 2

# Binarize the labels for ROC curve
all_labels = label_binarize(all_labels, classes=[i for i in range(num_classes)])
all_predictions_prob = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.float()
        labels = labels.long()
        outputs = model(images)
        all_predictions_prob.extend(outputs.cpu().numpy())

all_predictions_prob = np.array(all_predictions_prob)
