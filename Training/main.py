import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np


# Define the LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten the output of the convolutional layers
        x = x.view(-1, 16 * 5 * 5)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Output layer
        return x

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Pooling layer parameters
pooling_kernel_size = 2
pooling_stride = 2

# Print hyperparameters
print(f'Batch size: {batch_size}')
print(f'Learning rate: {learning_rate}')
print(f'Number of epochs: {num_epochs}')
print(f'Pooling kernel size: {pooling_kernel_size}')
print(f'Pooling stride: {pooling_stride}')

# Write hyperparameters to a file
with open('hyperparameters.txt', 'w') as f:
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Learning rate: {learning_rate}\n')
    f.write(f'Number of epochs: {num_epochs}\n')
    f.write(f'Pooling kernel size: {pooling_kernel_size}\n')
    f.write(f'Pooling stride: {pooling_stride}\n')

# Define the transform to convert the data to tensor and pad to 32x32
transform = transforms.Compose([
    transforms.Pad(2),  # Add 2 pixels of padding to each side of the 28x28 image to make it 32x32
    transforms.ToTensor()
])

# Download and load the training data
train_dataset = datasets.MNIST(root="./datasets/", train=True, download=True, transform=transform)

# Download and load the test data
test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True, transform=transform)

# Create the data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Display 5 images from the training dataset
# Function to show an image
def show_image(img):
    img = img.squeeze()
    plt.imshow(img, cmap='gray')
    plt.show()


dataiter = iter(test_loader)
images, labels = dataiter.next()
for i in range(5):
    image = images[i].numpy()  # Convert tensor to NumPy array
    csharp_array = []
    for row in image:
        formatted_row = ' '.join(f'{float(value):.2f},' for value in row.flat)
        csharp_array.append(f"{{ {formatted_row} }}")
        print(formatted_row)
    # show_image(image)

    # Create the C# array as a string
    csharp_array_str = "new float[,] {\n" + ",\n".join(csharp_array) + "\n};\n"

    # Save the C# array to a file
    with open(f'image_{i}.cs', 'w') as file:
        file.write(f"float[,] image{i} = {csharp_array_str}")

# for i in range(5):
#     image = images[i].numpy()  # Convert tensor to NumPy array
#     for row in image[0]:
#         formatted_row = ' '.join(f'{float(value):.2f}' for value in row.flat)
#         print(formatted_row)
#     print()
#     show_image(images[i])

# Output message
print("C# arrays saved to files.")

# Create the model
model = LeNet5()

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
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

# Save the model weights after training
torch.save(model.state_dict(), 'lenet5_weights.pth')

# Function to format the weights to 4 significant digits
def format_weights(tensor, digits=4):
    tensor = tensor.data.cpu().numpy()
    formatted_tensor = np.round(tensor, digits)
    return formatted_tensor

# Collect the model weights with 4 significant digits
weights_dict = {}
for name, param in model.named_parameters():
    formatted_param = format_weights(param)
    weights_dict[name] = formatted_param

# Convert the weights dictionary to a pandas DataFrame and save to an Excel file
with pd.ExcelWriter('model_weights.xlsx') as writer:
    for name, param in weights_dict.items():
        # Handle different shapes of the parameters
        if param.ndim == 1:
            df = pd.DataFrame(param)
        elif param.ndim == 2:
            df = pd.DataFrame(param)
        elif param.ndim == 3:
            df = pd.DataFrame(param.reshape(param.shape[0], -1))
        elif param.ndim == 4:
            df = pd.DataFrame(param.reshape(param.shape[0], -1))
        df.to_excel(writer, sheet_name=name)


# Create the C# script to save the weights and biases as arrays
csharp_code = """
using System;

public static class LeNet5Weights
{
"""

for name, param in weights_dict.items():
    if 'conv1.weight' in name:
        for i, sub_param in enumerate(param):
            flat_param = sub_param.flatten()
            array_str = ', '.join(map(str, flat_param))
            csharp_code += f"""
    public static readonly float[] {name.replace('.', '_')}_{i} = new float[]
    {{
        {array_str}
    }};
"""
    elif 'conv2.weight' in name:
        for i, sub_param in enumerate(param):
            flat_param = sub_param.flatten()
            array_str = ', '.join(map(str, flat_param))
            csharp_code += f"""
    public static readonly float[] {name.replace('.', '_')}_{i} = new float[]
    {{
        {array_str}
    }};
"""
    elif 'fc1.weight' in name:
        for i, sub_param in enumerate(param):
            flat_param = sub_param.flatten()
            array_str = ', '.join(map(str, flat_param))
            csharp_code += f"""
    public static readonly float[] {name.replace('.', '_')}_{i} = new float[]
    {{
        {array_str}
    }};
"""
    elif 'fc2.weight' in name:
        for i, sub_param in enumerate(param):
            flat_param = sub_param.flatten()
            array_str = ', '.join(map(str, flat_param))
            csharp_code += f"""
    public static readonly float[] {name.replace('.', '_')}_{i} = new float[]
    {{
        {array_str}
    }};
"""
    elif 'fc3.weight' in name:
        for i, sub_param in enumerate(param):
            flat_param = sub_param.flatten()
            array_str = ', '.join(map(str, flat_param))
            csharp_code += f"""
    public static readonly float[] {name.replace('.', '_')}_{i} = new float[]
    {{
        {array_str}
    }};
"""
    else:
        flat_param = param.flatten()
        array_str = ', '.join(map(str, flat_param))
        csharp_code += f"""
    public static readonly float[] {name.replace('.', '_')} = new float[]
    {{
        {array_str}
    }};
"""

csharp_code += """
}
"""

# Save the C# script to a file
with open('LeNet5Weights.cs', 'w') as f:
    f.write(csharp_code)

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 1000 test images: %d %%' % (100 * correct / total))

# Collect predictions and true labels
all_labels = []
all_predictions = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels[4000:5000], all_predictions[4000:5000])

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.set(font='Garamond', font_scale=1.5)  # Set font to Garamond and increase font scale
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 20})
plt.xlabel('Predicted', fontsize=22, fontname='Garamond')
plt.ylabel('True', fontsize=22, fontname='Garamond')
plt.title('Confusion Matrix', fontsize=24, fontname='Garamond')
plt.show()


# Assuming num_classes is the number of classes
num_classes = 10

# Binarize the labels for ROC curve
all_labels = label_binarize(all_labels, classes=[i for i in range(num_classes)])
all_predictions_prob = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        all_predictions_prob.extend(outputs.cpu().numpy())

all_predictions_prob = np.array(all_predictions_prob)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_predictions_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'purple', 'brown', 'pink', 'grey'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontname='Garamond')
plt.ylabel('True Positive Rate', fontsize=14, fontname='Garamond')
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16, fontname='Garamond')
plt.legend(loc="lower right", fontsize=12)
plt.show()
