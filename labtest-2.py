# Name- Nipun Bharadwaj
# Roll Number-22AG30027
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Experiment 1
import torch
import numpy as np
import random
seed_value = 2022
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
device = torch.device("cpu")

from google.colab import drive
drive.mount('/content/gdrive')
!unzip gdrive/MyDrive/lab_test_2_dataset.zip
# Experiment 2
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='/content/lab_test_2_dataset', transform=transform)

train_indices, temp_indices = train_test_split(list(range(len(dataset))), test_size=0.3, random_state=seed_value)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=seed_value)

train_loader = DataLoader(Subset(dataset, train_indices), batch_size=64, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=64)
test_loader = DataLoader(Subset(dataset, test_indices), batch_size=64)

print("Overall dataset size:", len(dataset))
print("Training dataset size:", len(train_loader.dataset))
print("Validation dataset size:", len(val_loader.dataset))
print("Testing dataset size:", len(test_loader.dataset))
# Experiment 3
import torch
import torch.nn as nn

class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)


        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)


        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)


        x = x.view(-1, 32 * 8 * 8)


        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
# Experiment 4
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt




model = CNNRegression()


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


model.to(device)


num_epochs = 25
train_losses = []
val_losses = []

for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)


    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)


    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"model_checkpoint_epoch_{epoch+1}.pt")


plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.show()
# Experiment 5
from sklearn.metrics import mean_squared_error

model.eval()
predicted_ages = []
true_ages = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.float().to(device)
        outputs = model(inputs)
        predicted_ages.extend(outputs.cpu().numpy())
        true_ages.extend(labels.cpu().numpy())

mse = mean_squared_error(true_ages, predicted_ages)
print(f"Mean Squared Error on Testing Dataset: {mse:.4f}")

plt.figure(figsize=(8, 8))
plt.scatter(true_ages, predicted_ages, alpha=0.5)
plt.title('Predicted vs. True Ages')
plt.xlabel('True Ages')
plt.ylabel('Predicted Ages')
plt.grid(True)
plt.show()

