# Ensure PyTorch and matplotlib compatibility for proper MC Dropout visualization
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# Adjust input to one-hot and change data distribution
input_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # One-hot encoding
output_data = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

# Collect data pairs as a list of tuples
data_pairs = []

for i in range(len(input_data)):
    for _ in range(2000):  # Increase samples to balance the dataset
        input_sample = input_data[i]
        if i == 0:  # For the first input [1, 0, 0, 0]
            output_sample = output_data[np.random.choice(4, p=[0.5, 0.5/3, 0.5/3, 0.5/3])]
        elif i == 1:  # For the second input [0, 1, 0, 0]
            output_sample = output_data[np.random.choice(4, p=[0.5/3, 0.5, 0.5/3, 0.5/3])]
        elif i == 2:  # For the third input [0, 0, 1, 0]
            output_sample = output_data[np.random.choice(4, p=[0.5/3, 0.5/3, 0.5, 0.5/3])]
        elif i == 3:  # For the fourth input [0, 0, 0, 1]
            output_sample = output_data[np.random.choice(4, p=[0.5/3, 0.5/3, 0.5/3, 0.5])]
        data_pairs.append((input_sample, output_sample))

# Convert data pairs to NumPy arrays
X_train = np.array([pair[0] for pair in data_pairs])
y_train = np.array([pair[1] for pair in data_pairs])


# Define a simple neural network with dropout
class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_rate):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# Model parameters
input_dim = 4
output_dim = 2
hidden_dim = 512
dropout_rate = 0.95
model = SimpleNet(input_dim, output_dim, hidden_dim, dropout_rate)

# Training settings
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 500

# Convert data to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

# Training loop
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Perform MC Dropout sampling
T = 300  # Number of MC samples
X_test = torch.FloatTensor(input_data)
mc_samples = np.zeros((len(input_data), T, output_dim))

# Collect samples (in training mode to keep dropout active)
model.train()  # Important to keep dropout active during sampling
for t in range(T):
    with torch.no_grad():
        mc_samples[:, t, :] = model(X_test).numpy()

# Plot separate figures for each input and a combined figure
colors = ['red', 'blue', 'green', 'purple']
input_labels = ["Input [1, 0, 0, 0]", "Input [0, 1, 0, 0]", "Input [0, 0, 1, 0]", "Input [0, 0, 0, 1]"]

# Separate plots for each input
for i, input_point in enumerate(input_data):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        mc_samples[i, :, 0], mc_samples[i, :, 1],
        color=colors[i], alpha=0.3, label=input_labels[i]
    )
    plt.title(f"MC Dropout Sampling Results - {input_labels[i]}")
    plt.xlabel("Output Dimension 1")
    plt.ylabel("Output Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Combined plot for all inputs
plt.figure(figsize=(10, 8))
for i, input_point in enumerate(input_data):
    plt.scatter(
        mc_samples[i, :, 0], mc_samples[i, :, 1],
        color=colors[i], alpha=0.3, label=input_labels[i]
    )
plt.title("MC Dropout Sampling Results - All Inputs Combined")
plt.xlabel("Output Dimension 1")
plt.ylabel("Output Dimension 2")
plt.legend()
plt.grid(True)
plt.show()
