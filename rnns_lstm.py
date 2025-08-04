import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Generate sine wave data
x = np.linspace(0, 100, 1000)
data = np.sin(x)

# Prepare sequences
seq_length = 50

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

X, y = create_sequences(data, seq_length)
X = X.reshape(-1, seq_length, 1)
y = y.reshape(-1, 1)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

class SineDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SineDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        out = self.linear(out[:, -1, :])  # take the last output
        return out

model = LSTMModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
for epoch in range(num_epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Predict next value using a sequence
model.eval()
with torch.no_grad():
    input_seq = torch.tensor(data[-seq_length:].reshape(1, seq_length, 1), dtype=torch.float32)
    predicted = model(input_seq)
    print("Predicted next value:", predicted.item())

