import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleSequenceModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_actions):
        super(SimpleSequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size=features_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        # x shape: (batch, seq_len, features_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_dim)
        
        # We take the output of the last time step
        last_time_step = lstm_out[:, -1, :]
        # last_time_step shape: (batch, hidden_dim)
        
        out = self.fc(last_time_step)
        # out shape: (batch, num_actions)
        return out

def generate_simple_dataset(size=500, seq_len=10, num_features=1):
    data = np.zeros((size, seq_len, num_features))
    labels = np.zeros(size, dtype=int)  # Assuming three classes: 0 (BUY), 1 (SELL), 2 (HOLD)

    for i in range(size):
        pattern_type = np.random.choice(['increasing', 'decreasing', 'constant'])
        if pattern_type == 'increasing':
            data[i, :, 0] = np.linspace(0, 1, seq_len)
            labels[i] = 0  # 'BUY'
        elif pattern_type == 'decreasing':
            data[i, :, 0] = np.linspace(1, 0, seq_len)
            labels[i] = 1  # 'SELL'
        else:  # 'constant'
            constant_value = np.random.uniform(0, 1)
            data[i, :, 0] = constant_value
            labels[i] = 2  # 'HOLD'

    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return data_tensor, labels_tensor

# Function to train the model
def train_model(model, data, labels, epochs=50, batch_size=32, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for i in range(0, len(data), batch_size):
            inputs = data[i:i + batch_size]
            target = labels[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data)}')

if __name__ == '__main__':
    features_dim = 3
    seq_len = 10
    data_tensor, labels_tensor = generate_simple_dataset(seq_len=seq_len, num_features=features_dim)

    # Define the model
    hidden_dim = 64
    num_actions = 3
    model = SimpleSequenceModel(features_dim, hidden_dim, num_actions)

    # Train the model
    train_model(model, data_tensor, labels_tensor)