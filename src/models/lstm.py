# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiLabelLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_classes: int, device: torch.device = DEVICE):
        super(MultiLabelLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.num_features = 18
        self.num_classes = num_classes

    def build_model(self):
        self.lstm = nn.LSTM(input_size=self.num_features, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.fc(last_time_step)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 1024):
        num_classes = y.shape[1]
        X = np.expand_dims(X, 1).reshape(X.shape[0], -1, self.num_features)
        self.build_model()
        optimizer = optim.Adam(self.parameters())
        criterion = nn.BCEWithLogitsLoss()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                optimizer.zero_grad()
                outputs = self.forward(X_tensor[i:i + batch_size])
                loss = criterion(outputs, y_tensor[i:i + batch_size])
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, X: np.ndarray, threshold: float = 0.5, batch_size: int = 1024) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X = np.expand_dims(X, 1).reshape(X.shape[0], -1, self.num_features)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

            all_predictions = []
            for i in range(0, len(X_tensor), batch_size):
                outputs = self.forward(X_tensor[i:i + batch_size])
                batch_predictions = torch.sigmoid(outputs).cpu().numpy() > threshold
                all_predictions.append(batch_predictions)

            return np.vstack(all_predictions).astype(int)