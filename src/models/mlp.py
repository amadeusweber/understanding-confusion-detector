# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes
class MultiLabelMLP(nn.Module):
    def __init__(self, hidden_layers: tuple, device: torch.device = DEVICE):
        super(MultiLabelMLP, self).__init__()
        self._hidden_layers = hidden_layers
        self._device = device

    def build_model(self, input_size: int, num_classes: int):
        layers = []
        layers.append(nn.Flatten())
        in_features = input_size
        
        for layer_size in self._hidden_layers:
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(nn.ReLU())
            in_features = layer_size
            
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(self._hidden_layers[-1], num_classes)
        self.to(self._device)

    def forward(self, x: torch.Tensor):
        return self.output_layer(self.network(x))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 1000):
        input_size = np.prod(X.shape[1:])
        num_classes = y.shape[1]
        self.build_model(input_size, num_classes)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self._device)
        optimizer = optim.Adam(self.parameters())
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                optimizer.zero_grad()
                outputs = self.forward(X_tensor[i:i + batch_size])
                loss = criterion(outputs, y_tensor[i:i + batch_size])
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, X: np.ndarray, threshold: float = 0.5, batch_size: int = 1000) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
            all_predictions = []
            for i in range(0, len(X_tensor), batch_size):
                outputs = self.forward(X_tensor[i:i + batch_size])
                batch_predictions = torch.sigmoid(outputs).cpu().numpy() > threshold
                all_predictions.append(batch_predictions)
            return np.vstack(all_predictions).astype(int)