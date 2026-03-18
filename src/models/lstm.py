# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiLabelLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, device: torch.device = DEVICE):
        super(MultiLabelLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

    def build_model(self, num_features, num_classes):
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.fc(last_time_step)

    def fit(self, X: np.ndarray, y: np.ndarray, 
            min_epochs: int = 20, max_epochs: int = 100, batch_size: int = 1000, 
            patience: int = 5):
        num_classes = y.shape[-1]
        num_features = X.shape[-1]
        self.build_model(num_features, num_classes)
        optimizer = optim.Adam(self.parameters())
        criterion = nn.BCEWithLogitsLoss()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        best_train_loss = float('inf')
        epochs_without_improvement = 0
        
        self.train()
        for epoch in range(max_epochs):
            for i in range(0, len(X_tensor), batch_size):
                optimizer.zero_grad()
                outputs = self.forward(X_tensor[i:i + batch_size])
                loss = criterion(outputs, y_tensor[i:i + batch_size])
                loss.backward()
                optimizer.step()
            train_loss = loss.item()
            print(f'Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss}')

            # Early stopping logic
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch > min_epochs and epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    def predict(self, X: np.ndarray, threshold: float = 0.5, batch_size: int = 1024) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

            all_predictions = []
            for i in range(0, len(X_tensor), batch_size):
                outputs = self.forward(X_tensor[i:i + batch_size])
                batch_predictions = torch.sigmoid(outputs).cpu().numpy() > threshold
                all_predictions.append(batch_predictions)

            return np.vstack(all_predictions).astype(int)
        
class Seq2SeqLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, device: torch.device = DEVICE):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

    def build_model(self, num_features, num_classes):
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)  # Using lstm_out directly for all time steps

    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, batch_size: int = 1000, 
            patience: int = 5):
        num_classes = y.shape[-1]
        num_features = X.shape[-1]
        self.build_model(num_features, num_classes)
        optimizer = optim.Adam(self.parameters())
        criterion = nn.BCEWithLogitsLoss()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        best_train_loss = float('inf')
        epochs_without_improvement = 0

        self.train()
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                optimizer.zero_grad()
                outputs = self.forward(X_tensor[i:i + batch_size])
                loss = criterion(outputs.view(-1, num_classes), y_tensor[i:i + batch_size].view(-1, num_classes))
                loss.backward()
                optimizer.step()
            train_loss = loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}')

            # Early stopping logic
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break


    def predict(self, X: np.ndarray, threshold: float = 0.5, batch_size: int = 1024) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            all_predictions = []
            for i in range(0, len(X_tensor), batch_size):
                outputs = self.forward(X_tensor[i:i + batch_size])
                batch_predictions = torch.sigmoid(outputs).cpu().numpy() > threshold
                all_predictions.append(batch_predictions)
            return np.vstack(all_predictions).astype(int)