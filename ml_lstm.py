import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- 1. HDF5 DATASET FOR RAW WAVEFORMS ---
class VitalRawDataset(Dataset):
    def __init__(self, ppg, labels):
        # We store as float32 to save memory
        # Input shape for LSTM: (Batch, Sequence_Length, Features)
        # Here: (Batch, 600, 1)
        self.ppg = torch.from_numpy(ppg).float().unsqueeze(-1)
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        return self.ppg[idx], self.labels[idx]

# --- 2. THE LSTM ARCHITECTURE ---
class BPLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(BPLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        # We use the final hidden state to predict BP
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Output: [Systolic, Diastolic]
        )

    def forward(self, x):
        # r_out shape: (batch, time_step, hidden_size)
        r_out, (h_n, h_c) = self.lstm(x)
        # Use the last time step output
        out = self.fc(r_out[:, -1, :])
        return out



# --- 3. TRAINING PIPELINE ---
def train_lstm_model(hdf5_path):
    # Load ALL samples
    print("Loading data from HDF5...")
    with h5py.File(hdf5_path, 'r') as hf:
        ppg = hf['ppg'][:]
        labels = hf['label'][:]
    
    # Split into train and test (90/10 split for massive data)
    X_train, X_test, y_train, y_test = train_test_split(ppg, labels, test_size=0.1, random_state=42)

    train_ds = VitalRawDataset(X_train, y_train)
    test_ds = VitalRawDataset(X_test, y_test)
    
    # Use larger batch size for faster training on GPU
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training on {device} with {len(train_ds)} samples...")
    
    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                val_mae += torch.abs(preds - by).mean().item()
        
        avg_val_mae = val_mae / len(test_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Test MAE: {avg_val_mae:.2f}")

    torch.save(model.state_dict(), 'lstm_bp_model.pth')
    print("LSTM training complete. Model saved.")

if __name__ == "__main__":
    train_lstm_model('vitaldb_research.h5')