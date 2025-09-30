import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
Define neural network to predict the next value (number of births) in a time series,
based on the previous `n_steps` values at monthly resolution.
"""

# Define reproducibility function
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f'Set random seed to {seed}')

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the neural network architecture
class TimeSeriesNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout):
        super(TimeSeriesNN, self).__init__()
        layers = []
        prev_size = input_size
        for h in hidden_size:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Define base class for the model
class NeuralBase():
    def __init__(self, device = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = None
        self.scaler = None
        self.model = None
        self.y_hats = None
        self.y_ts = None

    def preprocess(self):

        # Load data
        data = pd.read_csv('data/births/births.csv', sep=',', encoding='utf-8')
        
        # Select only California data
        data = data[data['county'] == 'California']
        
        # Sort data by year and month
        data = data.sort_values(['year', 'month']).reset_index(drop=True)

        self.data = data

    def scale(self):
        X = self.data['births'].values.reshape(-1, 1)
        
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.X = X_scaled

    def create_sequences(self, X, n_steps):
        Xs, ys = [], []
        for i in range(len(X) - n_steps):
            Xs.append(X[i:i+n_steps])
            ys.append(X[i+n_steps])
        return np.array(Xs), np.array(ys)

class NeuralNetwork(TimeSeriesNN, NeuralBase):
    def __init__(self, random_state, n_steps, n_epochs, batch_size, learning_rate, hidden_size, dropout, device=None):
        NeuralBase.__init__(self, device)
        set_seed(random_state)
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.preprocess()
        self.scale()

        TimeSeriesNN.__init__(self, input_size=n_steps, output_size=1, hidden_size=hidden_size, dropout=dropout)
        self.to(self.device)

    def train(self, train_loader, model, criterion, optimizer):
        train_loss = 0.0
        model.train()
        for X_batch, y_batch in train_loader:
            
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            optimizer.zero_grad()
            y_hat = model(X_batch).squeeze()
            loss = criterion(y_hat, y_batch.squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses = train_loss

        return train_losses

    def validate(self, val_loader, model, criterion):
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_hat = model(X_batch).squeeze()
                loss = criterion(y_hat, y_batch.squeeze())
                val_loss += loss.item()
            
        val_losses = val_loss / len(val_loader)
        
        return val_losses
    
    def test(self, test_loader, model):
        model.eval()
        y_hats = []
        y_ts = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_hat = model(X_batch).squeeze().cpu().numpy()
                y_hats.extend(y_hat)
                y_ts.extend(y_batch.numpy())

        return np.array(y_ts), np.array(y_hats)
    
    def pipeline(self, epochs, batch_size, lr):
        # Create sequences
        Xs, ys = self.create_sequences(self.X, self.n_steps)

        # Split data into train, validation, and test sets (70%, 15%, 15%)
        train_size = int(len(Xs) * 0.7)
        val_size = int(len(Xs) * 0.15)

        X_train, y_train = Xs[:train_size], ys[:train_size]
        X_val, y_val = Xs[train_size:train_size+val_size], ys[train_size:train_size+val_size]
        X_test, y_test = Xs[train_size+val_size:], ys[train_size+val_size:]

        # Scale y values
        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        y_test = scaler_y.transform(y_test)
        self.scaler_y = scaler_y

        # Create DataLoaders
        train_dataset = CustomDataset(X_train.reshape(X_train.shape[0], -1), y_train)
        val_dataset = CustomDataset(X_val.reshape(X_val.shape[0], -1), y_val)
        test_dataset = CustomDataset(X_test.reshape(X_test.shape[0], -1), y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model, criterion, and optimizer
        model = TimeSeriesNN(input_size=self.n_steps, output_size=1, hidden_size=self.hidden_size, dropout=self.dropout).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = utils.LearningRateSchedulerLinear(optimizer, start_factor=1, end_factor=0.1, n_epochs=epochs)

        # Initialize loss the plot and dataframes
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        line1, = ax.plot([], [], '-', label='loss train', color='red', linewidth=0.5)
        line2, = ax.plot([], [], '-', label='loss val', color='blue', linewidth=0.5)
        # line3, = ax.plot([], [], '-', label='loss test', color='green', linewidth=0.5)
        ax.set_yscale('log')
        ax.set_xlabel(r'epoch')
        ax.set_ylabel(r'loss')
        ax.set_title(r'Loss evolution')
        ax.legend()

        df_train = pd.DataFrame(columns=('epoch', 'loss_train'))
        df_val = pd.DataFrame(columns=('epoch', 'loss_val'))

        # Training and validation loop
        for epoch in range(epochs):
            epoch_train_loss = self.train(train_loader, model, criterion, optimizer)
            epoch_val_loss = self.validate(val_loader, model, criterion)
            
            scheduler(epoch_val_loss)

            # Update the plot
            df_train.loc[len(df_train)] = [epoch, epoch_train_loss]
            df_val.loc[len(df_val)] = [epoch, epoch_val_loss]

            line1.set_data(df_train['epoch'], df_train['loss_train'])
            line2.set_data(df_val['epoch'], df_val['loss_val'])
            # line3.set_data(df_test['epoch'], df_test['loss_test'])
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)  # Pause to update the plot

            logging.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}')
        
        plt.ioff()  # Turn off interactive mode

        # Save the plot
        plt.savefig('results/nn/loss_evolution.png')
        plt.close()

        # Test the model
        y_ts, y_hats = self.test(test_loader, model)
        self.y_ts = y_ts
        self.y_hats = y_hats

    def save_results(self):
        # Create results directory if it doesn't exist
        os.makedirs('results/nn', exist_ok=True)

        # Inverse transform the predictions and true values
        y_hats_inv = self.scaler.inverse_transform(self.y_hats.reshape(-1, 1))
        y_ts_inv = self.scaler.inverse_transform(self.y_ts.reshape(-1, 1))

        # Save predictions and true values to CSV
        results_df = pd.DataFrame({
            'Predicted': y_hats_inv.flatten(),
            'True': y_ts_inv.flatten()
        })

        results_df.to_csv('results/nn/predictions.csv', index=False)

if __name__ == "__main__":

    # Define hyperparameters
    random_state = 42
    n_steps = 48
    n_epochs = 500
    batch_size = 16
    learning_rate = 0.0005
    hidden_size = [16, 32, 16, 4]
    dropout = 0.05

    # Initialize and run the neural network pipeline
    nn_model = NeuralNetwork(random_state, n_steps, n_epochs, batch_size, learning_rate, hidden_size, dropout)
    nn_model.pipeline(epochs=n_epochs, batch_size=batch_size, lr=learning_rate)
    nn_model.save_results()

    # Inverse transform the predictions and true values for plotting
    y_ts_inv = nn_model.scaler.inverse_transform(nn_model.y_ts.reshape(-1, 1)).flatten()
    y_hats_inv = nn_model.scaler.inverse_transform(nn_model.y_hats.reshape(-1, 1)).flatten()

    # Plot final results
    plt.figure(figsize=(10, 6))
    plt.plot(y_ts_inv, label='True', color='blue')
    plt.plot(y_hats_inv, label='Predicted', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('True vs Predicted Values')
    plt.legend()

    # Save final plot
    plt.savefig('results/nn/forecast.png')

    # Calculate metrics
    mse = np.mean((nn_model.y_ts - nn_model.y_hats) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(nn_model.y_ts - nn_model.y_hats))
    mape = np.mean(np.abs((nn_model.y_ts - nn_model.y_hats) / nn_model.y_ts)) * 100
    logging.info(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%')