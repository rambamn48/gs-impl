import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 1. IMPORTS ---
from classes_collection import GSKAN, WavKAN, GeneralMLP

# Attempt to import Standard KAN
try:
    from efficient_kan import KAN as StandardKAN
    HAS_STD_KAN = True
except ImportError:
    HAS_STD_KAN = False
    print("Info: 'efficient-kan' missing. Standard KAN will be skipped.")

# --- 2. CONFIGURATION ---
SEEDS = [0, 1, 2]       
EPOCHS = 100            # Set to 1 for quick check
BATCH_SIZE = 64         
limit_samples = None    # Set to 64 for quick check, None for full run
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {DEVICE}")

# --- 3. DATA ---
def get_data(seed, n_samples=None):
    # Load data
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # --- QUICK CHECK: Limit data if n_samples is set ---
    if n_samples is not None:
        X = X[:n_samples]
        y = y[:n_samples]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize X (Important!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Seed for shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    # If running quick check (few samples), disable drop_last to avoid losing data
    drop = True if n_samples is None else False
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                             torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                            torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop, generator=g)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop)
    
    return train_loader, test_loader, X.shape[1]

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_seed(model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                total_loss += criterion(pred, y).item()
        
        avg_test_loss = total_loss / len(test_loader)
        if avg_test_loss < best_loss: best_loss = avg_test_loss
            
    return best_loss

# --- 4. MAIN ---
if __name__ == "__main__":
    input_dim = 8
    
    experiments = {
        "Nano (~200 params)": {
            "GS-KAN":  {"struct": [input_dim, 8,8, 1],"knots":20},   
            "Wav-KAN": {"struct": [input_dim, 5,5,1]},   
            "MLP":     {"struct": [input_dim, 10,10, 1]},  
            "Std-KAN": {"struct": [input_dim, 2, 1], "grid": 7} 
        },
        
        "Micro (~600 params)": {
            "GS-KAN":  {"struct": [input_dim, 18, 18, 1],"knots":20},   
            "Wav-KAN": {"struct": [input_dim, 11, 10, 1]},   
            "MLP":     {"struct": [input_dim, 20, 20, 1]},
            "Std-KAN": {"struct": [input_dim, 5, 1], "grid": 8} 
        },
        
        "Small (~2000 params)": {
            "GS-KAN":  {"struct": [input_dim, 38, 38, 1],"knots":40}, 
            "Wav-KAN": {"struct": [input_dim, 22, 21, 1]},   
            "MLP":     {"struct": [input_dim, 40, 40, 1]},
            "Std-KAN": {"struct": [input_dim, 8,8, 1], "grid": 10}
        }
    }

    print(f"{'Model':<10} | {'Params':<10} | {'Best Test MSE (Mean ± Std)':<30}")
    print("-" * 60)

    for class_name, models in experiments.items():
        print(f"\n--- {class_name} ---")
        
        for name, config in models.items():
            if name == "Std-KAN" and not HAS_STD_KAN: continue
            
            losses = []
            final_params = 0
            s = config["struct"]

            for seed in SEEDS:
                set_seed(seed)
                # Use limit_samples for quick check (or None for full run)
                tl, testl, _ = get_data(seed, n_samples=limit_samples)
                
                # --- CREATE MODEL ---
                if name == "GS-KAN":
                    k = config.get("knots",20)
                    model = GSKAN(s, degree=3, num_knots=k, grid_max=3, use_input_norm=False).to(DEVICE)
                elif name == "MLP":
                    model = GeneralMLP(s).to(DEVICE)
                elif name == "Wav-KAN":
                    model = WavKAN(s).to(DEVICE)
                elif name == "Std-KAN":
                    g = config.get("grid", 5)
                    model = StandardKAN(s, grid_size=g, spline_order=3).to(DEVICE)
                
                # Count params on the active model
                final_params = count_params(model)
                
                # Train
                loss = train_one_seed(model, tl, testl)
                losses.append(loss)

            mean = np.mean(losses)
            std = np.std(losses)
            print(f"{name:<10} | {final_params:<10} | {mean:.4f} ± {std:.4f}")