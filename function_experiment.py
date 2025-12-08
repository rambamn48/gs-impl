import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- IMPORTS ---
from classes_collection import GSKAN, WavKAN, GeneralMLP 

# Import Standard KAN (efficient-kan)
try:
    from efficient_kan import KAN as StandardKAN
    HAS_STD_KAN = True
except ImportError:
    HAS_STD_KAN = False
    print("Info: 'efficient-kan' saknas. Kör 'pip install efficient-kan'")

# --- CONFIGURATION ---
SEEDS = [0,1,2]       
EPOCHS = 150            # More epochs needed for high frequency
BATCH_SIZE = 128        # Larger batch for functions yields more stable gradients
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")



def get_function_data(seed, n_samples=5000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Input [-1, 1]
    X = 2 * torch.rand(n_samples, 2) - 1 
    
    # Function: Multiplication (Interaction)
    # y = sin(3*pi*x) * cos(3*pi*y)
    y = torch.sin(3 * torch.pi * X[:, 0]) * torch.cos(3 * torch.pi * X[:, 1])
    
    # Noise (Floor is 1e-4)
    y += 0.01 * torch.randn(n_samples)
    y = y.unsqueeze(1)
    
    split = int(0.8 * n_samples)
    train_ds = TensorDataset(X[:split], y[:split])
    test_ds = TensorDataset(X[split:], y[split:])
    
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True), \
           DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False), \
           X.shape[1]

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- TRAINING ---
def train_one_seed(model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Slightly higher LR for functions
    criterion = nn.MSELoss()
    best_loss = float('inf')
    
    # LR Scheduler helps refine at the end
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
            
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                total_loss += criterion(model(x), y).item()
        
        avg_loss = total_loss / len(test_loader)
        if avg_loss < best_loss: best_loss = avg_loss
            
    return best_loss

# --- MAIN ---

if __name__ == "__main__":
    input_dim = 2 # X and Y coordinates
    
    # Define weight classes
    
    experiments = {
        # --- CLASS 1: NANO (~200 params) ---
        "Nano (~200 params)": {
            # GS-KAN: Standard configuration
            "GS-KAN":  {"struct": [input_dim, 10, 9, 1], "knots": 20},   
            
            # Wav-KAN: Slightly narrower
            "Wav-KAN": {"struct": [input_dim, 7, 7, 1]},   
            
            # MLP: Slightly wider
            "MLP":     {"struct": [input_dim, 12, 12, 1]},
            
            # Std-KAN: Must be very small
            "Std-KAN": {"struct": [input_dim, 5, 1], "grid": 8} 
         },
        
        # --- CLASS 2: MICRO (~500 params) ---
        "Micro (~500 params)": {
 
            "GS-KAN":  {"struct": [input_dim, 16, 16, 1], "knots": 50},   
            
  
            "Wav-KAN": {"struct": [input_dim, 12, 11, 1]},   
            

            "MLP":     {"struct": [input_dim, 20, 20, 1]}, 

            
            "Std-KAN": {"struct": [input_dim, 5,5,1], "grid": 7} 
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
            
            for seed in SEEDS:
                set_seed(seed)
                tl, testl, _ = get_function_data(seed, n_samples=4096) 
                
                s = config["struct"]
                
                # --- CREATE MODEL ---
                if name == "GS-KAN":
                    k = config.get("knots", 20) 
                    # use_input_norm=False is good here (data is already -1 to 1)
                    model = GSKAN(s, degree=3, num_knots=k, grid_max=1, use_input_norm=False).to(DEVICE)
                elif name == "MLP":
                    model = GeneralMLP(s,activation=nn.SiLU).to(DEVICE)
                elif name == "Wav-KAN":
                    model = WavKAN(s).to(DEVICE)
                elif name == "Std-KAN":
                    g = config.get("grid", 5)
                    model = StandardKAN(s, grid_size=g, spline_order=3).to(DEVICE)
                
                final_params = count_params(model)
                loss = train_one_seed(model, tl, testl)
                losses.append(loss)

            mean = np.mean(losses)
            std = np.std(losses)
            
            # Scientific notation for small numbers
            print(f"{name:<10} | {final_params:<10} | {mean:.2e} ± {std:.2e}")