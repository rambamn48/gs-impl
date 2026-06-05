import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- IMPORTS ---
from models import GSKAN, WavKAN, GeneralMLP 

try:
    from efficient_kan import KAN as StandardKAN
    HAS_STD_KAN = True  
except ImportError:
    HAS_STD_KAN = False
    print("Info: 'efficient-kan' missing. Run '!pip install git+https://github.com/Blealtan/efficient-kan.git'")

# --- CONFIGURATION ---
SEEDS = [i for i in range(10)]
EPOCHS = 150            
BATCH_SIZE = 128        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# --- DATA GENERATION ---
def get_function_data(seed, func_id, n_samples=4096):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    if func_id in [1, 2, 3]: dim = 2
    elif func_id in [4, 5]: dim = 3
    else: raise ValueError("Invalid func_id")
    
    # Input [-1, 1]
    X = 2 * torch.rand(n_samples, dim) - 1 
    
    
    if func_id == 1:
        # F1: High-Frequency ripple (2D)
        y = torch.sin(3 * torch.pi * X[:, 0]) * torch.cos(3 * torch.pi * X[:, 1])
    elif func_id == 2:
        # F2: Bessel J0(20r) (2D)
        r = torch.sqrt(X[:, 0]**2 + X[:, 1]**2)
        y = torch.special.bessel_j0(20 * r)
    elif func_id == 3:
        # F3: x^2*y - 3*y^3*x + x*y + 2*x^3*y^2 (2D)
        x_val, y_val = X[:, 0], X[:, 1]
        y = (x_val**2 * y_val) - (3 * y_val**3 * x_val) + (x_val * y_val) + (2* x_val**3*y_val**2)
    elif func_id == 4:
        # F4: exp(x*y + cos(3*z)) * sin(2*x) (3D)
        x_val, y_val, z_val = X[:, 0], X[:, 1], X[:, 2]
        y = torch.exp(x_val * y_val + torch.cos(3 * z_val)) * torch.sin(2 * x_val)
    elif func_id == 5:
        # F5: (y+z)/(1+x^2) + (y^2-x^3)/(2-y*z+z^2+x) (3D)
        x_val, y_val, z_val = X[:, 0], X[:, 1], X[:, 2]
        term1 = (y_val + z_val) / (1.0 + x_val**2)
        term2 = (y_val**2 - x_val**3) / (2.0 - y_val * z_val + z_val**2+x_val)
        y = term1 + term2
    
    # Noise (Floor is 1e-4)
    y += 0.01 * torch.randn(n_samples)
    y = y.unsqueeze(1)
    
    
    X_np, y_np = X.numpy(), y.numpy()
    
    # SPLIT: Train 70%, Val 15%, Test 15%
    X_temp, X_test, y_temp, y_test = train_test_split(X_np, y_np, test_size=0.15, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=seed) 
    

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, dim

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- TRAINING ---
def train_one_seed(model, train_loader, val_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(EPOCHS):
        #TRAINING
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
            
        # 2. VALIDATION
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x), y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            
    # 3. TESTNING
    model.load_state_dict(best_model_state)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            test_loss += criterion(model(x), y).item()
            
    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss

# --- MAIN ---
if __name__ == "__main__":
    func_names = {
        1: "F1: 2D High-Frequency Ripple",
        2: "F2: 2D Bessel J0(20r)",
        3: "F3: 2D Polynomial",
        4: "F4: 3D Exp-Trig",
        5: "F5: 3D Rational"
    }
   
    experiments_2d = {
        "GS-KAN":  {"struct": [2, 10, 9, 1], "knots": 20},   
        "Wav-KAN": {"struct": [2, 7, 7, 1]},   
        "MLP":     {"struct": [2, 12, 12, 1]},
        "Std-KAN": {"struct": [2, 5, 1], "grid": 8} 
    }

    experiments_3d = {
        "GS-KAN":  {"struct": [3, 9, 9, 1], "knots": 20},  
        "Wav-KAN": {"struct": [3, 7, 6, 1]},               
        "MLP":     {"struct": [3, 12, 11, 1]},              
        "Std-KAN": {"struct": [3, 4, 1], "grid": 8}         
    }

    for func_id in range(1, 6):
        print(f"\n{'='*65}")
        print(f" EXPERIMENT: {func_names[func_id]}")
        print(f"{'='*65}")
        
        print(f"{'Model':<10} | {'Params':<10} | {'Best Test MSE (Mean ± Std)':<30}")
        print("-" * 60)

        
        dim = 2 if func_id in [1, 2, 3] else 3
        models = experiments_2d if dim == 2 else experiments_3d

        for name, config in models.items():
            if name == "Std-KAN" and not HAS_STD_KAN: continue
            
            losses = []
            final_params = 0 
            
            for seed in SEEDS:
                set_seed(seed)
               
                train_l, val_l, test_l, actual_dim = get_function_data(seed, func_id, n_samples=4096) 
                
                s = config["struct"].copy()
                
                # --- CREATE MODEL ---
                if name == "GS-KAN":
                    k = config.get("knots", 20) 
                    model = GSKAN(s, degree=3, num_knots=k, grid_max=2, use_input_norm=False).to(DEVICE)
                elif name == "MLP":
                    model = GeneralMLP(s, activation=nn.SiLU).to(DEVICE)
                elif name == "Wav-KAN":
                    model = WavKAN(s).to(DEVICE)
                elif name == "Std-KAN":
                    g = config.get("grid", 5)
                    model = StandardKAN(s, grid_size=g, spline_order=3).to(DEVICE)
                
                final_params = count_params(model)
                loss = train_one_seed(model, train_l, val_l, test_l)
                losses.append(loss)

            mean = np.mean(losses)
            std = np.std(losses)
            
        
            print(f"{name:<10} | {final_params:<10} | {mean:.2e} ± {std:.2e}")