import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
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
from models import GSKAN, WavKAN, GeneralMLP

try:
    from efficient_kan import KAN as StandardKAN
    HAS_STD_KAN = True
except ImportError:
    HAS_STD_KAN = False
    print("Info: 'efficient-kan' missing. Standard KAN will be skipped.")

# --- 2. CONFIGURATION ---
SEEDS = [i for i in range(10)]       
EPOCHS = 100            
BATCH_SIZE = 128         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# --- 3. Load data once globally ---
raw_data = fetch_openml(data_id=189, as_frame=False, parser='auto')
GLOBAL_X = np.array(raw_data.data, dtype=np.float32)
GLOBAL_y = np.array(raw_data.target, dtype=np.float32)


def get_dataloaders(seed, X, y):
   
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=seed)
    
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def train_one_seed(model, train_loader, val_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict()) 
    
    for epoch in range(EPOCHS):
        #Training
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        #Validation
        model.eval()
        val_loss = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                batch_loss = criterion(pred, y)
              
                val_loss += batch_loss.item() * x.size(0)
                val_total += x.size(0)
        
        avg_val_loss = val_loss / val_total
        
       
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            
    # TESTNING 
    model.load_state_dict(best_model_state)
    model.eval()
    test_loss = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            batch_loss = criterion(pred, y)
            test_loss += batch_loss.item() * x.size(0)
            test_total += x.size(0)
            
    return test_loss / test_total

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

    print(f"\n{'Model':<10} | {'Params':<10} | {'Best Test MSE (Mean ± Std)':<30}")
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
                train_l, val_l, test_l = get_dataloaders(seed, GLOBAL_X, GLOBAL_y)
                
                if name == "GS-KAN":
                    k = config.get("knots",20)
                    model = GSKAN(s, degree=3, num_knots=k, grid_max=3, use_input_norm=False).to(DEVICE)
                elif name == "MLP":
                    model = GeneralMLP(s).to(DEVICE)
                elif name == "Wav-KAN":
                    model = WavKAN(s).to(DEVICE)
                elif name == "Std-KAN":
                    g = config.get("grid", 5)
                    model = StandardKAN(s, grid_size=g, spline_order=3,grid_range=[-3,3]).to(DEVICE)
                
                final_params = count_params(model)
                loss = train_one_seed(model, train_l, val_l, test_l)
                losses.append(loss)
                
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            mean = np.mean(losses)
            std = np.std(losses)
            print(f"{name:<10} | {final_params:<10} | {mean:.4e} ± {std:.4e}")