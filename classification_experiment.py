import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# --- 1. IMPORTS ---
from models import GSKAN, GeneralMLP, WavKAN

try:
    from efficient_kan import KAN as StandardKAN
    HAS_STD_KAN = True
except ImportError:
    HAS_STD_KAN = False
    print("Info: 'efficient-kan' missing. Run '!pip install git+https://github.com/Blealtan/efficient-kan.git'")

# --- 2. CONFIGURATION ---
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 20 
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {DEVICE}")

# --- 3. HELPER FUNCTIONS ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_data(seed, full_train_ds, test_ds):
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train_ds, [50000, 10000], generator=generator)
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if DEVICE.type == 'cuda' else {}
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    
    return train_loader, val_loader, test_loader, 784

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 4. TRAINING LOOP ---
def train_one_seed(model, train_loader, val_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    best_val_loss = float('inf')
    best_model_state = None 
    pbar = tqdm(range(EPOCHS), desc="Training", leave=False)

    for epoch in pbar:
        # 1. TRAINING
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
        
        scheduler.step()

        # 2. VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                batch_loss = criterion(output, y)
                val_loss += batch_loss.item() * y.size(0) 
                
                _, predicted = torch.max(output.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        avg_val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        if avg_val_loss < best_val_loss: 
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        pbar.set_postfix({'Val Loss': f'{avg_val_loss:.4f}', 'Best Loss': f'{best_val_loss:.4f}', 'Val Acc': f'{val_acc:.2f}%'})

    # 3. TESTNING
    model.load_state_dict(best_model_state)
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            test_total += y.size(0)
            test_correct += (predicted == y).sum().item()

    final_test_acc = 100 * test_correct / test_total
    return final_test_acc

# --- 5. MAIN ---
if __name__ == "__main__":
    output_dim = 10
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    full_train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    
    experiments = {
        "GS-KAN":  {"struct": ["dim", 12, 12, output_dim], "use_norm": False}, 
        "MLP":     {"struct": ["dim", 13, 12, output_dim]},
        "Wav-KAN": {"struct": ["dim", 4, output_dim]},
        "Std-KAN": {"struct": ["dim", 1, output_dim]}
    }

    print("=" * 65)
    print(f" FASHION-MNIST BENCHMARK ({len(SEEDS)} seeds, {EPOCHS} epochs)")
    print("=" * 65)
    
    
    final_results = {}

    for name, config in experiments.items():
        if name == "Std-KAN" and not HAS_STD_KAN:
            continue
            
        print(f"\n--- Model: {name} ---")
        acc_list = []
        final_params = 0
        
        for seed in SEEDS:
            set_seed(seed)
            train_l, val_l, test_l, input_dim = get_data(seed, full_train_ds, test_ds)
            
            s = config["struct"].copy()
            s[0] = input_dim
            
            
            if name == "GS-KAN":
                model = GSKAN(s, degree=3, num_knots=60, grid_max=3 , use_input_norm=config["use_norm"]).to(DEVICE)
            elif name == "MLP":
                model = GeneralMLP(s).to(DEVICE)
            elif name == "Wav-KAN":
                model = WavKAN(s).to(DEVICE)
            elif name == "Std-KAN":
                model = StandardKAN(s, grid_size=8, spline_order=3).to(DEVICE)
                
            final_params = count_params(model)
            
            test_acc = train_one_seed(model, train_l, val_l, test_l)
            acc_list.append(test_acc)
            print(f"  Seed {seed} | Test Acc: {test_acc:.2f}%")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        final_results[name] = {
            "params": final_params,
            "mean": np.mean(acc_list),
            "std": np.std(acc_list)
        }

    #Report
    print("\n" + "="*65)
    print(f"{'Model':<10} | {'Params':<10} | {'Best Test Acc (Mean ± Std)':<30}")
    print("-" * 65)
    
    for name, data in final_results.items():
        print(f"{name:<10} | {data['params']:<10} | {data['mean']:.2f}% ± {data['std']:.2f}%")
    print("="*65)