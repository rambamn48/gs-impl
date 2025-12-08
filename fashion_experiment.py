import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# --- 1. IMPORTS ---
from classes_collection import GSKAN, GeneralMLP

# --- 2. CONFIGURATION ---
SEEDS = [0, 1, 2]
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

def get_data():
    # Flatten images: 28x28 -> 784
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # --- DATA LOADING OPTIMIZATION ---
    # num_workers=2: Use 2 CPU cores for loading
    # pin_memory=True: Faster transfer to GPU
    # persistent_workers=True: Keep workers alive between epochs
    kwargs = {'num_workers': 2, 'pin_memory': True, 'persistent_workers': True} if DEVICE.type == 'cuda' else {}
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    
    return train_loader, test_loader, 784

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_seed(model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    final_acc = 0.0

    model.train()

    # --- PROGRESS BAR ---
    pbar = tqdm(range(EPOCHS), desc="Training")

    for epoch in pbar:
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = 100 * correct / total
        if acc > best_acc: best_acc = acc
        final_acc = acc

        pbar.set_postfix({'Acc': f'{acc:.2f}%', 'Best': f'{best_acc:.2f}%'})

    return best_acc, final_acc

# --- 4. MAIN ---
if __name__ == "__main__":

    # Initialize results file
    with open("fashion_results.txt", "w") as f:
        f.write("STARTING BENCHMARK\n")

    train_loader, test_loader, input_dim = get_data()
    output_dim = 10

    results = {
        "GS-KAN": {"best": [], "final": [], "params": 0},
        "MLP":    {"best": [], "final": [], "params": 0}
    }

    print("-" * 50)
    print(f"Starting Benchmark ({len(SEEDS)} seeds, {EPOCHS} epochs)")
    print("-" * 50)

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        set_seed(seed)

        # 1. GS-KAN
        model_gs = GSKAN([input_dim, 15, 15, output_dim], degree=3, num_knots=60, grid_max=3, use_input_norm=False).to(DEVICE)
        results["GS-KAN"]["params"] = count_params(model_gs)

        print(f"Training GS-KAN ({results['GS-KAN']['params']} params)...")
        best_gs, final_gs = train_one_seed(model_gs, train_loader, test_loader)
        results["GS-KAN"]["best"].append(best_gs)
        results["GS-KAN"]["final"].append(final_gs)

        # Log immediately
        with open("fashion_results.txt", "a") as f:
            f.write(f"GS-KAN | Seed {seed} | Params: {results['GS-KAN']['params']} | Best: {best_gs:.2f}% | Final: {final_gs:.2f}%\n")

        # 2. MLP
        set_seed(seed)
        # Structure: [784, 16, 16, 10]
        model_mlp = GeneralMLP([input_dim, 16, 16, output_dim]).to(DEVICE)
        results["MLP"]["params"] = count_params(model_mlp)

        print(f"Training MLP    ({results['MLP']['params']} params)...")
        best_mlp, final_mlp = train_one_seed(model_mlp, train_loader, test_loader)
        results["MLP"]["best"].append(best_mlp)
        results["MLP"]["final"].append(final_mlp)

        # Log immediately
        with open("fashion_results.txt", "a") as f:
            f.write(f"MLP    | Seed {seed} | Params: {results['MLP']['params']} | Best: {best_mlp:.2f}% | Final: {final_mlp:.2f}%\n")

    # --- REPORT ---
    print("\n" + "="*60)
    header = f"{'Model':<10} | {'Params':<10} | {'Best Acc (Mean ± Std)':<25} | {'Final Acc (Mean ± Std)':<25}"
    print(header)
    print("-" * 60)

    # Save summary to file
    with open("fashion_results.txt", "a") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(header + "\n")
        f.write("-" * 60 + "\n")

    for name, data in results.items():
        best_mean = np.mean(data["best"])
        best_std = np.std(data["best"])
        final_mean = np.mean(data["final"])
        final_std = np.std(data["final"])

        res_str = f"{name:<10} | {data['params']:<10} | {best_mean:.2f} ± {best_std:.2f} %       | {final_mean:.2f} ± {final_std:.2f} %"
        print(res_str)

        with open("fashion_results.txt", "a") as f:
            f.write(res_str + "\n")

    print("="*60)