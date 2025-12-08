import torch
import torch.nn as nn
import numpy as np


class GSKAN(nn.Module):
    """
    Modified SKAN (Sprecher KAN).
    This is an implementation of the Generalized Sprecher KAN architecture based on the methodology 
    proposed in [Eliasson, "GS-KAN: Parameter-Efficient Kolmogorov-Arnold
    Networks via Sprecher-Type Shared Basis Functions"]
    """
    def __init__(self, structure, degree, num_knots, grid_max=3, use_silu=False, use_input_norm=True):
        super().__init__()
        self.degree = degree
        self.structure = structure  # List defining the layers, e.g., [2, 5, 1]
        self.grid_max = grid_max    # Defines the interval [-grid_max, grid_max] for splines
        self.use_silu = use_silu         # Flag: Should we use SiLU as a 'residual' connection?
        self.use_input_norm = use_input_norm # Flag: Should we normalize input to keep it within the grid?
        
        # --- 1. Create Grid (Knots / X-axis structure) ---
        # Knots are the FIXED positions on the x-axis that define the grid resolution.
        # They act as the "skeleton" or "grid lines" for the splines and generally do not change.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.knot_buffer = 0.2
        
        # Create the vector of knot positions (defining the intervals).
        self.knots = torch.arange(
            -(self.grid_max + self.knot_buffer), 
            (self.grid_max + self.knot_buffer), 
            step=2 * (self.grid_max + self.knot_buffer) / num_knots, 
            dtype=torch.float32, 
            device=device
        )
        
        # --- 2. Define Learnable Parameters (Coefficients / Y-axis control) ---
        # Coefficients are the LEARNABLE and determine the shape/height of the curve.
        # Mathematical constraint: We need fewer coefficients than knots.
        # Formula: num_learnable_params = num_knots - spline_degree - 1
        self.num_c = self.knots.size(0) - degree - 1
        self.std = 0.1
    
        # Parameters for ALL layers. These shape the curve itself (layer function).
        # Dimension: (Number of layers, Number of coefficients per function)
        self.cs = nn.Parameter(torch.normal(mean=0, std=self.std, size=(len(structure)-1, self.num_c)))

        
        # --- 3. Lambda Matrices ---
        # Scales the contribution of each layer spline function.
        self.lambda_matrices = nn.ParameterList([
            nn.Parameter(torch.normal(mean=0, std=self.std, size=(structure[i], structure[i+1]))) 
            for i in range(len(structure)-1)
        ])

        # --- 4. Weight Initialization (Residual / Base weight) ---
        # Xavier initialization to ensure a stable start.
        sum_in_out_nodes = [structure[i]+structure[i+1] for i in range(len(structure)-1)]
        self.std_list = torch.tensor([self.xavier(x) for x in sum_in_out_nodes], dtype=torch.float32, device=device)
        
        if self.use_silu:
            # If SiLU is on: Two weights per layer -> [Weight for Spline, Weight for SiLU]
            xavier = torch.normal(0, self.std_list)
            ones = torch.ones(len(structure)-1, device=device)
            weights = torch.stack([ones, xavier], dim=1)
        else:
            # If SiLU is off: Only one weight -> [Weight for Spline]
            weights = torch.ones(len(structure)-1, 1, device=device)
            
        self.weights = nn.Parameter(weights)

        # --- 5. Bias (Epsilon) ---
        # A learnable shift of the input BEFORE it enters the spline function.
        # This allows the splines to "see" different parts of the input data.
        self.epz = nn.Parameter(torch.abs(torch.normal(mean=0, std=0.1, size=(sum(structure[1:]),))))
        self.structure_bias = structure[1:] 

    def normalize_input(self, x):
        """
        Normalizes the input so it falls within the range where our splines are defined.
        Without this, splines might output 0 if the input is too large/small.
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x = (x - mean) / std
        return torch.clamp(x, min=-self.grid_max, max=self.grid_max)

    def xavier(self, x): return np.sqrt(2/x)
    
    def basis_function(self, degree, knots, t):
        """
        Recursive calculation of B-Spline basis functions (Cox-de Boor formula).
        Gives us the 'shapes' that we then scale with coefficients.
        """
        num_basis = len(knots) - degree - 1
        t = t.unsqueeze(-1)
        # Base case: Degree 0 (step functions)
        B = ((knots[:-1] <= t) & (t < knots[1:])).float()
        # Recursion up to desired degree
        for d in range(1, degree + 1):
            left = (t - knots[:-d-1]) / (knots[d:-1] - knots[:-d-1])
            right = (knots[d+1:] - t) / (knots[d+1:] - knots[1:-d])
            B = left * B[..., :-1] + right * B[..., 1:]
        return B

    def BSpline(self, c, knot_interval, degree, t):
        """
        Calculates the actual value of the Spline function:
        Sum of (coefficients * basis_functions).
        """
        B = self.basis_function(degree, knot_interval, t)
        return torch.sum(B * c, dim=-1)

    def phi(self, tensor, c, weight):
        """
        Phi is the layer function.
        Phi(x) = w_spline * Spline(x) + w_silu * SiLU(x)
        """
        spline_part = self.BSpline(c, self.knots, self.degree, tensor)
        
        if self.use_silu:
            silu_part = nn.functional.silu(tensor)
            return weight[0] * spline_part + weight[1] * silu_part
        else:
            return weight[0] * spline_part 

    def layer_pass(self, x, layer):
        """
        Performs the calculation for ONE layer in KAN.
        
        The math here differs from MLP:
        Instead of Matrix * Vector, we calculate Phi(x_i) for every 
        combination of input node and output node.
        """
        device = x.device
        current_batch = x.shape[0] # Dynamic batch size
        
        # Get the correct bias parameters for this layer
        start_idx = sum(self.structure_bias[:layer])
        end_idx = sum(self.structure_bias[:layer+1])
        current_epz = self.epz[start_idx:end_idx]
        
        # --- Tensor Expansion (The Magic of KAN) ---
        # We must create a tensor representing all connections.
        # x_expanded shape: (Batch, Input_dim, Output_dim)
        x_expanded = x.unsqueeze(2).repeat(1, 1, self.structure[layer+1])
        
        # bias_expanded is added to input. Each output node has its own bias shift for every input.
        bias_expanded = current_epz.unsqueeze(0).unsqueeze(0).repeat(current_batch, self.structure[layer], 1)
        term = x_expanded + bias_expanded
        
        # Calculate phi(x) for all edges simultaneously
        phi_out = self.phi(term, self.cs[layer], self.weights[layer])
        
        # Scale with the lambda matrix (how important is each connection?)
        weighted = phi_out * self.lambda_matrices[layer].unsqueeze(0)
        
        # Sum over the input dimension (dim=1) to get the value for each output node
        return torch.sum(weighted, dim=1)
    
    def forward(self, x):
        # --- Control normalization with flag ---
        if self.use_input_norm:
            x = self.normalize_input(x)
        
        input_val = x
        for layer in range(len(self.structure)-1):
            input_val = self.layer_pass(input_val, layer)
            
        return input_val
    


class WavKANLayer(nn.Module):
    """
    A layer using Wavelets (Mexican Hat) instead of Splines.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(output_dim, input_dim)) # Weights
        self.t = nn.Parameter(torch.randn(output_dim, input_dim)) # Translation (move the wave)
        self.s = nn.Parameter(torch.ones(output_dim, input_dim))  # Scaling (width of the wave)
    def forward(self, x):
        x = x.unsqueeze(1)
        # Normalize x based on learned translation and scaling
        u = (x - self.t) / self.s
        # Mexican Hat wavelet formula
        mexican_hat = (1 - u**2) * torch.exp(-0.5 * u**2)
        return torch.sum(self.w * mexican_hat, dim=2)



class WavKAN(nn.Module):
    def __init__(self, structure):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(structure) - 1):
            self.layers.append(WavKANLayer(structure[i], structure[i+1]))
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x


class GeneralMLP(nn.Module):
    """
    Standard Multi-Layer Perceptron for comparison.
    """
    def __init__(self, structure, activation=nn.ReLU):
        """
        Args:
            structure (list): E.g., [Input, Hidden, ..., Output]
            activation (class): Activation class (e.g., nn.ReLU or nn.SiLU). 
                                NOTE: Pass the class, not an instance.
        """
        super().__init__()
        layers = []
        for i in range(len(structure) - 1):
            # 1. Add Linear layer
            layers.append(nn.Linear(structure[i], structure[i+1]))
            
            # 2. Add Activation (Same logic as before: Not after the last layer)
            if i < len(structure) - 2:
                layers.append(activation()) 
                
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)