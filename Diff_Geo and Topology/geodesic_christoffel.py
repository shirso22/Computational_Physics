
#Computes geodesics for any general manifold, using Christoffel symbols, uses PyTorch for GPU acceleration

import torch
import torch.nn as nn
from torchdiffeq import odeint # PyTorch-compatible ODE solver
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
# Set the device to GPU if available, otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Set the manifold dimension (for a 2D surface, D=2)
D = 2 

# --- 1. GENERALIZED CHRISTOFFEL SYMBOLS (PYTORCH) ---

class ChristoffelCalculator(nn.Module):
    """
    A PyTorch module to compute the Christoffel Symbols (Gamma^i_jk)
    for a general manifold, given its metric tensor function.
    """
    def __init__(self, metric_fn, device=DEVICE):
        super().__init__()
        self.metric_fn = metric_fn
        self.device = device
        
    def get_metric_and_derivatives(self, u):
        """
        Computes the metric tensor g_ij and its partial derivatives with
        respect to the coordinates u^k using automatic differentiation.
        
        Args:
            u (Tensor): Coordinates [u^1, u^2, ...]. Requires grad=True.
            
        Returns:
            g (Tensor): Metric tensor g_ij(u). Shape (D, D).
            dg_du (Tensor): Partial derivatives dg_ij / du^k. Shape (D, D, D).
        """
        # Ensure u is attached to the computational graph and on the correct device
        u = u.to(self.device).requires_grad_(True)

        # 1. Compute the metric tensor
        g = self.metric_fn(u)
        
        # 2. Compute partial derivatives dg_ij / du^k
        # This loop uses autograd to compute the Jacobian of the metric w.r.t coordinates u
        dg_du_list = []
        for k in range(D):
            # Compute the derivative of g w.r.t u[k]
            # torch.autograd.grad returns a tuple, we take the first element (the gradient tensor)
            grad_g = torch.autograd.grad(
                outputs=g.sum(),
                inputs=u,
                grad_outputs=torch.eye(D, D, device=self.device) if g.dim() == 2 else g,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )[0]
            # grad_g has shape (D, D) for the derivatives of g_ij w.r.t u^k
            if grad_g is None:
                # Handle cases where metric doesn't depend on a coordinate
                dg_du_list.append(torch.zeros(D, D, device=self.device))
            else:
                dg_du_list.append(grad_g[:, :, k]) # Extract the partial derivatives
        
        dg_du = torch.stack(dg_du_list, dim=2) # Shape (D, D, D) -> dg_ij / du^k
        
        return g.detach(), dg_du.detach() # Detach from graph for Christoffel computation
        
    def forward(self, u):
        """Computes Gamma^i_jk(u) using the full formula."""
        # Ensure u is 1D (D,) for a single point evaluation
        if u.dim() > 1:
             raise ValueError("Input u must be a 1D tensor of coordinates.")
        
        # Get g_ij and its derivatives dg_ij/du^k
        u_for_grad = u.clone().requires_grad_(True)
        g, dg_du = self.get_metric_and_derivatives(u_for_grad)
        
        # 1. Compute the inverse metric tensor g^il
        g_inv = torch.inverse(g) # Shape (D, D)
        
        # 2. Compute the term in the parenthesis: (dg_lj/du^k + dg_lk/du^j - dg_jk/du^l)
        # Note: Indices are: l, j, k
        
        # Permute dg_du to easily access the required derivatives
        # dg_du_l = dg_lk / du^j -> Permute from (j, k, l) to (l, k, j)
        dg_du_permuted = dg_du.permute(2, 1, 0)
        
        # Christoffel sum term: (dg_lj/du^k + dg_lk/du^j - dg_jk/du^l)
        # Index mapping:
        # dg_lj/du^k is dg_jk / du^l from original notation: dg_du[j, l, k]
        # dg_lk/du^j is dg_jl / du^k from original notation: dg_du[l, k, j]
        # dg_jk/du^l is dg_kl / du^j from original notation: dg_du[k, j, l]
        
        Gamma_sum = torch.zeros(D, D, D, device=self.device)
        
        for l in range(D):
            for j in range(D):
                for k in range(D):
                    # Indices: l (top-left), j (bottom-left), k (bottom-right)
                    # The sum uses indices l (first index of g_inv), j, k
                    
                    # Term 1: dg_{lj} / du^k
                    T1 = dg_du[l, j, k]
                    
                    # Term 2: dg_{lk} / du^j
                    T2 = dg_du[l, k, j]
                    
                    # Term 3: dg_{jk} / du^l
                    T3 = dg_du[j, k, l]
                    
                    Gamma_sum[l, j, k] = T1 + T2 - T3
        
        # 3. Compute Gamma^i_jk = 1/2 * g^il * Gamma_sum_l_jk
        # This is a matrix multiplication: (g_inv)_i^l * (Gamma_sum)_l^jk
        Gamma = 0.5 * torch.einsum('il, ljk -> ijk', g_inv, Gamma_sum)
        
        # The result is Gamma^i_jk, shape (D, D, D)
        return Gamma

# --- EXAMPLE METRIC: UNIT SPHERE (u^1=theta, u^2=phi) ---

def sphere_metric(u):
    """
    Metric tensor for a unit sphere: 
    ds^2 = d(theta)^2 + sin(theta)^2 d(phi)^2.
    
    Args:
        u (Tensor): Coordinates [theta, phi].
        
    Returns:
        Tensor: g_ij matrix.
    """
    theta = u[0]
    sin_theta = torch.sin(theta)
    
    g = torch.zeros(D, D, device=u.device)
    
    # g_11 = 1 (index 0)
    g[0, 0] = 1.0
    
    # g_22 = sin(theta)^2 (index 1)
    g[1, 1] = sin_theta**2
    
    # g_12 = g_21 = 0
    
    return g

# Initialize the Christoffel Calculator with the sphere's metric
gamma_calculator = ChristoffelCalculator(sphere_metric).to(DEVICE)

# --- 2. THE GEODESIC EQUATION (ODE SYSTEM - PYTORCH) ---

class GeodesicODE(nn.Module):
    """
    The right-hand side f(t, y) of the 1st-order ODE system for the geodesic equation.
    dy/dt = f(t, y)
    
    y = [u^1, ..., u^D, v^1, ..., v^D] where v^i = du^i/dt
    f = [v^1, ..., v^D, v_dot^1, ..., v_dot^D]
    
    The acceleration v_dot is:
    v_dot^i = - Sum_{j,k} Gamma^i_{jk} * v^j * v^k
    """
    def __init__(self, gamma_calculator):
        super().__init__()
        self.gamma_calculator = gamma_calculator
        
    def forward(self, t, y):
        """
        Args:
            t (Tensor): Time parameter. (Not explicitly used, as the metric is time-independent)
            y (Tensor): State vector [u, v]. Shape (2*D,).
            
        Returns:
            Tensor: Derivative vector [v, v_dot]. Shape (2*D,).
        """
        # Separate coordinates u and velocities v
        u = y[:D] # [theta, phi]
        v = y[D:] # [v_theta, v_phi]
        
        # 1. Get the Christoffel symbols evaluated at the current coordinates u
        # Gamma[i, j, k] = Gamma^i_jk
        Gamma = self.gamma_calculator(u)
        
        # 2. Calculate acceleration (v_dot)
        # We use einsum for efficient tensor contraction:
        # Sum_{j,k} Gamma^i_{jk} * v^j * v^k
        # 'ijk, j, k -> i'
        acceleration_sum = torch.einsum('ijk, j, k -> i', Gamma, v, v)
        
        v_dot = -acceleration_sum
        
        # 3. Return the full derivative vector [v, v_dot]
        return torch.cat([v, v_dot])

geodesic_ode = GeodesicODE(gamma_calculator).to(DEVICE)

# --- 3. SOLVING THE ODE SYSTEM (PYTORCH) ---

# Simulation Time
T_SPAN = [0, 10]
# PyTorch likes to receive the time points as a 1D tensor
T_POINTS = torch.linspace(T_SPAN[0], T_SPAN[1], 500).to(DEVICE)

# Initial Conditions (IC): [theta_0, phi_0, d_theta/dt_0, d_phi/dt_0]
theta_0 = torch.tensor(np.radians(1.0), dtype=torch.float32) # Almost at North Pole (1 degree)
phi_0 = torch.tensor(0.0, dtype=torch.float32)

v_theta_0 = torch.tensor(0.5, dtype=torch.float32)
v_phi_0 = torch.tensor(0.1, dtype=torch.float32)

# Normalization (optional, but good practice for unit speed parameterization)
u_0 = torch.tensor([theta_0, phi_0])
g_0 = sphere_metric(u_0) # Metric at initial point
v_0 = torch.tensor([v_theta_0, v_phi_0])

# Speed squared: ||v||^2 = g_ij * v^i * v^j
# einsum: 'ij, i, j ->' (sum over i and j)
norm_sq = torch.einsum('ij, i, j ->', g_0, v_0, v_0)

normalization_factor = torch.sqrt(1.0 / norm_sq)
v_0_normalized = v_0 * normalization_factor

# Initial state vector y0 = [u, v]
y0 = torch.cat([u_0, v_0_normalized]).to(DEVICE)

# Solve the ODE system using a PyTorch-compatible solver
print("Solving ODE system...")
solution_tensor = odeint(
    func=geodesic_ode, 
    y0=y0, 
    t=T_POINTS, 
    method='rk4' # Standard Runge-Kutta 4th order
)
print("ODE solution complete.")

# Extract coordinates and move to CPU for plotting (NumPy required for matplotlib)
# solution_tensor has shape (T_POINTS, 2*D)
solution_np = solution_tensor.cpu().numpy()

theta_path = solution_np[:, 0]
phi_path = solution_np[:, 1]
# Velocities are in solution_np[:, 2] and solution_np[:, 3]

# --- 4. CONVERT TO CARTESIAN FOR 3D PLOTTING (NUMPY) ---

# Cartesian coordinates for the path
X_path = np.sin(theta_path) * np.cos(phi_path)
Y_path = np.sin(theta_path) * np.sin(phi_path)
Z_path = np.cos(theta_path)

# --- Create the background sphere surface ---
# U = phi, V = theta (standard convention for sphere parametrization)
U, V = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j] 
X_sphere = np.sin(V) * np.cos(U)
Y_sphere = np.sin(V) * np.sin(U)
Z_sphere = np.cos(V)

# 

# --- Plotting ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"Geodesic on a Unit Sphere (PyTorch/GPU Accelerated) | Device: {DEVICE}")

# 1. Plot the transparent sphere surface
ax.plot_surface(X_sphere, Y_sphere, Z_sphere, 
                color='lightblue', alpha=0.15, linewidth=0)

# 2. Plot the calculated geodesic path
ax.plot(X_path, Y_path, Z_path, color='red', linewidth=3, label='Geodesic Path')

# 3. Mark the start and end points
ax.scatter(X_path[0], Y_path[0], Z_path[0], color='green', s=50, label='Start')
ax.scatter(X_path[-1], Y_path[-1], Z_path[-1], color='blue', s=50, label='End')

# Set equal axis limits for a spherical appearance
max_range = 1.1
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()
plt.show()
