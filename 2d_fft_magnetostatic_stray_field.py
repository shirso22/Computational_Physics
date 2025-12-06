import torch
import torch.fft

class StrayFieldSolverGPU:
    """
    GPU-accelerated magnetostatic stray field solver for 2D thin films.
    
    This implementation uses Fourier-space kernels with full GPU optimization:
    - All computations stay on GPU (no CPU↔GPU transfers)
    - Precomputed k-space kernels for repeated calls
    - Support for mixed precision (float32/complex64)
    - Batch processing capability

    You can integrate the simple compute_B_field method on its own with minimal changes into your own pipeline and if you are not really looking for batch
    processing and extremely high throughput necessarily
    
    Physics: Computes B = μ₀(H_demag) above a thin magnetic film using
             the Fourier-space approach with thickness and standoff corrections.
    
    Reference: Standard magnetostatic kernel method for thin films.
    """
    
    def __init__(self, dx, dy, thickness, height, mu_0=4e-7*torch.pi, 
                 device=None, dtype=torch.float32):
        """
        Initialize solver with physical parameters.
        
        Args:
            dx, dy: Grid spacings in meters (x and y directions)
            thickness: Film thickness in meters
            height: Standoff height in meters (distance from film surface to observation plane)
            mu_0: Magnetic constant (default: 4π × 10⁻⁷ H/m)
            device: Torch device ('cuda', 'cpu', or None for auto-detection)
            dtype: Base dtype for real computations (torch.float32 or torch.float64)
        """
        # Set device: prefer GPU if available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Store parameters as torch tensors on target device
        self.dx = torch.tensor(dx, device=self.device, dtype=dtype)
        self.dy = torch.tensor(dy, device=self.device, dtype=dtype)
        self.thickness = torch.tensor(thickness, device=self.device, dtype=dtype)
        self.height = torch.tensor(height, device=self.device, dtype=dtype)
        self.MU_0 = torch.tensor(mu_0, device=self.device, dtype=dtype)
        
        # Precision settings
        self.dtype = dtype  # Real dtype (float32/float64)
        self.cplx_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        
        # Cache for precomputed k-space kernels (keyed by grid size)
        # This is the KEY OPTIMIZATION: compute k-space factors once, reuse many times
        self._k_cache = {}
        
        print(f"Initialized solver on {self.device} with dtype {self.dtype}")
        
    def _compute_kernels(self, nx, ny):
        """
        Compute ALL k-space kernels for a given grid size.
        
        This is the critical optimization: compute k-space factors entirely on GPU
        and cache them for reuse. No CPU intermediates!
        
        Args:
            nx, ny: Original grid dimensions (before padding)
            
        Returns:
            Dictionary containing all precomputed k-space tensors
        """
        # Double-size padding for convolution (avoids wraparound artifacts)
        nx_pad = 2 * nx
        ny_pad = 2 * ny
        
        # ------------------------------------------------------------
        # CRITICAL: Compute k-vectors ENTIRELY on GPU
        # Using torch.fft.fftfreq with device parameter ensures no CPU↔GPU transfer
        # ------------------------------------------------------------
        kx = 2.0 * torch.pi * torch.fft.fftfreq(
            nx_pad, d=self.dx, device=self.device
        ).to(dtype=self.dtype)
        
        ky = 2.0 * torch.pi * torch.fft.rfftfreq(
            ny_pad, d=self.dy, device=self.device
        ).to(dtype=self.dtype)
        
        # Create k-grids (shape: [nx_pad, ny_pad//2 + 1])
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
        
        # Magnitude of k-vector with small epsilon for numerical stability
        # Note: Using torch.hypot is more stable but sqrt(x²+y²) is fine here
        eps = torch.tensor(1e-12, device=self.device, dtype=self.dtype)
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + eps)
        
        # Mask for k=0 modes (DC component)
        k_zero_mask = k_mag <= eps  # Use epsilon for floating point comparison
        
        # Normalized k-vectors (kx/k, ky/k)
        # Initialize with zeros (handles k=0 case automatically)
        kx_norm = torch.zeros_like(kx_grid, dtype=self.dtype, device=self.device)
        ky_norm = torch.zeros_like(ky_grid, dtype=self.dtype, device=self.device)
        
        # Only compute normalization where k ≠ 0
        nonzero_mask = ~k_zero_mask
        if nonzero_mask.any():
            k_mag_nonzero = k_mag[nonzero_mask]
            kx_norm[nonzero_mask] = kx_grid[nonzero_mask] / k_mag_nonzero
            ky_norm[nonzero_mask] = ky_grid[nonzero_mask] / k_mag_nonzero
        
        # Store everything in cache
        kernels = {
            'kx_grid': kx_grid,          # kx components
            'ky_grid': ky_grid,          # ky components
            'k_mag': k_mag,              # |k|
            'kx_norm': kx_norm,          # kx/|k|
            'ky_norm': ky_norm,          # ky/|k|
            'k_zero_mask': k_zero_mask,  # Mask for k=0
            'nonzero_mask': nonzero_mask,# Mask for k≠0
            'nx_pad': nx_pad,            # Padded dimensions
            'ny_pad': ny_pad,
            'nx_orig': nx,               # Original dimensions (for cropping)
            'ny_orig': ny,
        }
        
        return kernels
    
    def _get_kernels(self, nx, ny):
        """
        Get cached kernels or compute if not already cached.
        
        This pattern ensures we only compute k-space factors once per grid size,
        which is the main source of GPU acceleration for repeated calls.
        """
        cache_key = (nx, ny)
        
        if cache_key not in self._k_cache:
            # Compute and cache kernels
            self._k_cache[cache_key] = self._compute_kernels(nx, ny)
            
            # Optional: Clear old cache entries if memory is constrained
            if len(self._k_cache) > 10:  # Keep last 10 grid sizes
                oldest_key = next(iter(self._k_cache))
                del self._k_cache[oldest_key]
        
        return self._k_cache[cache_key]
    
    def compute_B_field(self, M_grid, height=None):
        """
        Compute stray field B(x,y,h) above a thin magnetic film using Fourier-space kernels
        
        Args:
            M_grid: (Nx, Ny, 3) real tensor in units A/m (Mx, My, Mz).
                     Can be on CPU or GPU; will be moved to solver's device.
            height: Optional standoff height h (meters). If None, uses self.height.
            
        Returns:
            B: (Nx, Ny, 3) real tensor (Tesla) on the same device as solver.
        """
        # ------------------------------------------------------------
        # 1. SETUP AND INPUT VALIDATION
        # ------------------------------------------------------------
        if height is None:
            height_tensor = self.height
        else:
            height_tensor = torch.tensor(height, device=self.device, dtype=self.dtype)
        
        # Ensure input is on correct device and dtype
        # Note: .to() is efficient if already on correct device
        M_grid = M_grid.to(device=self.device, dtype=self.dtype)
        
        # Get original dimensions
        nx, ny = M_grid.shape[0], M_grid.shape[1]
        
        # ------------------------------------------------------------
        # 2. GET PRECOMPUTED K-SPACE KERNELS (GPU OPTIMIZATION KEY)
        # ------------------------------------------------------------
        kernels = self._get_kernels(nx, ny)
        
        # Extract cached values
        kx_grid = kernels['kx_grid']
        ky_grid = kernels['ky_grid']
        k_mag = kernels['k_mag']
        kx_norm = kernels['kx_norm']
        ky_norm = kernels['ky_norm']
        k_zero_mask = kernels['k_zero_mask']
        nx_pad = kernels['nx_pad']
        ny_pad = kernels['ny_pad']
        
        # ------------------------------------------------------------
        # 3. SYMMETRIC ZERO-PADDING
        # ------------------------------------------------------------
        pad_x = nx // 2
        pad_y = ny // 2
        
        # Create padded magnetization array
        M_padded = torch.zeros((nx_pad, ny_pad, 3), 
                              dtype=self.dtype, 
                              device=self.device)
        M_padded[pad_x:pad_x + nx, pad_y:pad_y + ny, :] = M_grid
        
        # ------------------------------------------------------------
        # 4. FORWARD FFT (real to complex)
        # ------------------------------------------------------------
        M_fft = torch.fft.rfftn(M_padded, s=(nx_pad, ny_pad), 
                               dim=(0, 1), norm='forward')
        # M_fft is complex (same dtype as cplx_dtype)
        
        # ------------------------------------------------------------
        # 5. COMPUTE PHYSICS FACTORS (all on GPU)
        # ------------------------------------------------------------
        # Attenuation factor: exp(-k·h)
        attenuation = torch.exp(-k_mag * height_tensor)
        
        # Thickness factor: 1 - exp(-k·t)
        # Using expm1 for numerical stability: expm1(x) = exp(x) - 1
        kt = k_mag * self.thickness
        thickness_factor = -torch.expm1(-kt)  # = 1 - exp(-kt)
        
        # Overall prefactor: (μ₀/2) × (1 - exp(-k·t)) × exp(-k·h)
        prefactor_real = 0.5 * self.MU_0 * thickness_factor * attenuation
        
        # Convert to complex dtype for multiplication with M_fft
        prefactor_c = prefactor_real.to(dtype=self.cplx_dtype)
        kx_c = kx_grid.to(dtype=self.cplx_dtype)
        ky_c = ky_grid.to(dtype=self.cplx_dtype)
        k_mag_c = k_mag.to(dtype=self.cplx_dtype)
        kx_norm_c = kx_norm.to(dtype=self.cplx_dtype)
        ky_norm_c = ky_norm.to(dtype=self.cplx_dtype)
        
        # ------------------------------------------------------------
        # 6. FOURIER-DOMAIN MAGNETIZATION COMPONENTS
        # ------------------------------------------------------------
        # Extract complex magnetization components
        Mx_fft = M_fft[..., 0]
        My_fft = M_fft[..., 1]
        Mz_fft = M_fft[..., 2]
        
        # k·M_parallel (in-plane dot product in Fourier domain)
        k_dot_M_par = kx_c * Mx_fft + ky_c * My_fft
        
        # ------------------------------------------------------------
        # 7. ENFORCE PHYSICAL BOUNDARY CONDITION AT k=0
        # ------------------------------------------------------------
        # Uniform magnetization (k=0) should produce no stray field
        # This is both physically correct and numerically necessary
        
        # Create writable copies to avoid in-place modification errors
        k_dot_M_par = k_dot_M_par.clone()
        Mz_fft = Mz_fft.clone()
        
        # Zero out k=0 components
        if k_zero_mask.any():
            k_dot_M_par[k_zero_mask] = 0.0 + 0.0j
            Mz_fft[k_zero_mask] = 0.0 + 0.0j
            # Note: kx_norm_c and ky_norm_c are already zero at k=0
        
        # ------------------------------------------------------------
        # 8. COMPUTE B-FIELD IN FOURIER DOMAIN 
        # ------------------------------------------------------------
        # Initialize complex B-field in Fourier domain
        B_fft = torch.zeros_like(M_fft, dtype=self.cplx_dtype)
        
        # x-component: B_x = -(μ₀/2) e^{-kh} [ (kx/k)(k·M_par) + i·kx·M_z ]
        B_fft[..., 0] = -prefactor_c * (kx_norm_c * k_dot_M_par + 1j * kx_c * Mz_fft)
        
        # y-component: B_y = -(μ₀/2) e^{-kh} [ (ky/k)(k·M_par) + i·ky·M_z ]
        B_fft[..., 1] = -prefactor_c * (ky_norm_c * k_dot_M_par + 1j * ky_c * Mz_fft)
        
        # z-component: B_z = (μ₀/2) e^{-kh} [ k·M_z - i·(k·M_par) ]
        B_fft[..., 2] = prefactor_c * (k_mag_c * Mz_fft - 1j * k_dot_M_par)
        
        # ------------------------------------------------------------
        # 9. INVERSE FFT BACK TO REAL SPACE
        # ------------------------------------------------------------
        B_padded = torch.fft.irfftn(B_fft, s=(nx_pad, ny_pad), 
                                   dim=(0, 1), norm='forward')
        # B_padded is real-valued (same dtype as self.dtype)
        
        # ------------------------------------------------------------
        # 10. CROP BACK TO ORIGINAL GRID SIZE
        # ------------------------------------------------------------
        B = B_padded[pad_x:pad_x + nx, pad_y:pad_y + ny, :]
        
        return B
    
    def compute_B_field_batch(self, M_input, height=None):
    """
    A version of the previous method optimized for batch processing 
    Compute stray field B(x,y,h) for a single grid or a batch of grids.
    This implementation is fully vectorized using batch FFT.

    Args:
        M_input: (Nx, Ny, 3) or (B, Nx, Ny, 3) real tensor (A/m).
        height: Optional standoff height h (meters). If None, uses self.height.
                Can be scalar (for all grids) or a 1D tensor of shape (B).

    Returns:
        B: (Nx, Ny, 3) or (B, Nx, Ny, 3) real tensor (Tesla) on the solver's device.
    """
    is_batched = (M_input.ndim == 4)
    if not is_batched:
        # Add batch dimension of size 1 for vectorized processing
        M_grid = M_input.unsqueeze(0)
    else:
        M_grid = M_input

    # 1. SETUP AND INPUT VALIDATION
    batch_size, nx, ny = M_grid.shape[0], M_grid.shape[1], M_grid.shape[2]
    
    # Ensure input is on correct device and dtype
    M_grid = M_grid.to(device=self.device, dtype=self.dtype)
    
    if height is None:
        height_tensor = self.height
    else:
        # Convert height to a tensor compatible with broadcasting (B, 1, 1) or (1, 1, 1)
        height_tensor = torch.tensor(height, device=self.device, dtype=self.dtype)
        if height_tensor.ndim == 0:
            height_tensor = height_tensor.unsqueeze(0) # (1)
        
        # Ensure height_tensor is broadcastable to (B, Nx_pad, Ny_pad//2 + 1)
        if height_tensor.ndim == 1:
            height_tensor = height_tensor.view(batch_size, 1, 1)

    # 2. GET PRECOMPUTED K-SPACE KERNELS
    kernels = self._get_kernels(nx, ny)
    
    kx_grid = kernels['kx_grid']
    ky_grid = kernels['ky_grid']
    k_mag = kernels['k_mag']
    kx_norm = kernels['kx_norm']
    ky_norm = kernels['ky_norm']
    k_zero_mask = kernels['k_zero_mask']
    nx_pad = kernels['nx_pad']
    ny_pad = kernels['ny_pad']

    # 3. SYMMETRIC ZERO-PADDING
    pad_x = nx // 2
    pad_y = ny // 2
    
    # Create padded magnetization array: (B, Nx_pad, Ny_pad, 3)
    M_padded = torch.zeros((batch_size, nx_pad, ny_pad, 3), 
                            dtype=self.dtype, 
                            device=self.device)
    M_padded[:, pad_x:pad_x + nx, pad_y:pad_y + ny, :] = M_grid

    # 4. FORWARD BATCH FFT (over spatial dims 1 and 2)
    # M_fft shape: (B, Nx_pad, Ny_pad//2 + 1, 3)
    M_fft = torch.fft.rfftn(M_padded, s=(nx_pad, ny_pad), 
                            dim=(1, 2), norm='forward')

    # 5. COMPUTE PHYSICS FACTORS (Batch-ready)
    
    # Reshape k_mag to (1, Nx_pad, Ny_pad//2 + 1) for batch broadcasting
    k_mag_broadcast = k_mag.unsqueeze(0)
    
    # Attenuation factor: exp(-k·h) (Calculated per batch item if h is batched)
    # Result shape: (B, Nx_pad, Ny_pad//2 + 1)
    attenuation = torch.exp(-k_mag_broadcast * height_tensor)
    
    # Thickness factor: 1 - exp(-k·t) (kt is spatial only, same for all batch items)
    kt = k_mag_broadcast * self.thickness
    thickness_factor = -torch.expm1(-kt) 
    
    # Overall prefactor: (μ₀/2) × (1 - exp(-k·t)) × exp(-k·h)
    prefactor_real = 0.5 * self.MU_0 * thickness_factor * attenuation
    
    # Convert kernel arrays to complex dtype and ensure batch-ready shape
    # (1, Nx_pad, Ny_pad//2 + 1) to multiply with M_fft (B, Nx_pad, Ny_pad//2 + 1, 3)
    prefactor_c = prefactor_real.to(dtype=self.cplx_dtype)
    kx_c = kx_grid.to(dtype=self.cplx_dtype).unsqueeze(0)
    ky_c = ky_grid.to(dtype=self.cplx_dtype).unsqueeze(0)
    k_mag_c = k_mag.to(dtype=self.cplx_dtype).unsqueeze(0)
    kx_norm_c = kx_norm.to(dtype=self.cplx_dtype).unsqueeze(0)
    ky_norm_c = ky_norm.to(dtype=self.cplx_dtype).unsqueeze(0)

    # 6. FOURIER-DOMAIN MAGNETIZATION COMPONENTS
    
    # k·M_parallel (in-plane dot product)
    k_dot_M_par = kx_c * M_fft[..., 0] + ky_c * M_fft[..., 1]
    Mz_fft = M_fft[..., 2]
    
    # 7. ENFORCE PHYSICAL BOUNDARY CONDITION AT k=0
    
    # k_zero_mask shape is (Nx_pad, Ny_pad//2 + 1). Use it to index the (B, ...) tensors.
    # Note: The k=0 entry is always at index [0, 0].
    
    k_dot_M_par[:, k_zero_mask] = 0.0 + 0.0j
    Mz_fft[:, k_zero_mask] = 0.0 + 0.0j
    
    # 8. COMPUTE B-FIELD IN FOURIER DOMAIN
    
    B_fft = torch.zeros_like(M_fft, dtype=self.cplx_dtype)
    
    # Kernel expression uses prefactor_c which now correctly has batch dimension (B, ...)
    B_fft[..., 0] = -prefactor_c * (kx_norm_c * k_dot_M_par + 1j * kx_c * Mz_fft)
    B_fft[..., 1] = -prefactor_c * (ky_norm_c * k_dot_M_par + 1j * ky_c * Mz_fft)
    B_fft[..., 2] = prefactor_c * (k_mag_c * Mz_fft - 1j * k_dot_M_par)
    
    # 9. INVERSE BATCH FFT BACK TO REAL SPACE
    # B_padded shape: (B, Nx_pad, Ny_pad, 3)
    B_padded = torch.fft.irfftn(B_fft, s=(nx_pad, ny_pad), 
                                dim=(1, 2), norm='forward')
    
    # 10. CROP BACK TO ORIGINAL GRID SIZE
    B_cropped = B_padded[:, pad_x:pad_x + nx, pad_y:pad_y + ny, :]
    
    if not is_batched:
        # Remove the batch dimension if the input was a single grid
        B_cropped = B_cropped.squeeze(0)
        
    return B_cropped
    
    def clear_cache(self):
        """Clear the k-space kernel cache to free GPU memory."""
        self._k_cache.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("Cleared kernel cache")
    
    def to(self, device=None, dtype=None):
        """
        Move solver to a different device or change dtype.
        
        Args:
            device: Target device ('cuda', 'cpu')
            dtype: Target dtype (torch.float32, torch.float64)
            
        Returns:
            self (modified in-place)
        """
        if device is not None:
            self.device = torch.device(device)
            
        if dtype is not None:
            self.dtype = dtype
            self.cplx_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        
        # Update all stored tensors
        attrs = ['dx', 'dy', 'thickness', 'height', 'MU_0']
        for attr in attrs:
            tensor = getattr(self, attr)
            setattr(self, attr, tensor.to(device=self.device, dtype=self.dtype))
        
        # Clear cache since kernels need to be recomputed
        self.clear_cache()
        
        return self


