import torch

def calculate_k_space(Ny, Nx, dx, device):
    """
    Calculates the 2D wave vector magnitude k = |k| tensor.
    This tensor is only calculated once and moved to the GPU.
    """
    # Use torch.fft.fftfreq which returns the unshifted k-space
    # k = 2 * pi * f
    kx_unsh = 2 * torch.pi * torch.fft.fftfreq(Nx, d=dx, device=device)
    ky_unsh = 2 * torch.pi * torch.fft.fftfreq(Ny, d=dx, device=device)

    # Create the 2D meshgrid for kx and ky
    kx, ky = torch.meshgrid(kx_unsh, ky_unsh, indexing='ij')

    # Calculate the magnitude of the wave vector k = |k|
    k = torch.sqrt(kx**2 + ky**2)
    return k

def fourier_inversion_magnetization_mz(Bz_batch, z, dx, mu0=4 * torch.pi * 1e-7):
    """
    Reconstructs the 2D out-of-plane magnetization (Mz). Uses PyTorch for GPU acceleration and supports batch processing.
    Based on the theory presented in the paper "Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements" by Broadway et al, Physical Review Applied, 2020
    Important thing to note, this method only cleanly reconstructs for a pure out of plane magnetization Mz, from the Bz component of the field, as in plane components amplify the noise.

    Args:
        Bz_batch (torch.Tensor): Tensor of measured Bz fields.
                                 Shape: [Batch_Size, Ny, Nx]. Units: Tesla.
        z (float): The constant lift height above the source plane (in meters).
        dx (float): The pixel size/spatial resolution of the map (in meters).
        mu0 (float): Permeability of free space (default: 4 * pi * 1e-7 T*m/A).

    Returns:
        torch.Tensor: Reconstructed Mz tensor. Shape: [Batch_Size, Ny, Nx]. Units: A/m.
    """
    # 1. Device and Shape Information
    device = Bz_batch.device
    B, Ny, Nx = Bz_batch.shape

    # 2. Calculate k-space and the Inversion Kernel K_Mz
    k = calculate_k_space(Ny, Nx, dx, device)

    # Define the Magnetization Inversion Kernel K_Mz
    # K_Mz = (2 / (mu0 * k)) * exp(k * z)
    
    K_Mz = torch.zeros_like(k, dtype=torch.float32)

    # Find all non-zero k components (avoid division by zero and k=0 singularity)
    non_zero_k = k != 0
    
    # Apply the kernel to the non-zero components
    # The kernel is applied element-wise and must be float for real math
    K_Mz[non_zero_k] = (2.0 / (mu0 * k[non_zero_k])) * torch.exp(k[non_zero_k] * z)
    
    # Convert K_Mz to complex to match the FFT output
    K_Mz_complex = K_Mz.to(dtype=torch.complex64)

    # 3. Compute the Fourier Transform of the measured Bz field
    # torch.fft.fft2 handles the batch dimension automatically
    Bz_tilde = torch.fft.fft2(Bz_batch)

    # 4. Apply the kernel
    # K_Mz_complex is [Ny, Nx]. Bz_tilde is [B, Ny, Nx].
    # PyTorch broadcasting automatically applies the kernel to every map in the batch.
    Mz_tilde = K_Mz_complex * Bz_tilde

    # 5. Compute the Inverse Fourier Transform (IFFT)
    Mz_batch = torch.fft.ifft2(Mz_tilde)

    # The result is complex; Mz must be real, so we take the real part
    return Mz_batch.real

# ----------------------------------------------------------------------
# Example Usage: Batch Processing
# ----------------------------------------------------------------------

# Set up device: Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define physical parameters
LIFT_HEIGHT = 10e-6 # 10 um (z)
PIXEL_SIZE = 0.5e-6 # 0.5 um (dx)
BATCH_SIZE = 4      # Process 4 maps simultaneously
NY, NX = 128, 128

# Create a mock batch of BZ maps (4 maps with slightly different fields)
# Start with a base field (e.g., a simple Gaussian)
y, x = torch.meshgrid(torch.arange(NY), torch.arange(NX), indexing='ij')
center_y, center_x = NY // 2, NX // 2
sigma = 10.0
base_field = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

# Create the batch with different amplitudes
Bz_batch_example = torch.stack([
    base_field * 1e-3,              # Map 1: 1 mTesla max
    base_field * 0.5e-3,            # Map 2: 0.5 mTesla max
    base_field * 2e-3,              # Map 3: 2 mTesla max
    (base_field * 0.75e-3).roll(10, dims=0) # Map 4: shifted
]).to(DEVICE).to(torch.float32)

print(f"--- Starting Batched Mz Reconstruction ({BATCH_SIZE} maps) ---")
print(f"Input Shape: {Bz_batch_example.shape}")

# Perform the reconstruction
try:
    Mz_reconstructed_batch = fourier_inversion_magnetization_mz(
        Bz_batch_example, LIFT_HEIGHT, PIXEL_SIZE
    )
    
    print("Reconstruction successful.")
    print(f"Output Mz Shape: {Mz_reconstructed_batch.shape}")
    print(f"Max Mz for Map 1: {Mz_reconstructed_batch[0].max():.2f} A/m")
    print(f"Max Mz for Map 3: {Mz_reconstructed_batch[2].max():.2f} A/m")
    
except Exception as e:
    print(f"An error occurred during reconstruction: {e}")
