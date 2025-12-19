"""
Schrödinger Equation Training Data Generator

This generates thousands of potential functions and their corresponding
eigenstates for training a Fourier Neural Operator!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import simpson
import h5py
from tqdm import tqdm
from pathlib import Path


class SchrodingerDataGenerator:
    """
    Generates training data for neural operators by solving the 1D
    time-independent Schrödinger equation for various potentials.
    """
    
    def __init__(
        self,
        n_points=256,
        x_min=-10.0,
        x_max=10.0,
        n_eigenstates=4,
        hbar=1.0,
        mass=1.0
    ):
        """
        Args:
            n_points: Grid resolution. Higher = more accurate, slower.
            x_min, x_max: Domain bounds. 
            n_eigenstates: How many energy levels to compute. More = slower.
            hbar, mass: Physical constants. Normalized and set to 1 to avoid units
        """
        self.n_points = n_points
        self.x = np.linspace(x_min, x_max, n_points)
        self.dx = self.x[1] - self.x[0]
        self.n_eigenstates = n_eigenstates
        self.hbar = hbar
        self.mass = mass
        
    def solve_schrodinger(self, V):
        """
        Solve -ℏ²/2m d²ψ/dx² + V(x)ψ = Eψ using finite differences.
        
        We discretize the Laplacian as a tridiagonal matrix 
        
        Args:
            V: Potential function on grid (n_points,)
            
        Returns:
            energies: First n_eigenstates eigenvalues
            wavefunctions: Corresponding eigenfunctions (normalized)
        """
        # Kinetic energy: -ℏ²/2m d²/dx² → tridiagonal matrix
        # Second derivative stencil: [1, -2, 1] / dx²
        coeff = -self.hbar**2 / (2 * self.mass * self.dx**2)
        
        # Diagonal: -2 * coeff + V(x)
        diagonal = -2 * coeff * np.ones(self.n_points) + V
        
        # Off-diagonal: coeff
        off_diagonal = coeff * np.ones(self.n_points - 1)
        
        # Solve the eigenvalue problem
        # eigh_tridiagonal is faster than eigh for tridiagonal matrices
        
        energies, wavefunctions = eigh_tridiagonal(
            diagonal, 
            off_diagonal,
            select='i',  # Select by index
            select_range=(0, self.n_eigenstates - 1)
        )
        
        # Normalize wavefunctions: ∫|ψ|² dx = 1
        # Because square of wave function is a probability density
        
        for i in range(self.n_eigenstates):
            norm = np.sqrt(simpson(wavefunctions[:, i]**2, self.x))
            wavefunctions[:, i] /= norm
            
            # Enforce phase convention: max(|ψ|) has positive value
            # This helps with the sign ambiguity 
            idx_max = np.argmax(np.abs(wavefunctions[:, i]))
            if wavefunctions[idx_max, i] < 0:
                wavefunctions[:, i] *= -1
        
        return energies, wavefunctions.T  # Shape: (n_eigenstates, n_points)
    
    def generate_random_smooth_potential(self, n_modes=8, v_scale=10.0):
        """
        Generate a random smooth potential using Fourier series.
        
        V(x) = Σ (a_n cos(nπx/L) + b_n sin(nπx/L))
        
        This gives you potentials that are:
        - Smooth (differentiable everywhere)
        - Diverse (random coefficients)
        - Reasonably well behaved (bounded by v_scale)
        
        Args:
            n_modes: Number of Fourier modes. More = wigglier potentials.
            v_scale: Amplitude scale. Bigger = deeper wells, higher barriers.
        """
        L = self.x[-1] - self.x[0]
        V = np.zeros_like(self.x)
        
        for n in range(1, n_modes + 1):
            a_n = np.random.randn() * v_scale / n  # Decay with frequency
            b_n = np.random.randn() * v_scale / n
            V += a_n * np.cos(n * np.pi * self.x / L)
            V += b_n * np.sin(n * np.pi * self.x / L)
        
        # Add a random offset so the potential isn't always centered at 0
        V += np.random.randn() * v_scale * 0.5
        
        return V
    
    def generate_double_well(self, a=1.0, b=1.0):
        """Double well potential: V(x) = a(x² - b)²"""
        return a * (self.x**2 - b)**2
    
    def generate_polynomial(self, max_degree=4, scale=5.0):
        """Random polynomial potential: V(x) = Σ c_n x^n"""
        degree = np.random.randint(2, max_degree + 1)
        coeffs = np.random.randn(degree + 1) * scale / (np.arange(degree + 1) + 1)
        V = np.zeros_like(self.x)
        for n, c in enumerate(coeffs):
            V += c * self.x**n
        return V
    
    def generate_gaussian_wells(self, n_wells=None, depth_scale=10.0, width_scale=2.0):
        """Random combination of Gaussian wells/barriers"""
        if n_wells is None:
            n_wells = np.random.randint(1, 5)
        
        V = np.zeros_like(self.x)
        for _ in range(n_wells):
            center = np.random.uniform(self.x[0] + 2, self.x[-1] - 2)
            depth = np.random.randn() * depth_scale
            width = np.random.uniform(0.5, width_scale)
            V += depth * np.exp(-((self.x - center) / width)**2)
        
        return V
    
    def generate_griffiths_classics(self):
        """
        From Griffiths Chapter 2.
        Use these for validation, not training.
        """
        classics = {}
        
        # Infinite square well (approximated with very steep walls)
        V_well = np.where(
            (self.x > -5) & (self.x < 5),
            0.0,
            1000.0  # "Infinite" as in very very high barrier
        )
        classics['infinite_square_well'] = V_well
        
        # Harmonic oscillator: V(x) = ½kx²
        k = 1.0
        classics['harmonic_oscillator'] = 0.5 * k * self.x**2
        
        # Finite square well
        depth = 20.0
        V_finite = np.where(
            (self.x > -3) & (self.x < 3),
            -depth,
            0.0
        )
        classics['finite_square_well'] = V_finite
        
        # Double well
        classics['double_well'] = self.generate_double_well(a=0.5, b=4.0)
        
        # Quartic potential
        classics['quartic'] = 0.1 * self.x**4
        
        return classics
    
    def generate_dataset(
        self,
        n_samples=5000,
        save_path='schrodinger_data.h5',
        include_classics=True
    ):
        """
        Generate the full training dataset.
        
        Args:
            n_samples: Number of random potentials to generate
            save_path: Where to save the HDF5 file
            include_classics: Whether to save Griffiths examples separately
        """
        print(f"Generating {n_samples} Schrödinger equation solutions...")
        print(f"Grid points: {self.n_points}, Eigenstates: {self.n_eigenstates}")
        print("Enjoy.\n")
        
        # Preallocate arrays
        potentials = np.zeros((n_samples, self.n_points))
        energies = np.zeros((n_samples, self.n_eigenstates))
        wavefunctions = np.zeros((n_samples, self.n_eigenstates, self.n_points))
        
        # Mix of generation strategies for diversity
        strategies = [
            ('fourier', 0.4),      # 40% Fourier series
            ('polynomial', 0.2),   # 20% polynomials  
            ('gaussian', 0.2),     # 20% Gaussian combinations
            ('double_well', 0.2),  # 20% double wells
        ]
        
        for i in tqdm(range(n_samples)):
            # Choose generation strategy
            strategy = np.random.choice(
                [s[0] for s in strategies],
                p=[s[1] for s in strategies]
            )
            
            # Generate potential
            if strategy == 'fourier':
                V = self.generate_random_smooth_potential()
            elif strategy == 'polynomial':
                V = self.generate_polynomial()
            elif strategy == 'gaussian':
                V = self.generate_gaussian_wells()
            else:  # double_well
                a = np.random.uniform(0.1, 2.0)
                b = np.random.uniform(1.0, 5.0)
                V = self.generate_double_well(a, b)
            
            # Solve
            try:
                E, psi = self.solve_schrodinger(V)
                potentials[i] = V
                energies[i] = E
                wavefunctions[i] = psi
            except Exception as e:
                print(f"\nWarning: Failed on sample {i}: {e}")
                print("Replacing with harmonic oscillator because I'm not dealing with your edge cases.")
                V = 0.5 * self.x**2
                E, psi = self.solve_schrodinger(V)
                potentials[i] = V
                energies[i] = E
                wavefunctions[i] = psi
        
        # Save to HDF5 (training dataset)
        print(f"\nSaving to {save_path}...")
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('potentials', data=potentials)
            f.create_dataset('energies', data=energies)
            f.create_dataset('wavefunctions', data=wavefunctions)
            f.create_dataset('x_grid', data=self.x)
            
            # Metadata for convenience
            f.attrs['n_samples'] = n_samples
            f.attrs['n_points'] = self.n_points
            f.attrs['n_eigenstates'] = self.n_eigenstates
            f.attrs['x_min'] = self.x[0]
            f.attrs['x_max'] = self.x[-1]
            f.attrs['hbar'] = self.hbar
            f.attrs['mass'] = self.mass
            
            # Save Griffiths classics as separate group for validation
            if include_classics:
                classics = self.generate_griffiths_classics()
                classic_group = f.create_group('validation')
                
                for name, V in classics.items():
                    E, psi = self.solve_schrodinger(V)
                    classic_subgroup = classic_group.create_group(name)
                    classic_subgroup.create_dataset('potential', data=V)
                    classic_subgroup.create_dataset('energies', data=E)
                    classic_subgroup.create_dataset('wavefunctions', data=psi)
        
        print(f"✓ Dataset saved successfully!")
        print(f"  File size: {Path(save_path).stat().st_size / 1e6:.1f} MB")
        print(f"  Training samples: {n_samples}")
        if include_classics:
            print(f"  Validation examples: {len(classics)} Griffiths classics")
        
        return potentials, energies, wavefunctions
    
    def visualize_samples(self, potentials, energies, wavefunctions, n_show=4):
        """
        Plot some random samples to verify they look reasonable
        """
        indices = np.random.choice(len(potentials), n_show, replace=False)
        
        fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
        
        for i, idx in enumerate(indices):
            # Plot potential
            axes[0, i].plot(self.x, potentials[idx], 'k-', linewidth=2)
            axes[0, i].set_ylabel('V(x)')
            axes[0, i].set_title(f'Sample {idx}')
            axes[0, i].grid(alpha=0.3)
            
            # Plot energy levels and wavefunctions
            for n in range(self.n_eigenstates):
                E = energies[idx, n]
                psi = wavefunctions[idx, n]
                
                # Plot energy level
                axes[0, i].axhline(E, color=f'C{n}', linestyle='--', alpha=0.5)
                
                # Plot wavefunction (shifted by energy for visualization)
                axes[1, i].plot(
                    self.x, 
                    psi * 5 + E,  # Scale and shift for visibility
                    label=f'n={n}, E={E:.2f}',
                    color=f'C{n}'
                )
            
            axes[1, i].plot(self.x, potentials[idx], 'k-', alpha=0.3, linewidth=1)
            axes[1, i].set_xlabel('x')
            axes[1, i].set_ylabel('ψ(x) + E')
            axes[1, i].legend(fontsize=8)
            axes[1, i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sample_potentials.png', dpi=150, bbox_inches='tight')
        print("Sample visualization saved to sample_potentials.png")
        plt.show()


def main():
    """
    Run this
    """
    # Initialize generator
    generator = SchrodingerDataGenerator(
        n_points=256,      # Good balance of resolution vs speed
        x_min=-10.0,
        x_max=10.0,
        n_eigenstates=4,   # Ground state + first 3 excited states
    )
    
    # Generate training data
    potentials, energies, wavefunctions = generator.generate_dataset(
        n_samples=5000,
        save_path='schrodinger_training_data.h5',
        include_classics=True
    )
    
    # Visualize a few samples
    generator.visualize_samples(potentials, energies, wavefunctions, n_show=4)
    
    print("\n" + "="*70)
    print("Data generation complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Verify the visualizations look reasonable")
    print("2. Train your FNO on this data")
    

if __name__ == "__main__":
    main()
