# Computational_Physics: A Python based scientific computing library

**A Library for useful computational physics implementations**

This repository contains efficient and high throughput implementations, usually in Python and GPU optimized, for different computations, solvers and algorithms in physics. Mostly tailored towards computational electromagnetics but many of the implementations are designed for more general applicability to other domains.

**List of implementations**

**Magnetostatic Stray Field solver for 2d thin films**: FFT enhanced stray field computation at a specified observation plane at height h, for a 2d magnetic thin film. The method is built using PyTorch to be fully differentiable and incorporates GPU optimization. Supports batch processing.

**Fourier based 2d Magnetization reconstruction**: Reconstructs the out of plane Mz component of a 2d magnetization from the measured Bz stray field component. Based on the paper "Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements" by Broadway et al in Physical Review Applied (2020). Uses PyTorch for GPU acceleration and supports batch processing.

**Geodesic computation on general curved manifold**: Computes geodesic paths on a general curved manifold by taking as input the metric tensor. Uses PyTorch for GPU acceleration. Shows an example with a spherical manifold.


