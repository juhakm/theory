"""
Classical Past and Quantum Future Simulation

This module models a temporal sequence of 2D grayscale images (representing an observer’s perspective)
as a 3D signal across time and space. It fits this signal using sinusoidal basis functions in
space and time, then reconstructs or extrapolates the sequence using:

1. A nonlinear sinusoidal model (amplitude and phase optimization)
2. A linear least-squares projection using a fixed frequency basis

Usage:
- Place image frames in 'simulations/paper2' named 'observer0.png', ..., 'observerN.png'
- Adjust num_images, t0, resolution, and frequency parameters to control fitting behavior
"""

import os
import numpy as np
from typing import List, Tuple
from PIL import Image
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# === Configuration ===
path = os.path.join("simulations", "paper2")
num_images: int = 8
image_paths: List[str] = [os.path.join(path, f"observer{i}.png") for i in range(num_images)]
num_freqs: int = 10  # Number of sinusoidal frequencies per axis
t0: int = 5  # Observer's "present": fit to frames [0, ..., t0-1]


def load_images(paths: List[str], size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """Load grayscale images, normalize to [0, 1], resize to common size, and return as (T, H, W) array."""
    frames: List[np.ndarray] = []
    for p in paths:
        img = Image.open(p).convert("L").resize(size, Image.BILINEAR)
        frame = np.array(img, dtype=np.float32) / 255.0
        frames.append(frame)
    return np.stack(frames, axis=0)


def make_basis_vectors(T: int, H: int, W: int, num_freqs: int = 4) -> np.ndarray:
    """
    Construct a matrix of flattened 3D sinusoidal basis vectors over time and 2D space.

    Returns:
        A (T*H*W, 2 * num_freqs^3) array
    """
    t = np.linspace(0, 1, T)
    x = np.linspace(0, 1, H)
    y = np.linspace(0, 1, W)
    tt, xx, yy = np.meshgrid(t, x, y, indexing='ij')

    components = []
    for i in range(num_freqs):
        for j in range(num_freqs):
            for k in range(num_freqs):
                sin_term = np.sin(2 * np.pi * (i * tt + j * xx + k * yy))
                cos_term = np.cos(2 * np.pi * (i * tt + j * xx + k * yy))
                components.append(sin_term.ravel())
                components.append(cos_term.ravel())
    return np.stack(components, axis=1)


def fit_linear_model(obs: np.ndarray, num_freqs: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a linear model to the observed tensor using sinusoidal basis."""
    T, H, W = obs.shape
    A = make_basis_vectors(T, H, W, num_freqs)
    b = obs.ravel()
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return params, A


def reconstruct_from_params(params: np.ndarray, A: np.ndarray, T: int, H: int, W: int) -> np.ndarray:
    """Reconstruct the 3D image volume from model parameters and basis matrix."""
    reconstruction = A @ params
    return reconstruction.reshape((T, H, W))


def make_basis(freqs_t: np.ndarray, freqs_x: np.ndarray, freqs_y: np.ndarray, shape: Tuple[int, int, int]) -> Tuple[List[Tuple[float, float, float]], np.ndarray, np.ndarray]:
    """Generate 3D sinusoidal bases (sin & cos) with phase-separated components."""
    T, H, W = shape
    t = np.linspace(0, 1, T, dtype=np.float32)
    x = np.linspace(0, 1, W, dtype=np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)
    tt, yy, xx = np.meshgrid(t, y, x, indexing='ij')

    triples = []
    sin_list = []
    cos_list = []

    for ft in freqs_t:
        for fx in freqs_x:
            for fy in freqs_y:
                phase = 2 * np.pi * (ft * tt + fx * xx + fy * yy)
                sin_list.append(np.sin(phase))
                cos_list.append(np.cos(phase))
                triples.append((ft, fx, fy))

    return triples, np.stack(sin_list), np.stack(cos_list)


# === Load and Prepare Data ===
obs: np.ndarray = load_images(image_paths, size=(8, 8))  # shape (T, H, W)
num_frames, height, width = obs.shape
obs_known = obs[:t0]

# === Build Nonlinear Basis ===
freqs = np.linspace(0, 3, num_freqs)
freq_triples, sin_basis, cos_basis = make_basis(freqs, freqs, freqs, obs.shape)
num_components = len(freq_triples)


def model(params: np.ndarray) -> np.ndarray:
    """Construct the predicted 3D signal volume from amplitude and phase parameters."""
    amps = params[:num_components]
    phases = params[num_components:]
    img = np.tensordot(amps * np.cos(phases), cos_basis, axes=(0, 0)) + \
          np.tensordot(amps * np.sin(phases), sin_basis, axes=(0, 0))
    return img.reshape(obs.shape)


def residuals(params: np.ndarray) -> np.ndarray:
    """Compute flattened residual vector for frames 0 to t0-1 (the observer’s known past)."""
    prediction = model(params)
    return (prediction[:t0] - obs_known).ravel()


# === Initial Guess and Optimization ===
np.random.seed(0)
init_amps = 0.1 * np.random.randn(num_components)
init_phases = 2 * np.pi * np.random.rand(num_components)
params0 = np.concatenate([init_amps, init_phases])

result = least_squares(residuals, params0, verbose=2, max_nfev=500)
Ψ_reconstructed_nl = model(result.x)

# === Optional: Construct Complex-Valued Ψ(x, y, t) ===
t = np.linspace(0, 1, num_frames, dtype=np.float32)
x = np.linspace(0, 1, width, dtype=np.float32)
y = np.linspace(0, 1, height, dtype=np.float32)
tt, yy, xx = np.meshgrid(t, y, x, indexing='ij')

amps = result.x[:num_components]
phases = result.x[num_components:]
Ψ = np.zeros((num_frames, height, width), dtype=np.complex128)
for i, (ft, fx, fy) in enumerate(freq_triples):
    phase = 2 * np.pi * (ft * tt + fx * xx + fy * yy) + phases[i]
    Ψ += amps[i] * np.exp(1j * phase)

# === Linear Fit to Past, Extrapolated to Full Sequence ===
params_lin, A_past = fit_linear_model(obs_known, num_freqs=num_freqs)
A_full = make_basis_vectors(*obs.shape, num_freqs=num_freqs)
Ψ_reconstructed_lin = reconstruct_from_params(params_lin, A_full, *obs.shape)

# === Evaluate Prediction Accuracy ===
actual_future = obs[t0:]
predicted_future_nl = np.clip(Ψ_reconstructed_nl[t0:], 0.0, 1.0)
predicted_future_lin = np.clip(Ψ_reconstructed_lin[t0:], 0.0, 1.0)

# === Animation (Linear vs Actual) ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

def update(frame_idx: int) -> None:
    ax1.clear()
    ax2.clear()
    ax1.set_title(f"Actual t={t0 + frame_idx}")
    ax2.set_title(f"Linear Predicted t={t0 + frame_idx}")
    ax1.imshow(actual_future[frame_idx], cmap="gray")
    ax2.imshow(predicted_future_lin[frame_idx], cmap="viridis")
    for ax in [ax1, ax2]:
        ax.axis('off')

ani = animation.FuncAnimation(fig, update, frames=num_frames - t0, interval=500)
plt.tight_layout()
plt.show()

# === Error Reporting ===
print("\nPrediction errors:")
for i in range(num_frames - t0):
    diff = np.abs(actual_future[i] - predicted_future_lin[i])
    print(f"t={t0+i}: max |obs - pred| = {np.max(diff):.4f}, mean = {np.mean(diff):.4f}")
