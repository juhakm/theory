"""
Classical Past and Quantum Future Simulation

This module models a temporal sequence of 2D grayscale images (representing an observer’s perspective)
as a 3D signal across time and space. It fits this signal using sinusoidal basis functions in
space and time, then reconstructs or extrapolates the sequence using:

1. A nonlinear sinusoidal model (amplitude and phase optimization)
2. A linear least-squares projection using a fixed frequency basis
3. Smooth interpolation of the classical past parameters (amplitude and phase)
4. Quantum evolution extrapolation into the future
5. Saving a combined animation of past interpolation + future extrapolation

Usage:
- Place image frames in 'simulations/paper2' named 'observer0.png', ..., 'observerN.png'
- Adjust num_images, t0, resolution, and frequency parameters to control fitting behavior
"""

import os
import numpy as np
from typing import List, Tuple
from PIL import Image
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from numpy import unwrap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

# === Configuration ===
path = os.path.join("simulations", "paper2")
num_images: int = 8
image_paths: List[str] = [os.path.join(path, f"observer{i}.png") for i in range(num_images)]
num_freqs: int = 10  # Number of sinusoidal frequencies per axis
t0: int = 5  # Observer's "present": fit to frames [0, ..., t0-1]
quantum_speed: float = 0.1  # 👈 1.0 = normal, <1 = slower, >1 = faster
observer_speed :float = 0.1 # for generate_synthetic_observer_images
observer_resolution : tuple[int, int] = (16,16)


def load_images(paths: List[str], size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """Load grayscale images, normalize to [0, 1], resize to common size, and return as (T, H, W) array."""
    frames: List[np.ndarray] = []
    for p in paths:
        img = Image.open(p).convert("L").resize(size, Image.BILINEAR)
        frame = np.array(img, dtype=np.float32) / 255.0
        frames.append(frame)
    return np.stack(frames, axis=0)


import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Tuple


def generate_synthetic_observer_images(
    num_frames: int = 32,
    size: Tuple[int, int] = (32, 32),
    radius: float = 0.3,
    fade_tail: bool = False
) -> np.ndarray:
    """
    Generate a sequence of synthetic grayscale images showing a shaded sphere circulating in 2D.

    Args:
        num_frames: number of images in the sequence (T)
        size: (H, W) size of each image
        radius: radius of the sphere as fraction of image width
        fade_tail: if True, applies a fading motion trail (optional)

    Returns:
        Array of shape (T, H, W) normalized to [0, 1]
    """
    H, W = size
    center_x = W / 2
    center_y = H / 2
    r_px = int(radius * min(W, H))

    frames: List[np.ndarray] = []

    for t in range(num_frames):
        angle = observer_speed * 2 * np.pi * t / num_frames
        x = int(center_x + (W // 3) * np.cos(angle))
        y = int(center_y + (H // 3) * np.sin(angle))

        img = Image.new("L", (W, H), color=0)
        draw = ImageDraw.Draw(img)

        # Shaded circle using radial gradient
        for i in range(r_px, 0, -1):
            gray = int(255 * (1 - i / r_px)**2)  # quadratic brightness profile
            draw.ellipse(
                [x - i, y - i, x + i, y + i],
                fill=gray
            )

        # Optional trail effect
        if fade_tail:
            blur_strength = int(radius * 10)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_strength))

        # Normalize and convert to float32
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
#obs: np.ndarray = load_images(image_paths, size=observer_resolution)  # shape (T, H, W)
obs = generate_synthetic_observer_images(num_frames=5, size=observer_resolution)


import matplotlib.pyplot as plt

# Assume obs is (T, H, W) numpy array of grayscale frames normalized [0..1]

def show_frames(frames: np.ndarray, delay=500):
    """
    Display frames one by one in a blocking loop.
    
    Args:
        frames: (T, H, W) array of frames (float values 0..1)
        delay: pause time in milliseconds between frames
    """
    T = frames.shape[0]
    plt.figure(figsize=(4,4))
    for t in range(T):
        plt.imshow(frames[t], cmap='gray', vmin=0, vmax=1)
        plt.title(f"Frame {t}")
        plt.axis('off')
        plt.pause(delay / 1000)  # convert ms to seconds
        plt.clf()
    plt.close()

# Example usage
show_frames(obs, delay=500)  # half-second delay between frames


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


def get_single_frame_basis(time_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sin and cos basis vectors for a single time slice.

    Returns arrays of shape (num_components, H*W)
    """
    sin_slice = sin_basis[:, time_idx].reshape(num_components, -1)
    cos_slice = cos_basis[:, time_idx].reshape(num_components, -1)
    return sin_slice, cos_slice



# === Smooth Past Interpolation and Quantum Future Extrapolation ===
def fit_params_to_frame(frame: np.ndarray, time_idx: int = 0) -> np.ndarray:
    """
    Fit nonlinear model parameters (amps + phases) to a single 2D frame at one time.

    Args:
        frame: 2D image array (H, W)
        time_idx: which time slice basis to use

    Returns:
        fitted parameter vector (amps + phases)
    """
    sin_slice, cos_slice = get_single_frame_basis(time_idx)

    def res(p):
        amps = p[:num_components]
        phases = p[num_components:]
        img = np.tensordot(amps * np.cos(phases), cos_slice, axes=(0, 0)) + \
              np.tensordot(amps * np.sin(phases), sin_slice, axes=(0, 0))
        return (img - frame.ravel()).ravel()

    p0 = np.concatenate([0.1 * np.random.randn(num_components), 2 * np.pi * np.random.rand(num_components)])
    res_fit = least_squares(res, p0, max_nfev=300, verbose=0)
    return res_fit.x



# Fit parameter trajectory on past frames t=0..t0-1
past_params = np.stack([fit_params_to_frame(obs[t], time_idx=t) for t in range(t0)], axis=0)


# Separate amplitudes and unwrap phases over time for smooth interpolation
amps_past = past_params[:, :num_components]
phases_past = unwrap(past_params[:, num_components:], axis=0)

# Create cubic splines for each parameter over the past time
time_past = np.arange(t0)
amps_splines = [CubicSpline(time_past, amps_past[:, i]) for i in range(num_components)]
phases_splines = [CubicSpline(time_past, phases_past[:, i]) for i in range(num_components)]


def interp_past_params(t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate amplitude and phase parameters smoothly for continuous past time t.

    Args:
        t (float): time within past [0, t0-1]

    Returns:
        Tuple[np.ndarray, np.ndarray]: amplitudes and phases at time t
    """
    amps_t = np.array([spline(t) for spline in amps_splines])
    phases_t = np.array([spline(t) for spline in phases_splines])
    return amps_t, phases_t


freqs_t = np.array([ft for (ft, fx, fy) in freq_triples])



def quantum_evolve(amps: np.ndarray, phases: np.ndarray, dt: float, speed: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve the wavefunction parameters forward in time by dt via phase advance.

    Args:
        amps (np.ndarray): amplitudes at initial time
        phases (np.ndarray): phase angles at initial time
        dt (float): time delta
        speed (float): speed multiplier for evolution (default 1.0)

    Returns:
        Updated amplitudes and phases
    """
    evolved_phases = phases + 2 * np.pi * freqs_t * dt * speed
    return amps, evolved_phases



def reconstruct_frame(amps: np.ndarray, phases: np.ndarray, time_idx: int = 0) -> np.ndarray:
    """
    Reconstruct a single 2D image frame from amplitudes and phases parameters.

    Args:
        amps (np.ndarray): amplitude coefficients
        phases (np.ndarray): phase coefficients

    Returns:
        np.ndarray: reconstructed 2D image frame of shape (height, width)
    """
    sin_slice, cos_slice = get_single_frame_basis(time_idx)
    img = np.tensordot(amps * np.cos(phases), cos_slice, axes=(0, 0)) + \
          np.tensordot(amps * np.sin(phases), sin_slice, axes=(0, 0))
    return img.reshape((height, width))


# Prepare timeline for the full animation: past interpolation + future extrapolation
t_future_start = t0 - 1
t_final = num_frames + 20  # extend 20 frames into the future beyond known data

# Oversample time for smooth animation frames
frame_times = np.linspace(0, t_final, num=int((t_final + 1) * 5))

# Generate frames by interpolating past params or evolving future params
frames = []
for t in frame_times:
    if t <= t_future_start:
        amps_t, phases_t = interp_past_params(t)
    else:
        dt = t - t_future_start
        amps0, phases0 = interp_past_params(t_future_start)
        amps_t, phases_t = quantum_evolve(amps0, phases0, dt, speed=quantum_speed)

    frame_img = reconstruct_frame(amps_t, phases_t, time_idx=0)
    frame_img = np.clip(frame_img, 0, 1)
    frames.append(frame_img)
frames = np.array(frames)


# === Save combined animation of past and future ===

fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap='viridis', vmin=0, vmax=1)

# Correctly store the returned Text object
title_obj = ax.set_title(f"t = {frame_times[0]:.2f}")

def update_anim(i):
    im.set_array(frames[i])
    title_obj.set_text(f"t = {frame_times[i]:.2f}")
    return [im, title_obj]  # Return title_obj instead of title string

ani = animation.FuncAnimation(fig, update_anim, frames=len(frames), interval=40, blit=False)


# Save as mp4 video file using ffmpeg writer (requires ffmpeg installed)
save_path = os.path.join(path, "observer_past_future.mp4")
Writer = animation.FFMpegWriter
writer = Writer(fps=25, metadata=dict(artist='Simulation'), bitrate=1800)
ani.save(save_path, writer=writer)

print(f"Saved animation to {save_path}")

plt.show()


# === Report prediction errors on future frames ===
print("\nPrediction errors (Linear model vs Actual future frames):")
for i in range(num_frames - t0):
    diff = np.abs(actual_future[i] - predicted_future_lin[i])
    print(f"t={t0 + i}: max |obs - pred| = {np.max(diff):.4f}, mean = {np.mean(diff):.4f}")
