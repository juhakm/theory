import numpy as np
from PIL import Image
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# === Configuration ===
path = os.path.join("simulations", "paper2")
num_images = 8
image_paths = [os.path.join(path, f"observer{i}.png") for i in range(num_images)]
num_freqs = 10

# === Load grayscale images ===
def load_images(paths):
    frames = []
    for p in paths:
        img = Image.open(p).convert("L")
        frame = np.array(img, dtype=np.float32) / 255.0
        frames.append(frame)
    return np.stack(frames, axis=0)

obs = load_images(image_paths)  # shape: (T, H, W)
num_frames, height, width = obs.shape

# === Define fitting boundary (observer's present) ===
t0 = 5  # Observer has seen frames [0, ..., t0-1]
obs_known = obs[:t0]

# === Build 3D sinusoidal basis ===
def make_basis(freqs_t, freqs_x, freqs_y, shape):
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

freqs = np.linspace(0, 3, num_freqs)
freq_triples, sin_basis, cos_basis = make_basis(freqs, freqs, freqs, obs.shape)
num_components = len(freq_triples)

# === Optimized model function ===
def model(params):
    amps = params[:num_components]
    phases = params[num_components:]
    img = np.tensordot(amps * np.cos(phases), cos_basis, axes=(0, 0)) + \
          np.tensordot(amps * np.sin(phases), sin_basis, axes=(0, 0))
    return img.reshape(obs.shape)

# === Residuals for fitting only past ===
def residuals(params):
    prediction = model(params)
    return (prediction[:t0] - obs_known).ravel()

# === Initial guess ===
np.random.seed(0)
init_amps = 0.1 * np.random.randn(num_components)
init_phases = 2 * np.pi * np.random.rand(num_components)
params0 = np.concatenate([init_amps, init_phases])

# === Fit only to known past ===
result = least_squares(residuals, params0, verbose=2, max_nfev=500)
Ψ_reconstructed = model(result.x)

# === Build complex wavefunction Ψ(x,y,t) ===
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

# === Compare real vs predicted future ===
actual_future = obs[t0:]
predicted_future = np.abs(Ψ[t0:]) ** 2
predicted_future /= np.max(predicted_future, axis=(1, 2), keepdims=True)

# === Animate actual vs predicted ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

def update(frame_idx):
    ax1.clear()
    ax2.clear()
    ax1.set_title(f"Actual Observer t={t0 + frame_idx}")
    ax2.set_title(f"Predicted |Ψ|² t={t0 + frame_idx}")

    ax1.imshow(actual_future[frame_idx], cmap="gray")
    ax2.imshow(predicted_future[frame_idx], cmap="viridis")
    for ax in [ax1, ax2]:
        ax.axis('off')

ani = animation.FuncAnimation(fig, update, frames=num_frames - t0, interval=500)
plt.tight_layout()
plt.show()

# === Optional: error metric ===
for i in range(num_frames - t0):
    diff = np.abs(actual_future[i] - predicted_future[i])
    print(f"t={t0+i}: max |obs - |Ψ|²| = {np.max(diff):.4f}, mean = {np.mean(diff):.4f}")
