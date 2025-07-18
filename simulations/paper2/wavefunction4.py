import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# === Configuration ===
path = os.path.join("simulations", "paper2")
image_path = os.path.join(path, "observer.png")
cache_path = os.path.join(path, "observer.npz")
num_freqs = 10  # Number of frequencies to fit in each dimension (total = num_freqs^2)

# === Load grayscale image ===
image = Image.open(image_path).convert("L")
obs = np.array(image, dtype=np.float32) / 255.0
height, width = obs.shape

# === Generate 2D sinusoids ===
def make_basis(freqs_x, freqs_y, shape):
    basis = []
    x = np.linspace(0, 1, shape[1])
    y = np.linspace(0, 1, shape[0])
    xx, yy = np.meshgrid(x, y)
    for fx in freqs_x:
        for fy in freqs_y:
            sin_part = np.sin(2 * np.pi * (fx * xx + fy * yy))
            cos_part = np.cos(2 * np.pi * (fx * xx + fy * yy))
            basis.append(((fx, fy), sin_part, cos_part))
    return basis

freqs = np.linspace(0, 5, num_freqs)  # Low frequencies only
basis = make_basis(freqs, freqs, obs.shape)

# === Flattened fitting function ===
def model(params):
    amps = params[: len(basis)]
    phases = params[len(basis) :]
    img = np.zeros_like(obs)
    for i, ((fx, fy), sin_part, cos_part) in enumerate(basis):
        img += amps[i] * (np.cos(phases[i]) * cos_part + np.sin(phases[i]) * sin_part)
    return img.ravel()

# === Objective function ===
def residuals(params):
    return model(params) - obs.ravel()

# === Initial guess: small random amplitudes and phases ===
np.random.seed(0)
init_amps = 0.1 * np.random.randn(len(basis))
init_phases = 2 * np.pi * np.random.rand(len(basis))
params0 = np.concatenate([init_amps, init_phases])

# === Fit! ===
result = least_squares(residuals, params0, verbose=2, max_nfev=500)

# === Reconstruct fitted image ===
fitted = model(result.x).reshape(obs.shape)

# === Visualize original and fitted ===
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(obs, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Fitted")
plt.imshow(fitted, cmap="gray")

plt.show()

# === Prepare extended grid for wavefunction evolution ===
extrap_height = 64
extrap_width = 64

pad_x = (extrap_width - width) // 2
pad_y = (extrap_height - height) // 2

x_ext = np.linspace(0 - pad_x / width, 1 + pad_x / width, extrap_width)
y_ext = np.linspace(0 - pad_y / height, 1 + pad_y / height, extrap_height)
xx_ext, yy_ext = np.meshgrid(x_ext, y_ext)

# === Extract fitted amplitudes and phases ===
amps = result.x[: len(basis)]
phases = result.x[len(basis) :]

# === Build complex initial wavefunction Psi(x,y,t=0) ===
Psi_t0 = np.zeros((extrap_height, extrap_width), dtype=np.complex128)
for i, ((fx, fy), _, _) in enumerate(basis):
    component = amps[i] * np.exp(1j * (2 * np.pi * (fx * xx_ext + fy * yy_ext) + phases[i]))
    Psi_t0 += component

# === Time evolution operator ===
def time_evolve(Psi, dt, hbar=1.0, mass=1.0, dx=1.0, dy=1.0):
    ny, nx = Psi.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_squared = KX**2 + KY**2

    Psi_k = np.fft.fft2(Psi)
    phase_shift = np.exp(-1j * (hbar * K_squared / (2 * mass)) * dt)
    Psi_k_new = Psi_k * phase_shift
    Psi_new = np.fft.ifft2(Psi_k_new)
    return Psi_new

# === Animate evolution ===
num_frames = 100
dt = 0.1  # time step

fig, ax = plt.subplots()

Psi = Psi_t0.copy()

# real valued wavefunction slide
def update_real_valued(frame):
    global Psi
    Psi = time_evolve(Psi, dt)
    ax.clear()
    # Show real part instead of amplitude for direct comparison
    ax.imshow(np.real(Psi), cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"Time step {frame}")
    ax.axis('off')


def update_complex(frame):
    global Psi
    Psi = time_evolve(Psi, dt)
    ax.clear()
    ax.imshow(np.abs(Psi), cmap="gray")
    ax.set_title(f"Time step {frame}")
    ax.axis('off')

def update(frame):
    global Psi
    Psi = time_evolve(Psi, dt)
    ax.clear()
    # Plot probability density |Psi|^2, normalized for better contrast
    prob_density = np.abs(Psi)**2
    prob_density /= prob_density.max()  # Normalize to 1
    ax.imshow(prob_density, cmap="viridis")  # 'viridis' or 'plasma' for nice color maps
    ax.set_title(f"Probability density |Ψ|² at time {frame}")
    ax.axis('off')


ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Observer")
plt.imshow(obs, cmap="gray")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Amplitude of Psi_t0")
plt.imshow(np.abs(Psi_t0), cmap="gray")
plt.axis('off')

plt.show()

# Optionally print the max absolute difference:
diff = np.max(np.abs(obs - np.abs(Psi_t0[:height, :width])))
print(f"Max abs difference between original and Psi_t0 amplitude: {diff:.6f}")

plt.show()
