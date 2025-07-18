import numpy as np
from PIL import Image
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# === Configuration ===
path = os.path.join("simulations", "paper2")
num_images = 8
image_paths = [os.paJth.join(path, f"observer{i}.png") for i in range(num_images)]
num_freqs = 10  # Number of frequencies in each spatial dimension

# === Load grayscale images as a 3D volume: time x height x width ===
def load_images(paths):
    frames = []
    for p in paths:
        image = Image.open(p).convert("L")
        frame = np.array(image, dtype=np.float32) / 255.0
        frames.append(frame)
    return np.stack(frames, axis=0)

obs = load_images(image_paths)  # shape: (T, H, W)
num_frames, height, width = obs.shape

# === Generate 3D sinusoids: time + x + y ===
def make_basis(freqs_t, freqs_x, freqs_y, shape):
    T, H, W = shape
    basis = []
    t = np.linspace(0, 1, T)
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    tt, yy, xx = np.meshgrid(t, y, x, indexing='ij')
    for ft in freqs_t:
        for fx in freqs_x:
            for fy in freqs_y:
                sin_part = np.sin(2 * np.pi * (ft * tt + fx * xx + fy * yy))
                cos_part = np.cos(2 * np.pi * (ft * tt + fx * xx + fy * yy))
                basis.append(((ft, fx, fy), sin_part, cos_part))
    return basis

freqs = np.linspace(0, 3, num_freqs)
basis = make_basis(freqs, freqs, freqs, obs.shape)

# === Flattened fitting function ===
def model(params):
    amps = params[: len(basis)]
    phases = params[len(basis):]
    img = np.zeros_like(obs)
    for i, ((ft, fx, fy), sin_part, cos_part) in enumerate(basis):
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

# === Visualize one slice for comparison ===
plt.subplot(1, 2, 1)
plt.title("Original Frame 0")
plt.imshow(obs[0], cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Fitted Frame 0")
plt.imshow(fitted[0], cmap="gray")

plt.show()

# === Build complex wavefunction Psi(x,y,t) ===
x = np.linspace(0, 1, width)
y = np.linspace(0, 1, height)
t = np.linspace(0, 1, num_frames)
tt, yy, xx = np.meshgrid(t, y, x, indexing='ij')

amps = result.x[: len(basis)]
phases = result.x[len(basis):]

Psi_t = np.zeros((num_frames, height, width), dtype=np.complex128)
for i, ((ft, fx, fy), _, _) in enumerate(basis):
    component = amps[i] * np.exp(1j * (2 * np.pi * (ft * tt + fx * xx + fy * yy) + phases[i]))
    Psi_t += component

# === Animate evolution as probability density |Psi|^2 ===
fig, ax = plt.subplots()

def update(frame):
    prob_density = np.abs(Psi_t[frame])**2
    prob_density /= prob_density.max()
    ax.clear()
    ax.imshow(prob_density, cmap="viridis")
    ax.set_title(f"|Ψ|² at t={frame}")
    ax.axis('off')

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=400)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Observer t=0")
plt.imshow(obs[0], cmap="gray")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("|Ψ(x,y,t=0)|")
plt.imshow(np.abs(Psi_t[0]), cmap="gray")
plt.axis('off')

plt.show()

# Print difference
diff = np.max(np.abs(obs[0] - np.abs(Psi_t[0])))
print(f"Max abs difference (frame 0): {diff:.6f}")
