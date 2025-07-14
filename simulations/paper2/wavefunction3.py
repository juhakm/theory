import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from pathlib import Path
import matplotlib.pyplot as plt
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
    """Returns list of (fx, fy, basis_image)"""
    basis = []
    x = np.linspace(0, 1, shape[1])
    y = np.linspace(0, 1, shape[0])
    xx, yy = np.meshgrid(x, y)
    for fx in freqs_x:
        for fy in freqs_y:
            phase = 0  # Placeholder, actual phases will be learned
            sin_part = np.sin(2 * np.pi * (fx * xx + fy * yy))
            cos_part = np.cos(2 * np.pi * (fx * xx + fy * yy))
            basis.append(((fx, fy), sin_part, cos_part))
    return basis


def get_basis_with_cache(freqs_x, freqs_y, shape, image_path, cache_path):
    image_path = Path(image_path)
    cache_path = Path(cache_path)

    image_mtime = os.path.getmtime(image_path)
    if cache_path.exists():
        cache_mtime = os.path.getmtime(cache_path)
        if cache_mtime >= image_mtime:
            print("Loading cached basis functions.")
            data = np.load(cache_path, allow_pickle=True)
            return data["basis"].tolist()

    print("Regenerating basis functions.")
    basis = make_basis(freqs_x, freqs_y, shape)
    np.savez_compressed(cache_path, basis=np.array(basis, dtype=object))
    return basis


freqs = np.linspace(0, 5, num_freqs)  # Low frequencies only
# basis = get_basis_with_cache(freqs, freqs, obs.shape, image_path, cache_path)
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

# === Visualize ===
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(obs, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Fitted")
plt.imshow(fitted, cmap="gray")

plt.show()
print("done")

# === Extrapolation config ===

# === Extrapolation config ===
extrap_height = 64
extrap_width = 64

# === Coordinate grid: DO NOT scale frequency domain ===
# Keep physical spacing fixed; extend coordinate domain.
# So if original spanned [0, width) and [0, height), new one spans [-(W-w)//2, W + (W-w)//2)
# === Coordinate grid: DO NOT scale frequency domain ===
pad_x = (extrap_width - width) // 2
pad_y = (extrap_height - height) // 2

# x_ext = np.linspace(-pad_x / width, (width + pad_x) / width, extrap_width)
# y_ext = np.linspace(-pad_y / height, (height + pad_y) / height, extrap_height)

x_ext = np.linspace(0 - pad_x / width, 1 + pad_x / width, extrap_width)
y_ext = np.linspace(0 - pad_y / height, 1 + pad_y / height, extrap_height)

xx_ext, yy_ext = np.meshgrid(x_ext, y_ext)


# === Reconstruct wavefunction on extended grid ===
amps = result.x[: len(basis)]
phases = result.x[len(basis) :]

extrap_img = np.zeros((extrap_height, extrap_width))

for i, ((fx, fy), _, _) in enumerate(basis):
    component = np.cos(phases[i]) * np.cos(
        2 * np.pi * (fx * xx_ext + fy * yy_ext)
    ) + np.sin(phases[i]) * np.sin(2 * np.pi * (fx * xx_ext + fy * yy_ext))
    extrap_img += amps[i] * component

# === Plot ===
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Extrapolated World (Centered Observer)")
plt.imshow(extrap_img, cmap="gray")
plt.axvline(pad_x, color="red", linestyle="--")
plt.axvline(pad_x + width, color="red", linestyle="--")
plt.axhline(pad_y, color="red", linestyle="--")
plt.axhline(pad_y + height, color="red", linestyle="--")
plt.text(pad_x + 2, pad_y + 2, "Observer", color="red", fontsize=8)

plt.subplot(1, 2, 2)
plt.title("Original Observer")
plt.imshow(obs, cmap="gray")

plt.tight_layout()
plt.show()
print("done")
