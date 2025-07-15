import numpy as np
from PIL import Image
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os

# === Configuration ===
path = os.path.join("simulations", "paper2")
image_path = os.path.join(path, "observer.png")
num_freqs = 20  # Number of frequencies to fit in each dimension

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


freqs = np.linspace(0, 5, num_freqs)
basis = make_basis(freqs, freqs, obs.shape)
assert obs.shape == basis[0][1].shape


# === Model and residuals ===
def model(params):
    amps = params[: len(basis)]
    phases = params[len(basis) :]
    img = np.zeros_like(obs)
    for i, ((fx, fy), sin_part, cos_part) in enumerate(basis):
        img += amps[i] * (np.cos(phases[i]) * cos_part + np.sin(phases[i]) * sin_part)
    return img.ravel()


def residuals(params):
    return model(params) - obs.ravel()


# === Initial guess ===
np.random.seed(0)
init_amps = 0.1 * np.random.randn(len(basis))
init_phases = 2 * np.pi * np.random.rand(len(basis))
params0 = np.concatenate([init_amps, init_phases])

# === Fit ===
result = least_squares(residuals, params0, verbose=2, max_nfev=500)

# === Reconstruct fitted image ===
fitted = model(result.x).reshape(obs.shape)

# === Show original and fitted ===
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Observer")
plt.imshow(obs, cmap="gray", vmin=0, vmax=1)

plt.subplot(1, 2, 2)
plt.title("Fitted (Wave Interpolation)")
plt.imshow(fitted, cmap="gray", vmin=0, vmax=1)
plt.tight_layout()
plt.show()

# === Extrapolation setup ===
extrap_height = 128
extrap_width = 128
pad_x = (extrap_width - width) // 2
pad_y = (extrap_height - height) // 2

x_ext = np.linspace(-pad_x / width, (width + pad_x) / width, extrap_width)
y_ext = np.linspace(-pad_y / height, (height + pad_y) / height, extrap_height)
xx_ext, yy_ext = np.meshgrid(x_ext, y_ext)

# === Extrapolate the wavefunction ===
amps = result.x[: len(basis)]
phases = result.x[len(basis) :]
extrap_img = np.zeros((extrap_height, extrap_width))

for i, ((fx, fy), _, _) in enumerate(basis):
    component = np.cos(phases[i]) * np.cos(
        2 * np.pi * (fx * xx_ext + fy * yy_ext)
    ) + np.sin(phases[i]) * np.sin(2 * np.pi * (fx * xx_ext + fy * yy_ext))
    extrap_img += amps[i] * component

# === Normalize for display using percentiles ===
vmin, vmax = np.percentile(extrap_img, [1, 99])
norm_extrap = np.clip((extrap_img - vmin) / (vmax - vmin), 0, 1)

# === Extract center patch to verify match ===
extrap_center = norm_extrap[pad_y : pad_y + height, pad_x : pad_x + width]

# === Display extrapolation ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Extrapolated World (Normalized)")
plt.imshow(norm_extrap, cmap="gray")
plt.axvline(pad_x, color="red", linestyle="--")
plt.axvline(pad_x + width, color="red", linestyle="--")
plt.axhline(pad_y, color="red", linestyle="--")
plt.axhline(pad_y + height, color="red", linestyle="--")
plt.text(pad_x + 2, pad_y + 2, "Observer", color="red", fontsize=8)

plt.subplot(1, 3, 2)
plt.title("Extrapolated Center Patch")
plt.imshow(extrap_center, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Original Observer")
plt.imshow(obs, cmap="gray", vmin=0, vmax=1)

plt.tight_layout()
plt.show()

# === Diagnostic test: sanity check ===
region_from_full = norm_extrap[pad_y : pad_y + height, pad_x : pad_x + width]
delta = region_from_full - extrap_center

print("Sanity check — max diff:", np.abs(delta).max())
print("Center patch mean/std:", extrap_center.mean(), extrap_center.std())
print(
    "Region from full image mean/std:", region_from_full.mean(), region_from_full.std()
)

assert np.allclose(region_from_full, extrap_center, atol=1e-6), "Center mismatch!"

plt.figure(figsize=(12, 3))

plt.subplot(1, 3, 1)
plt.title("extrap_center")
plt.imshow(extrap_center, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("norm_extrap center region")
plt.imshow(region_from_full, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Difference (should be ~0)")
plt.imshow(delta, cmap="bwr", vmin=-0.1, vmax=0.1)
plt.colorbar()

plt.tight_layout()
plt.show()
print("Done")
