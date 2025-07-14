import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
import os

# === PARAMETERS ===
image_path = os.path.join(
    "simulations", "paper2", "observer.png"
)  # <-- your input image
extrapolated_size = (256, 256)  # desired output resolution

# === LOAD YOUR IMAGE ===
img_raw = imread(image_path)

# Convert to grayscale if RGB
if img_raw.ndim == 3:
    img_small = rgb2gray(img_raw)
else:
    img_small = img_raw

observer_size = img_small.shape[:2]  # (height, width)

# === FOURIER ANALYSIS ===
fft_coeffs = np.fft.fftshift(np.fft.fft2(img_small))

# Flatten and sort FFT coefficients by magnitude
flat_fft = fft_coeffs.flatten()
indices = np.argsort(np.abs(flat_fft))[::-1]

# Keep only top N components
N = 2000  # adjust for quality/speed tradeoff
mask = np.zeros_like(flat_fft, dtype=bool)
mask[indices[:N]] = True
filtered_fft = np.zeros_like(flat_fft, dtype=complex)
filtered_fft[mask] = flat_fft[mask]
filtered_fft = filtered_fft.reshape(fft_coeffs.shape)

# --- Define small image spatial domain ---
h_small, w_small = observer_size
x_small = np.linspace(-0.5, 0.5, w_small)
y_small = np.linspace(-0.5, 0.5, h_small)
dx = x_small[1] - x_small[0]
dy = y_small[1] - y_small[0]

# --- Compute frequency axes with correct scaling ---
kx = np.fft.fftshift(np.fft.fftfreq(w_small, d=dx))
ky = np.fft.fftshift(np.fft.fftfreq(h_small, d=dy))

# --- Define large image spatial domain for extrapolation ---
h_large, w_large = extrapolated_size
scale_factor_x = w_large / w_small
scale_factor_y = h_large / h_small

x_large = np.linspace(-0.5 * scale_factor_x, 0.5 * scale_factor_x, w_large)
y_large = np.linspace(-0.5 * scale_factor_y, 0.5 * scale_factor_y, h_large)
X, Y = np.meshgrid(x_large, y_large)

# --- Reconstruct wavefunction on large grid ---
output_image = np.zeros((h_large, w_large), dtype=complex)

for i, f_y in enumerate(ky):
    for j, f_x in enumerate(kx):
        amp = filtered_fft[i, j]
        output_image += amp * np.exp(2j * np.pi * (f_x * X + f_y * Y))

# --- Normalize real part for display ---
wave_real = np.real(output_image)
wave_real -= wave_real.min()
wave_real /= wave_real.max()

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Extrapolated Wavefunction Amplitude")
plt.imshow(wave_real, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Extrapolated Wavefunction Phase")
plt.imshow(np.angle(output_image), cmap="twilight")
plt.axis("off")

plt.show()

print("Done.")
