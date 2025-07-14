import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import os

# === PARAMETERS ===
image_path = os.path.join(
    "simulations", "paper2", "observer.png"
)  # <-- your input image
extrapolated_size = (256, 256)  # desired output

# === LOAD YOUR IMAGE ===
img_raw = imread(image_path)

# Convert to grayscale if RGB
if img_raw.ndim == 3:
    img_small = rgb2gray(img_raw)
else:
    img_small = img_raw
observer_size = img_small.shape[:2]


# === FOURIER ANALYSIS ===
fft_coeffs = np.fft.fftshift(np.fft.fft2(img_small))

# Setup larger output grid
h_large, w_large = extrapolated_size
output_image = np.zeros((h_large, w_large), dtype=np.complex128)

# Frequency axes
ky = np.fft.fftshift(np.fft.fftfreq(observer_size[0]))
kx = np.fft.fftshift(np.fft.fftfreq(observer_size[1]))

# Spatial grid over the larger image
y = np.linspace(-0.5, 0.5, h_large)
x = np.linspace(-0.5, 0.5, w_large)
X, Y = np.meshgrid(x, y)

# === RECONSTRUCT WAVEFUNCTION ===
for i, f_y in enumerate(ky):
    for j, f_x in enumerate(kx):
        amp = fft_coeffs[i, j]
        phase = 2j * np.pi * (f_y * Y + f_x * X)
        output_image += amp * np.exp(phase)

# === NORMALIZE AND SHOW ===
wave = np.real(output_image)
wave -= wave.min()
wave /= wave.max()

plt.figure(figsize=(8, 4))
plt.imshow(wave, cmap="gray")
plt.title("Extrapolated Wavefunction")
plt.axis("off")
plt.show()
