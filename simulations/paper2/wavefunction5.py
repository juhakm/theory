import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft2, fftshift

# === Simulation configuration ===
H, W = 128, 128  # spacetime dimensions (time x space)

# === Create base wavefield ===
R = np.zeros((H, W), dtype=np.float32)


# === Generate a light-speed particle (photon) ===
def generate_light_particle(shape, center, k=5.0, sigma=2.0):
    """
    Create a Gaussian-modulated light-speed wave packet (photon).
    shape: (H, W) of the simulation field
    center: (y0, x0) initial center of the photon
    k: base frequency
    sigma: width of the gaussian envelope
    """
    H, W = shape
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x, y)

    # Diagonal coordinate (45 deg line): light-speed axis
    diag = (xx + yy) / np.sqrt(2)
    diag0 = (center[1] / W + center[0] / H) / np.sqrt(2)

    # Gaussian envelope along the light-speed trajectory
    envelope = np.exp(-((diag - diag0) ** 2) / (2 * sigma**2))

    # Oscillating carrier wave along same direction
    carrier = np.cos(2 * np.pi * k * diag)

    return envelope * carrier


# === Inject photon into wavefield ===
photon_wave = generate_light_particle(shape=(H, W), center=(32, 32), k=8.0)
R += photon_wave

# === Visualize photon propagation over time ===
plt.figure(figsize=(8, 6))
for t in range(16, H - 16, 4):
    frame = R[t - 16 : t + 16, :]
    plt.clf()
    plt.imshow(frame, cmap="hot", aspect="auto")
    plt.title(f"Photon centered at t={t}")
    plt.xlabel("Space")
    plt.ylabel("Time window")
    plt.colorbar(label="Amplitude")
    plt.pause(0.2)

plt.show()
