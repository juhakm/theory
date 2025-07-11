import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft2

# Configuration
width = 64  # bits per space
height = 64  # bits per time
num_samples = 1000
num_freqs = 1024

# Observer pattern (circular shape, now float-valued)
circle = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
]


def generate_circular_pattern(size):
    y, x = np.ogrid[:size, :size]
    center = size / 2
    mask = (x - center) ** 2 + (y - center) ** 2 <= (size / 2.5) ** 2
    return mask.astype(int)


observer_pattern = generate_circular_pattern(13)
# observer_pattern = np.array(circle, dtype=float)
pattern_height, pattern_width = observer_pattern.shape
observer_norm = np.linalg.norm(observer_pattern.flatten())


def generate_fourier_wavefield(width, height, num_freqs=10):
    freq_data = np.zeros((height, width), dtype=complex)
    for _ in range(num_freqs):
        fy = np.random.randint(0, height // 4)
        fx = np.random.randint(0, width // 4)
        amp = np.random.randn() + 1j * np.random.randn()
        freq_data[fy, fx] = amp
        freq_data[-fy % height, -fx % width] = np.conj(amp)
    wave = np.real(ifft2(freq_data))
    wave = (wave - np.mean(wave)) / np.std(wave)
    return wave


def cosine_similarity(patch, pattern, pattern_norm):
    v1 = patch.flatten()
    v2 = pattern.flatten()
    norm1 = np.linalg.norm(v1)
    if norm1 == 0 or pattern_norm == 0:
        return 0
    return np.dot(v1, v2) / (norm1 * pattern_norm)


def scan_similarity_map(field, pattern):
    h, w = pattern.shape
    H, W = field.shape
    result = np.zeros((H - h + 1, W - w + 1))
    for i in range(H - h + 1):
        for j in range(W - w + 1):
            patch = field[i : i + h, j : j + w]
            result[i, j] = cosine_similarity(patch, pattern, observer_norm)
    return result


def find_best_spacetime():
    best_score = -1
    best_wave = None
    best_similarity_map = None

    for _ in range(num_samples):
        wave = generate_fourier_wavefield(width, height, num_freqs)
        similarity_map = scan_similarity_map(wave, observer_pattern)
        max_score = np.max(similarity_map)
        if max_score > best_score:
            best_score = max_score
            best_wave = wave
            best_similarity_map = similarity_map

    return best_wave, best_similarity_map, best_score


if __name__ == "__main__":
    wave, similarity_map, score = find_best_spacetime()

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(wave, cmap="seismic", interpolation="nearest")
    axs[0].set_title("Raw Wave Field")
    axs[0].invert_yaxis()

    axs[1].imshow(similarity_map, cmap="hot", interpolation="nearest")
    axs[1].set_title(f"Observer Similarity Heatmap (score={score:.4f})")
    axs[1].invert_yaxis()

    for ax in axs:
        ax.set_xlabel("Space")
        ax.set_ylabel("Time")

    plt.tight_layout()
    plt.show()
