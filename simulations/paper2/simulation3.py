import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft2
from scipy.signal import convolve2d
from matplotlib.patches import Rectangle

# Configuration
width = 64
height = 64
num_samples = 2000
num_freqs = 9182
threshold = 0.0
top_n_matches = 3


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


def generate_circular_pattern(size):
    y, x = np.ogrid[:size, :size]
    center = size / 2
    mask = (x - center) ** 2 + (y - center) ** 2 <= (size / 2.5) ** 2
    return mask.astype(int)


def count_observer_matches(space, pattern):
    conv = convolve2d(space, pattern[::-1, ::-1], mode="valid")
    match_count = np.sum(conv == np.sum(pattern))
    return match_count, conv


def find_best_spacetime():
    best_score = -1
    best_wave = None
    best_binary = None
    best_conv = None

    for _ in range(num_samples):
        wave = generate_fourier_wavefield(width, height, num_freqs)
        binary = (wave > threshold).astype(int)
        score, conv = count_observer_matches(binary, observer_pattern)
        if score > best_score:
            best_score = score
            best_wave = wave
            best_binary = binary
            best_conv = conv

    return best_wave, best_binary, best_conv, best_score


def find_top_matches(conv_map, pattern_shape, top_n):
    indices = np.dstack(
        np.unravel_index(np.argsort(conv_map.ravel())[::-1], conv_map.shape)
    )[0]
    seen = set()
    matches = []

    for y, x in indices:
        if len(matches) >= top_n:
            break
        if (y, x) not in seen:
            matches.append((y, x))
            for dy in range(-pattern_shape[0] // 2, pattern_shape[0] // 2):
                for dx in range(-pattern_shape[1] // 2, pattern_shape[1] // 2):
                    seen.add((y + dy, x + dx))
    return matches


def plot_with_overlays(ax, data, title, match_coords, color_map="gray"):
    ax.imshow(data, cmap=color_map, interpolation="nearest")
    ax.set_title(title)
    ax.invert_yaxis()
    for y, x in match_coords:
        rect = Rectangle(
            (x, y),
            observer_pattern.shape[1],
            observer_pattern.shape[0],
            linewidth=1.5,
            edgecolor="cyan",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_xlabel("Space")
    ax.set_ylabel("Time")


# Observer pattern
observer_pattern = generate_circular_pattern(13)

# Run simulation
wave, binary, conv, score = find_best_spacetime()
wave_match_conv = convolve2d(wave, observer_pattern[::-1, ::-1], mode="valid")
normalized_wave_match = wave_match_conv**2
normalized_wave_match /= np.max(normalized_wave_match)

# Get top match locations
top_matches = find_top_matches(conv, observer_pattern.shape, top_n_matches)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

plot_with_overlays(
    axs[0, 0], binary, "Binary Observer Match", top_matches, color_map="gray"
)
plot_with_overlays(
    axs[0, 1], conv, "Wave Convolution Match", top_matches, color_map="hot"
)
plot_with_overlays(
    axs[1, 0], wave, "Underlying Wavefield (Reality)", top_matches, color_map="seismic"
)
plot_with_overlays(
    axs[1, 1],
    normalized_wave_match,
    "Normalized |Ψ|² Probability Map",
    top_matches,
    color_map="inferno",
)

plt.tight_layout()
plt.savefig("observer_wave_evolution.png", format="png", dpi=300)
plt.show()
