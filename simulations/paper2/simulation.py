"""This progrma simulates how an **observer**, represented as a 2D binary pattern, finds itself embedded in
a universe described by **finite information**. The goal is to find the most likely configuration of
reality from the observer's point of view.

Finite **Fourier components** is used for compressing data - to increase the number of observers a spacetime can embed.
This naturally leads to  quantum probability interpretations of observer presence via wavefunction overlap.

Note: in the spirit of the theory, we should construct all the possibles universes, all the possible observers, and find
the best matches. For performance reasons, we settle to study one observer only.


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft2
from scipy.signal import convolve2d
from matplotlib.patches import Rectangle
from typing import Tuple, List, Optional

# Configuration parameters
# Note: a Theory of Everything grounded in pure information must avoid arbitrary or privileged constants
# like “num_freqs” or “observer_size”. These are conveniences for simulation — not part of the theory itself.
width: int = 128  # bits per space
height: int = 128  # bits per time
num_samples: int = 1000  # Number of random wavefields to sample
num_freqs: int = 100  # Number of frequency components used in wave generation
threshold: float = 0.0  # Binary threshold for wavefield binarization
top_n_matches: int = 3  # Number of top observer matches to highlight
observer_size: int = (
    16  # observer is assumed to be disk - starting out small, getting bigger, and then fading a way
)

circle = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
]


def generate_fourier_wavefield(
    width: int, height: int, num_freqs: int = 10
) -> np.ndarray:
    """
    Generate a real-valued 2D wavefield by randomly populating a Fourier domain
    and applying inverse FFT, simulating a spacetime structure.

    Args:
        width: Width of the wavefield.
        height: Height of the wavefield.
        num_freqs: Number of non-zero frequency components.

    Returns:
        A 2D numpy array representing the normalized real-valued wavefield.
    """
    freq_data = np.zeros((height, width), dtype=complex)
    for _ in range(num_freqs):
        fy = np.random.randint(0, height // 4)
        fx = np.random.randint(0, width // 4)
        amp = np.random.randn() + 1j * np.random.randn()
        freq_data[fy, fx] = amp
        freq_data[-fy % height, -fx % width] = np.conj(amp)  # Ensure Hermitian symmetry
    wave = np.real(ifft2(freq_data))
    wave = (wave - np.mean(wave)) / np.std(
        wave
    )  # Normalize to zero mean and unit variance
    return wave


def generate_circular_pattern(size: int) -> np.ndarray:
    """
    Create a circular binary pattern to represent the observer's structure.
    Circular shape implies the observer is born small, grows bigger, then looses weight and disappears.

    Args:
        size: Diameter of the pattern.

    Returns:
        A 2D numpy array of 0s and 1s forming a circular mask.
    """
    y, x = np.ogrid[:size, :size]
    center = size / 2
    mask = (x - center) ** 2 + (y - center) ** 2 <= (size / 2.5) ** 2
    return mask.astype(int)


def count_observer_matches(
    space: np.ndarray, pattern: np.ndarray
) -> Tuple[int, np.ndarray]:
    """
    Count how many times the observer pattern exactly matches within the binary space.

    Args:
        space: The binary wavefield to scan.
        pattern: The observer pattern.

    Returns:
        A tuple of (match count, full convolution map).
    """
    conv = convolve2d(space, pattern[::-1, ::-1], mode="valid")
    match_count = np.sum(conv == np.sum(pattern))
    return match_count, conv


def find_best_spacetime() -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Search over many random wavefields to find the one that best matches the observer pattern.

    Returns:
        Tuple of (best wavefield, binarized wavefield, match convolution map, match count).
    """
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


def find_top_matches(
    conv_map: np.ndarray, pattern_shape: Tuple[int, int], top_n: int
) -> List[Tuple[int, int]]:
    """
    Identify the top-N most compatible observer locations in the convolution map.

    Args:
        conv_map: The full convolution result of pattern matching.
        pattern_shape: Shape of the observer pattern.
        top_n: Number of top matches to return.

    Returns:
        A list of (y, x) coordinates where the observer pattern most strongly appears.
    """
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


def plot_with_overlays(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    match_coords: List[Tuple[int, int]],
    color_map: str = "gray",
) -> None:
    """
    Plot a 2D data field with rectangles overlaid at observer match locations.

    Args:
        ax: Matplotlib Axes object to draw on.
        data: 2D array to visualize.
        title: Plot title.
        match_coords: List of (y, x) match coordinates to highlight.
        color_map: Matplotlib colormap string.
    """
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


# Observer structure definition
observer_pattern = np.array(circle, dtype=int)

# Run simulation and find best wave that matches the observer
wave, binary, conv, score = find_best_spacetime()

# Compute squared wavefunction match (|Ψ|²)
wave_match_conv: np.ndarray = convolve2d(
    wave, observer_pattern[::-1, ::-1], mode="valid"
)
normalized_wave_match: np.ndarray = wave_match_conv**2
normalized_wave_match /= np.max(normalized_wave_match)

# Find top-N most likely observer locations
# top_matches: List[Tuple[int, int]] = find_top_matches(
#    conv, observer_pattern.shape, top_n_matches

top_matches: List[Tuple[int, int]] = find_top_matches(
    normalized_wave_match, observer_pattern.shape, top_n_matches
)


# Plotting all visual outputs
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
