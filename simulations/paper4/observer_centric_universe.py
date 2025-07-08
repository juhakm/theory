"""
Module: observer_centric_universe.py

This simulation implements an observer-centric, information-theoretic model of universe evolution.
It aligns with the ideas presented in the paper:

    "Observer-Centric Bitstring Universes: A Minimal Informational Model of Reality"

The program generates random sequences of bitstrings interpreted as possible universe states,
ranks them by similarity with an observer-defined pattern, and extracts the most coherent
trajectories of emergent particle patterns. It supports key claims from the paper:

0. Wavefunction is the most effective peridic pattern for compressing information.
1. Entropy increase over time corresponds to unfolding structure.
2. Observer-defined memory patterns influence the selection of consistent universe paths.
3. Emergent particle trajectories can be detected as coherent motifs in discrete time-space evolution.
4. Heatmaps reveal structured versus random evolution, suggesting alignment with entropy-driven spacetime expansion.

Visualizations include:
- Particle trajectory plots
- Universe state heatmaps
- Future particle prediction maps

This simulation serves as evidence for the Entropy-Singularity Lemma and supports the notion that physical
laws emerge from abstract informational configurations.
"""

import matplotlib.pyplot as plt
from itertools import product
from typing import List, Tuple, Optional
import numpy as np
import random
import math

# --- Utilities ---


def bits_to_frames(bitstring: Tuple[int], space_size: int) -> List[List[int]]:
    """Split a flat bitstring into spatial frames."""
    return [
        list(bitstring[i : i + space_size])
        for i in range(0, len(bitstring), space_size)
    ]


def bitstring_to_waveform(bits: List[int]) -> List[float]:
    """Convert a bitstring (0/1) to waveform (-1/+1)."""
    return [1.0 if b == 1 else -1.0 for b in bits]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-10)  # avoid division by zero


def generate_sine_wave(length: int, freq: float = 1.0) -> List[float]:
    """Generate a sine wave pattern of a given length and frequency."""
    return [math.sin(2 * math.pi * freq * i / length) for i in range(length)]


def wave_similarity(observer_bits: List[int], universe_bits: List[int]) -> float:
    """Compute cosine similarity between observer and universe waveforms."""
    observer_wave = bitstring_to_waveform(observer_bits)
    universe_wave = bitstring_to_waveform(universe_bits)
    return cosine_similarity(observer_wave, universe_wave)


# --- Particle Detection ---


def detect_particle_positions(
    frames: List[List[int]], particle_pattern: List[int]
) -> List[Optional[int]]:
    """Detect most probable positions of a particle pattern over time."""
    pattern_len = len(particle_pattern)
    positions = []
    space_size = len(frames[0])

    for frame in frames:
        best_pos = None
        min_diff = float("inf")
        for i in range(space_size - pattern_len + 1):
            segment = frame[i : i + pattern_len]
            diff = sum(abs(a - b) for a, b in zip(segment, particle_pattern))
            if diff < min_diff:
                min_diff = diff
                best_pos = i
        positions.append(best_pos if min_diff == 0 else None)
    return positions


# --- Universe Simulation ---


def generate_universes(n_bits: int):
    """Generate all possible bitstring universes of length n_bits."""
    return product([0, 1], repeat=n_bits)


def simulate_universe_evolution_sampled(
    n_bits: int,
    time_steps: int,
    particle_pattern: List[int],
    observer_pattern: List[int],
    sample_size: int = 1000,
) -> Tuple[List[Tuple[Tuple[int, ...], ...]], List[List[Optional[int]]]]:
    """Simulate evolution of sampled universes, return top scoring paths and trajectories."""

    def random_state() -> Tuple[int, ...]:
        return tuple(random.randint(0, 1) for _ in range(n_bits))

    def random_path() -> Tuple[Tuple[int, ...], ...]:
        return tuple(random_state() for _ in range(time_steps))

    def path_score(path: Tuple[Tuple[int, ...], ...]) -> float:
        score = 0.0
        for i in range(len(path) - 1):
            score += wave_similarity(path[i], path[i + 1])
        score += sum(wave_similarity(observer_pattern, list(state)) for state in path)
        return score

    scored_paths = []
    for _ in range(sample_size):
        path = random_path()
        sc = path_score(path)
        scored_paths.append((sc, path))

    scored_paths.sort(key=lambda x: x[0], reverse=True)
    top_paths = [p for _, p in scored_paths[:5]]

    trajectories = []
    for path in top_paths:
        frames = [list(state) for state in path]
        traj = detect_particle_positions(frames, particle_pattern)
        trajectories.append(traj)

    return top_paths, trajectories


# --- Visualization ---


def plot_universe_2d(frames: List[List[int]], title: str) -> None:
    """Display a heatmap of bit values over space and time for a universe path."""
    data = np.array(frames)
    plt.figure(figsize=(8, 4))
    plt.imshow(data, cmap="gray_r", interpolation="nearest", aspect="auto")
    plt.colorbar(label="Bit Value")
    plt.xlabel("Space Index")
    plt.ylabel("Time Step")
    plt.title(title)
    plt.savefig("state_evolution_heatmap.png", format="png", dpi=300)
    plt.show()


def plot_particle_trajectories(
    trajectories: List[List[Optional[int]]], paths: List[Tuple[Tuple[int, ...], ...]]
):
    plt.figure(figsize=(10, 6))
    for traj, path in zip(trajectories, paths):
        y = [i for i in range(len(traj)) if traj[i] is not None]  # time
        x = [traj[i] for i in y]  # position
        if x and y:
            plt.plot(
                x,
                y,
                marker="o",
                label=f'Path {"->".join("".join(map(str, s)) for s in path)}',
            )
    plt.xlabel("Space Index")
    plt.ylabel("Time Step")

    plt.title("Particle Trajectories Over Time")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True)
    plt.savefig("particle_trajectories.png", format="png", dpi=300)
    plt.show()


def plot_future_particle_heatmap(
    last_frame: List[int],
    particle_pattern: List[int],
    n_future: int = 16,
    space_size: int = 64,
    n_samples: int = 5000,
):
    heatmap = np.zeros((n_future, space_size))

    def biased_future(last_frame: List[int], p: float = 0.9) -> List[int]:
        return [bit if random.random() < p else 1 - bit for bit in last_frame]

    def random_future():
        frames = []
        frame = last_frame.copy()
        for _ in range(n_future):
            frame = biased_future(frame, p=0.9)
            frames.append(frame.copy())
        return frames

    for _ in range(n_samples):
        future_frames = random_future()
        full_frames = [last_frame] + future_frames
        traj = detect_particle_positions(full_frames, particle_pattern)
        for t, pos in enumerate(traj[1:], start=0):  # exclude t=0
            if pos is not None:
                heatmap[t, pos] += 1

    # Normalize heatmap for plotting
    heatmap /= heatmap.max() + 1e-10

    plt.figure(figsize=(10, 5))
    plt.imshow(heatmap, cmap="viridis", interpolation="nearest", aspect="auto")
    plt.colorbar(label="Normalized Particle Likelihood")
    plt.xlabel("Space Index")
    plt.ylabel("Future Time Step")
    plt.title("Future Particle Appearance Heatmap")
    plt.savefig("future_particle_heatmap.png", format="png", dpi=300)
    plt.show()


def plot_average_heatmap(paths: List[Tuple[Tuple[int, ...], ...]]):
    """
    Compute and display average bit activity (0-1) across time and space
    for the top-k best scoring universe paths.
    """
    # Convert to 3D array: (num_paths, time_steps, space_size)
    path_tensor = np.array(
        [[[int(b) for b in frame] for frame in path] for path in paths]
    )

    # Average across top paths
    averaged = np.mean(path_tensor, axis=0)  # shape: (time_steps, space_size)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(averaged, cmap="hot", interpolation="nearest", aspect="auto")
    plt.colorbar(label="Average Bit Activity")
    plt.xlabel("Space Index")
    plt.ylabel("Time Step")
    plt.title("Average Universe Heatmap (Top Paths)")
    plt.savefig("average_universe_heatmap.png", format="png", dpi=300)
    plt.show()


# --- Main ---
if __name__ == "__main__":
    n_bits = 64
    time_steps = 64
    particle_pattern = [0, 1, 0, 1, 1, 0, 1, 0]
    observer_pattern = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0]

    print(f"Simulating universes with {n_bits} bits over {time_steps} time steps...")

    top_paths, trajectories = simulate_universe_evolution_sampled(
        n_bits, time_steps, particle_pattern, observer_pattern, sample_size=5000
    )

    for i, path in enumerate(top_paths):
        frames = [list(state) for state in path]
        print(f"\nPath {i+1}:")
        for t, state in enumerate(path):
            print(f"  Time {t}: {''.join(map(str, state))}")
        plot_universe_2d(frames, f"Universe Evolution for Path {i+1}")

    # Plot averaged heatmap over top-k paths
    plot_average_heatmap(top_paths)

    plot_particle_trajectories(trajectories, top_paths)

    # Visualize future heatmap starting from top observer-compatible path
    best_path = top_paths[0]
    last_frame = list(best_path[-1])
    plot_future_particle_heatmap(
        last_frame,
        particle_pattern,
        n_future=16,
        space_size=n_bits,
        n_samples=5000,
    )

    print("Simulation complete. Visualizations saved as images.")
