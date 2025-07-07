import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Optional
import argparse
import os


"""

This program simulates the radial infall of dust particles in the
Schwarzschild metric — a simplified model of a non-rotating black hole.

Each particle falls radially inward from a given radius towards the
black hole singularity at r=0. The event horizon is at Schwarzschild radius
rs=2 (in geometric units). Particle motion is parameterized by proper time τ.

The simulation tracks particles' radial coordinates over time, quantizes
their positions into bits representing computer memory states, and calculates
Shannon entropy of these bitstrings. The bit string evolution can 
be pictured as information-theoretical view to black hole physics.

As particles approach the singularity, their coordinates converge to zero,
leading to entropy dropping towards zero. The program visualizes entropy
and radial trajectories versus proper time.

"""


class SchwarzschildParticle:
    """
    Represents a single test particle falling radially in a Schwarzschild spacetime.

    Attributes:
        rs (float): Schwarzschild radius of the black hole.
        r (Optional[float]): Current radial coordinate of the particle.
        E (float): Conserved energy per unit mass for the particle.
        tau (float): Proper time experienced by the particle.
        dtau (float): Proper time step increment for numerical integration.
        alive (bool): Whether the particle is still falling (not at singularity).
    """

    def __init__(self, r0: float, dtau: float = 0.01) -> None:
        """
        Initialize the particle with a starting radius and time step.

        Args:
            r0 (float): Initial radius coordinate.
            dtau (float): Proper time step for numerical integration.
        """
        self.rs: float = 2.0
        self.r: Optional[float] = r0
        self.E: float = math.sqrt(1 - self.rs / self.r)  # Energy constant for free-fall
        self.tau: float = 0.0
        self.dtau: float = dtau
        self.alive: bool = True
        self.r -= 1e-8  # slight nudge to avoid zero velocity at start

    def step(self) -> Optional[float]:
        """
        Advance the particle by one time step in proper time τ.

        Computes radial velocity from energy conservation, updates position and τ.
        Snaps r to zero and marks dead if particle reaches near singularity.

        Returns:
            Optional[float]: Updated radial position or None if particle is dead.
        """
        if not self.alive or self.r is None or self.r <= 1e-3:
            self.alive = False
            return None

        velocity_sq = self.E**2 - (1 - self.rs / self.r)
        if velocity_sq <= 0:
            velocity_sq = 1e-10  # avoid sqrt of negative

        dr_dtau = -math.sqrt(velocity_sq)
        self.r += dr_dtau * self.dtau
        self.tau += self.dtau

        if self.r <= 1e-3:
            self.r = 0.0
            self.alive = False

        return self.r


class SchwarzschildDustCloud:
    """
    Simulates a cloud of radially infalling particles around a Schwarzschild black hole.

    Attributes:
        num_particles (int): Number of particles in the cloud.
        max_steps (int): Maximum number of simulation steps.
        particles (List[SchwarzschildParticle]): List of particles.
        history_r (List[List[float]]): Radial coordinate history per particle.
        history_tau (List[List[float]]): Proper time history per particle.
        entropies (List[float]): Shannon entropy values per step.
        entropy_taus (List[float]): Proper time values for entropy measurements.
        max_radius_bin (int): Maximum radius bin for quantization (not currently used).
    """

    def __init__(
        self, num_particles: int = 50, r_start: float = 10.0, max_steps: int = 1000
    ) -> None:
        """
        Initialize the dust cloud with given parameters.

        Args:
            num_particles (int): Number of particles to simulate.
            r_start (float): Starting radius for the first particle.
            max_steps (int): Number of time steps to simulate.
        """
        self.num_particles: int = num_particles
        self.max_steps: int = max_steps
        self.particles: List[SchwarzschildParticle] = [
            SchwarzschildParticle(r0=r_start + i * 0.1) for i in range(num_particles)
        ]
        self.history_r: List[List[float]] = [[] for _ in self.particles]
        self.history_tau: List[List[float]] = [[] for _ in self.particles]
        self.entropies: List[float] = []
        self.entropy_taus: List[float] = []

        self.max_radius_bin: int = 15000  # Supports r up to 15.0 (r*1000)

    def run(self) -> None:
        """
        Run the simulation, stepping all particles through proper time.

        Records radial positions, proper times, and computes entropy from particle positions
        at each step.
        """
        for step in range(self.max_steps):
            current_positions: List[float] = []

            for i, p in enumerate(self.particles):
                r = p.step()
                if r is None:
                    r = 0.0  # dead particle

                self.history_r[i].append(r)
                self.history_tau[i].append(p.tau)
                current_positions.append(r)

            # Compute entropy from current positions' bit representation
            bits = self._positions_to_bits(current_positions)
            self.entropies.append(self._shannon_entropy(bits))

            # Append current simulation time based on steps and dtau
            self.entropy_taus.append(step * self.particles[0].dtau)

    def _positions_to_bits(self, positions: List[float]) -> np.ndarray:
        """
        Convert particle radial positions into a bit array representing memory bits.

        Each position is quantized and represented as a 16-bit binary string.
        All bitstrings are concatenated into one numpy array of bits.

        Args:
            positions (List[float]): List of particle radial positions.

        Returns:
            np.ndarray: Array of bits (0 or 1) representing all positions.
        """
        coords: List[int] = [int(max(0.0, min(65535.0, r * 1000))) for r in positions]
        bitstring: str = "".join(format(c, "016b") for c in coords)
        return np.array([int(b) for b in bitstring])

    def _shannon_entropy(self, bits: np.ndarray) -> float:
        """
        Calculate the Shannon entropy of a bit array.

        Shannon entropy quantifies the uncertainty or information content.

        Args:
            bits (np.ndarray): Array of bits (0 or 1).

        Returns:
            float: Shannon entropy in bits.
        """
        n: int = len(bits)
        if n == 0:
            return 0.0
        p1: float = np.sum(bits) / n
        p0: float = 1 - p1
        entropy: float = 0.0
        if p0 > 0:
            entropy -= p0 * np.log2(p0)
        if p1 > 0:
            entropy -= p1 * np.log2(p1)
        return entropy

    def visualize(self) -> None:
        """
        Visualize the entropy over proper time and the radial infall trajectories.

        Produces a two-panel matplotlib figure:
        - Top: Shannon entropy vs proper time
        - Bottom: Radial coordinate trajectories of particles vs proper time
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 16))

        max_particle_tau: float = (
            max(max(tau_list) for tau_list in self.history_tau if tau_list) * 1.02
        )
        axes[0].set_xlim(0, max_particle_tau)
        axes[1].set_xlim(0, max_particle_tau)

        # 1) Entropy vs Proper Time
        axes[0].plot(self.entropy_taus, self.entropies, label="Shannon Entropy")
        axes[0].set_ylabel("Entropy (bits)")
        axes[0].set_xlabel("Proper Time τ")
        axes[0].legend()
        axes[0].grid(True)

        # 2) Radial infall trajectories
        for i, r_vals in enumerate(self.history_r):
            axes[1].plot(self.history_tau[i], r_vals, alpha=0.5)
        axes[1].axhline(y=2.0, color="red", linestyle="--", label="Event Horizon (r=2)")
        axes[1].set_xlabel("Proper Time τ")
        axes[1].set_ylabel("Radius r")
        axes[1].set_title("Radial Infall of Particles")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Schwarzschild entropy simulation."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the output plot (entropy + trajectories).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="entropy_plot.png",
        help="Filename for the saved figure (PNG).",
    )
    args = parser.parse_args()

    # Run the simulation
    cloud = SchwarzschildDustCloud(num_particles=50, max_steps=7000)
    cloud.run()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)

    # Save figure to file (instead of plt.show)
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    max_particle_tau = (
        max(max(tau_list) for tau_list in cloud.history_tau if tau_list) * 1.02
    )
    axes[0].set_xlim(0, max_particle_tau)
    axes[1].set_xlim(0, max_particle_tau)

    axes[0].plot(cloud.entropy_taus, cloud.entropies, label="Shannon Entropy")
    axes[0].set_ylabel("Entropy (bits)")
    axes[0].set_xlabel("Proper Time τ")
    axes[0].legend()
    axes[0].grid(True)

    for i, r_vals in enumerate(cloud.history_r):
        axes[1].plot(cloud.history_tau[i], r_vals, alpha=0.5)
    axes[1].axhline(y=2.0, color="red", linestyle="--", label="Event Horizon (r=2)")
    axes[1].set_xlabel("Proper Time τ")
    axes[1].set_ylabel("Radius r")
    axes[1].set_title("Radial Infall of Particles")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved figure to: {output_path}")
