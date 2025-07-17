import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Optional


"""
By extrapolating the information dynamics of a black hole simulation, we infer 
qualitative properties of the black hole singularity, even though it lies beyond
the reach of both simulation and known physics.

This leads to the conlusion that any set of information in zero-entropy 
state maps to the same geometric object - a point. 

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

import numpy as np
from scipy.integrate import solve_ivp

initial_radius: float = 3.1
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple


class GeodesicParticle:
    """
    Simulates a radially infalling, massless test particle in Schwarzschild spacetime
    under the geodesic equations for a non-rotating black hole. Uses solve_ivp with the Radau method.

    Attributes:
        M (float): Mass of the central object (in geometrized units, G = c = 1).
        r0 (float): Initial radial position of the particle.
        E (float): Conserved energy per unit mass (for a particle starting at rest).
        v0 (float): Initial radial velocity (zero for starting from rest).
        max_tau (float): Maximum proper time to integrate over.
        max_step (float): Maximum step size for the integrator.
        tolerance (float): Minimum radial coordinate allowed to avoid singularity.
        trajectory (List[Tuple[float, float]]): List of (tau, r) positions during integration.
        stopped_due_to_precision (bool): True if the integration terminated due to the stopping condition.
    """

    def __init__(
        self,
        r0: float,
        M: float = 1.0,
        max_tau: float = 120.0,
        max_step: float = 0.01,
        tolerance: float = 1e-10,
    ) -> None:
        """
        Initializes the particle and integrates its geodesic trajectory.

        Args:
            r0 (float): Initial radial position.
            M (float): Mass of the central object.
            max_tau (float): Maximum proper time to integrate.
            max_step (float): Maximum integration step size.
            tolerance (float): Minimum radius to avoid singularity.
        """
        self.M = M
        self.r0 = r0
        self.E = np.sqrt(1 - 2 * M / r0)  # Energy for particle at rest at r0
        self.v0 = 0.0
        self.max_tau = max_tau
        self.max_step = max_step
        self.tolerance = tolerance
        self.trajectory: List[Tuple[float, float]] = []
        self.stopped_due_to_precision: bool = False
        self.integrate_geodesic()

    def geodesic_rhs(self, tau: float, y: List[float]) -> List[float]:
        """
        Computes the right-hand side of the geodesic equations.

        Args:
            tau (float): Proper time parameter.
            y (List[float]): State vector [r, dr/dtau].

        Returns:
            List[float]: Derivatives [dr/dtau, d²r/dtau²].
        """
        r, v = y
        r = max(r, self.tolerance)
        dv_dtau = -self.M / r**2 + 3 * self.M / r**3 * v**2
        dv_dtau = np.clip(dv_dtau, -1e6, 1e6)  # prevent numerical explosion
        return [v, dv_dtau]

    def integrate_geodesic(self):
        def stop_near_singularity(tau, y):
            r, _ = y
            return r - self.tolerance  # stops when r - tolerance = 0

        stop_near_singularity.terminal = True
        stop_near_singularity.direction = -1  # only trigger when r is decreasing

        result = solve_ivp(
            self.geodesic_rhs,
            [0, self.max_tau],
            [self.r0, self.v0],
            method="Radau",  # better for stiff
            max_step=self.max_step,
            rtol=1e-6,
            atol=1e-8,
            events=stop_near_singularity,
        )

        self.trajectory = list(zip(result.t, result.y[0]))
        self.stopped_due_to_precision = result.status == 1

    def get_positions(self) -> List[float]:
        """
        Returns:
            List[float]: Radial positions `r` sampled over proper time.
        """
        return [r for tau, r in self.trajectory]


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
        self,
        num_particles: int = 5,
        r_start: float = initial_radius,
        max_steps: int = 1000,
        mass: float = 1.0,
        max_tau: float = 20.0,
        max_step: float = 0.01,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Initialize the dust cloud using geodesic particles falling in a Schwarzschild metric.

        Args:
            num_particles (int): Number of particles to simulate.
            r_start (float): Starting radius for the first particle.
            max_steps (int): Maximum number of entropy steps to record (used for plotting).
            mass (float): Mass of the central object (M in Schwarzschild metric).
            max_tau (float): Maximum proper time for particle trajectories.
            max_step (float): Integration step size in proper time.
            tolerance (float): Tolerance for singularity (to halt integration).
        """
        self.num_particles: int = num_particles
        self.max_steps: int = max_steps
        self.mass: float = mass

        # Initialize Geodesic Particles
        self.particles: List[GeodesicParticle] = [
            GeodesicParticle(
                r0=r_start + i * 0.1,  # Slight offset per particle
                M=self.mass,
                max_tau=max_tau,
                max_step=max_step,
                tolerance=tolerance,
            )
            for i in range(num_particles)
        ]

        self.history_r: List[List[float]] = [[] for _ in self.particles]
        self.history_tau: List[List[float]] = [[] for _ in self.particles]
        self.entropies: List[float] = []
        self.entropy_taus: List[float] = []

        self.max_radius_bin: int = 15000  # for entropy binning (e.g. r*1000 up to 15.0)

    def compute_entropy(self, positions, num_bins=30):
        """
        Compute Shannon entropy of positions distribution.

        positions: list or numpy array of particle positions (floats)
        num_bins: number of bins for histogram
        """
        hist, bin_edges = np.histogram(positions, bins=num_bins, density=True)
        # Remove zero probabilities to avoid log(0)
        probs = hist[hist > 0]
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def run(self, num_steps=500):
        max_tau: float = min(p.trajectory[-1][0] for p in self.particles)
        self.entropy_taus = np.linspace(0, max_tau, num_steps)

        # Clear histories and initialize empty lists per particle
        self.history_r = [[] for _ in self.particles]
        self.history_tau = [[] for _ in self.particles]
        self.entropies = []

        trajectory_maps = []
        tau_ranges = []

        for p in self.particles:
            taus, rs = zip(*p.trajectory)
            taus = np.array(taus)
            rs = np.array(rs)
            trajectory_maps.append((taus, rs))
            tau_ranges.append((taus[0], taus[-1]))

        for step, current_tau in enumerate(self.entropy_taus):
            current_positions = []
            for i, (taus, rs) in enumerate(trajectory_maps):
                min_tau, max_tau = tau_ranges[i]
                if current_tau < min_tau:
                    r = rs[0]
                elif current_tau > max_tau:
                    r = np.nan
                else:
                    r = np.interp(current_tau, taus, rs)
                current_positions.append(r)
                self.history_r[i].append(r)
                self.history_tau[i].append(current_tau)

            valid_rs = [r for r in current_positions if not np.isnan(r)]
            entropy = self.compute_entropy(valid_rs) if valid_rs else 0.0
            self.entropies.append(entropy)

            if step % 50 == 0 or step == num_steps - 1:
                print(
                    f"Step {step}, τ = {current_tau:.5f}, mean r = {np.mean(valid_rs):.5f}, entropy = {entropy:.5f}"
                )

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
        coords: List[int] = [
            int(max(0.0, min(65535.0, r * 1000))) for r in positions if not np.isnan(r)
        ]

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
        plt.savefig("schwarzschild_raudau.png", dpi=300)

        plt.show()


import time


if __name__ == "__main__":
    start = time.time()
    p = GeodesicParticle(r0=initial_radius, M=1.0, max_tau=50, max_step=0.05)
    print(f"Took {time.time() - start:.2f} seconds")

    #    p = GeodesicParticle(r0=initial_radius, max_tau=100, max_step=0.01)
    for tau, r in p.trajectory[:10]:  # first 10 steps
        print(f"τ = {tau:.5f}, r = {r:.5f}")

    cloud = SchwarzschildDustCloud(
        num_particles=50,
        r_start=initial_radius,
        max_steps=1000,
        mass=1.0,
        max_tau=50.0,
        max_step=0.005,
        tolerance=1e-6,
    )
    cloud.run()
    cloud.visualize()
    print("Done.")
