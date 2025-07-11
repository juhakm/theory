# Observer-Centric Universe Simulation

**Version 2.0 — A Probabilistic, Information-Theoretic Search for Emergent Observers in Compressed Spacetimes**

This project implements a simulation of universes based on binary information patterns and evaluates their suitability to contain embedded observers. Instead of brute-forcing all possible configurations, it leverages wave-coherent generative models (e.g. random Fourier wavefields) to find highly compressible spacetimes in which a fixed observer pattern appears maximally.

---

## Core Hypothesis

> **The universe we find ourselves in is one where the observer is maximally embedded within the informational fabric of reality.**

This model assumes that:
- Reality is fundamentally informational.
- Observers are compressible local patterns within bitstrings.
- Universes (spacetimes) with better compression (lower algorithmic entropy) are more probable.
- Emergence of observers is the selection criterion for which universe is "observed."

---

## Overview

### Universe as Bitstrings

- Generate candidate spacetimes as 2D bitstrings (e.g., 32×32).
- Each cell is either 0 or 1, representing binary field intensity at that space-time point.

### Observer Pattern

- Define a small binary pattern (e.g., 4×5 block of 1s) representing a compact "observer."
- This is a stand-in for any self-model or consciousness representation that is local and compressible.

### Search for Maximal Observer Density

- Slide the observer pattern across each candidate spacetime.
- Count how many times the pattern appears exactly.
- Normalize by total information content or complexity.

### From Brute Force to Wave-Based Universes

Version 1 brute-forced all bitstring configurations. Version 2 improves efficiency and insight:

- **Generate wave-coherent spacetimes** using low-frequency random Fourier components.
- Bitstrings are derived by thresholding real-valued wavefields.
- This constrains the search to *compressible*, *wave-like* universes—those with fewer frequency components.

Unitary evolution, interference, and continuity all emerge from simple wave compression.

---

## Technical Insight: Compression, Fourier, and Bayesian Filtering

### Compression and Probability

- Compression is equivalent to probability: more compressible states are more likely under an algorithmic prior (Solomonoff).
- Fourier modes serve as a natural basis for compressible spacetimes.
- Limiting to a sparse Fourier spectrum implies we are only sampling from the most probable universes.

### Bayesian Filtering Analogy

This simulation acts like a **Bayesian filter**:

- **Hypothesis space** = All possible spacetime bitstrings
- **Prior** = Simpler (more compressible) bitstrings are more likely (Fourier-limited)
- **Likelihood** = Observer pattern matches (how often the observer appears)
- **Posterior** ∝ Prior × Likelihood → Highest when a compressible spacetime contains a dense observer

---

## Special Cases

### Case 1: `O = U` (Observer = Universe)

- The observer pattern fills the entire spacetime.
- This is a degenerate but compressible case: the universe is only the observer.
- Minimum entropy
- Interpretation: The universe is solipsistic—nothing exists beyond the observer.

### Case 2: `O = []` (Empty Observer)

- The observer pattern is empty.
- Every universe contains this observer trivially.
- Maximum entropy
- Interpretation: No filtering occurs; all universes are equally valid → the prior dominates.

These two cases serve as **boundaries** in the Bayesian filter:
- `O = []` gives no information.
- `O = U` gives maximal, but uninteresting compression.

---

## Scoring and Selection

Each generated universe is evaluated by:

1. **Wave Compression Score**:
   - How sparse the Fourier spectrum is (few dominant frequencies = high compression).
2. **Observer Embedding Score**:
   - How many exact matches of the observer pattern occur.
3. **Final Score**:
   - A weighted combination (e.g. normalized match count × compression factor).

The best spacetime is the one with **high compression and high observer density**.

---

## Implications

- **Wavefunctions** emerge naturally from high-compression bitstring universes.
- **Interference** arises as the detection mechanism for observer motifs.
- **Physics** is recast as a filtering process where informational constraints select for structured realities.

From this perspective, quantum behavior, continuity, and even time are statistical artifacts of optimal observer embedding in compressible informational structures.

---

## Future Work

- Generalize to 3D or dynamic observer shapes.
- Replace hard-coded patterns with self-replicating motifs.
- Evaluate entropy gradients and emergent fields.
---

## References

- Bayesian Inference as Information Filtering

---

© 2025 The Abstract Universe Project
