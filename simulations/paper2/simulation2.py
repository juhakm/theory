import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d


def create_wavefield(T=100, X=100, seed=42):
    """Create a random wavefield (time × space)."""
    np.random.seed(seed)
    wave = np.random.choice([-1, 1], size=(T, X))
    return wave


def create_dynamic_observer(T_obs, max_length, shape="triangle"):
    """
    Create a 1D spatial observer growing and shrinking over time.
    Returns a 2D mask: (T_obs, X_obs)
    """
    X_obs = max_length
    obs = np.zeros((T_obs, X_obs))

    for t in range(T_obs):
        if shape == "triangle":
            # Linear growth then shrink
            if t <= T_obs // 2:
                length = 1 + 2 * t  # grows by 2 each step
            else:
                length = 1 + 2 * (T_obs - t - 1)
        else:
            raise ValueError("Only 'triangle' shape supported for now")

        start = (X_obs - length) // 2
        end = start + length
        obs[t, start:end] = 1

    return obs


def match_observer(wavefield, observer):
    """
    Slide the observer across the wavefield and compute a match score
    using correlation. Returns the score matrix.
    """
    # Correlate observer with wavefield (normalized cross-correlation)
    # Observer must be flipped in both axes for correct sliding-window convolution
    corr = correlate2d(wavefield, observer[::-1, ::-1], mode="same")
    return corr


def main():
    # Parameters
    T, X = 100, 100
    T_obs = 15
    max_length = 50  # max spatial length of observer

    # Generate data
    wavefield = create_wavefield(T, X)
    observer = create_dynamic_observer(T_obs, max_length)

    # Match observer
    scores = match_observer(wavefield, observer)

    # Find best match
    scores.shape == wavefield.shape == (100, 100)
    max_idx = np.unravel_index(np.argmax(scores), scores.shape)
    max_score = scores[max_idx]
    print(
        f"Best match at (time={max_idx[0]}, space={max_idx[1]}) with score {max_score}"
    )

    # Plot results
    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    for ax in axs:
        ax.set_xlabel("Space")
        ax.set_ylabel("Time")

    axs[0].imshow(wavefield, cmap="bwr", aspect="auto")
    axs[0].set_title("Wavefield (T × X)")

    axs[1].imshow(observer, cmap="gray", aspect="auto")
    axs[1].set_title("Observer Mask (T_obs × X_obs)")

    axs[2].imshow(
        scores,
        cmap="viridis",
        aspect="auto",
        extent=[0, X, T, 0],
    )
    axs[2].set_title("Match Scores")
    axs[2].plot(max_idx[1], max_idx[0], "r+", markersize=12)

    # Overlay observer shape on match scores plot
    from matplotlib.patches import Rectangle

    # Get top-left corner of where the observer would be placed in wavefield
    t0 = max_idx[0] - observer.shape[0] // 2
    x0 = max_idx[1] - observer.shape[1] // 2

    # Add a translucent rectangle for observer shape
    rect = Rectangle(
        (x0, t0),
        observer.shape[1],
        observer.shape[0],
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axs[2].add_patch(rect)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
