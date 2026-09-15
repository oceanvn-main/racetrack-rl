import json
from pathlib import Path
import matplotlib.pyplot as plt

root = Path("results")
files = {
    "Harbor Bend Map":    root / "exp_harbor_bend_bidirectional/01_harbor_bend_seed0.json",
    "N-Shape Map":        root / "exp_n_shape/01_n_shape_seed0.json",
    "Track 1 Benchmark": root / "exp_track_1/01_track_1_seed0.json",
    "Two-Path to Finish": root / "exp_two_path_10k/01_two_path_to_finish_seed0.json",
}

NUM_BINS = 10   # fixed bin count so all series share the same X resolution


def bin_training(training, num_bins):
    """Slice training records into num_bins equal chunks; return (episode, mean_return) pairs."""
    total = len(training)
    bin_size = max(1, total // num_bins)
    episodes, returns = [], []
    for i in range(0, total, bin_size):
        chunk = training[i:i + bin_size]
        if not chunk:
            continue
        episodes.append(chunk[-1]["episode"])
        returns.append(sum(x["return"] for x in chunk) / len(chunk))
    return episodes, returns


with plt.style.context("dark_background"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, path in files.items():
        if not path.exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        data = json.loads(path.read_text())
        episodes, returns = bin_training(data["training"], NUM_BINS)
        ax.plot(episodes, returns, label=name, linewidth=2)

    ax.set_xlabel("Episodes", fontsize=11)
    ax.set_ylabel("Mean Return", fontsize=11)
    ax.set_title("Off-Policy Monte Carlo — Benchmark Map Comparison", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path("assets/comparison.png")
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")
