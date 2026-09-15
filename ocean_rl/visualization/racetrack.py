"""Optional racetrack plots; no training dependency on Matplotlib."""
import numpy as np

def plot_results(history_episodes, history_returns, eval_results, env, output_path=None, *, show=True, algorithm_name="RL control"):
    """Optional Python renderer; training has no plotting dependency."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].plot(history_episodes, history_returns, label="Mean training return")
        axes[0].set(xlabel="Episodes", ylabel="Return", title=algorithm_name)
        axes[0].legend()
        palette = ["#15151f", "#50505d", "#f56619", "#19d94d"]
        axes[1].imshow(env.grid, cmap=ListedColormap(palette), vmin=0, vmax=3, interpolation="nearest")
        for result in eval_results:
            path = np.asarray(result["path"], dtype=float)
            # Break crash teleport segments rather than implying a drivable shortcut.
            pieces, previous = [], 0
            for crash in result.get("crash_steps", []):
                pieces.extend([path[previous:crash], np.full((1, 2), np.nan)])
                previous = crash
            pieces.append(path[previous:])
            path = np.concatenate(pieces)
            axes[1].plot(path[:, 1], path[:, 0], "-", alpha=.75,
                         label=f"{result['start']}: {'finished' if result['finished'] else 'timeout'}")
        handles, labels = axes[1].get_legend_handles_labels()
        handles += [Patch(color=color, label=label) for color, label in zip(palette, ["Wall", "Track", "Start", "Finish"])]
        axes[1].legend(handles=handles, fontsize=7)
        axes[1].set(xlabel="Column", ylabel="Row", title="Learned trajectories (acceleration noise off)")
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=160)
        if show:
            plt.show()
        plt.close(fig)



def plot_combined_learning_curves(results_dict, output_path=None, *, show=True):
    import matplotlib.pyplot as plt
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, (episodes, returns, _) in results_dict.items():
            ax.plot(episodes, returns, label=name)
        ax.set(xlabel="Episodes", ylabel="Mean return", title="RL control across imported maps")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=160)
        if show:
            plt.show()
        plt.close(fig)


