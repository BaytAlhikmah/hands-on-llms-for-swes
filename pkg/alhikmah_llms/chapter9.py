"""Visualization and data helpers for Chapter 9: Attention & the Transformer.

The *mechanism* (attention itself) is always built by the student in the
notebook. These helpers only handle data loading and visualization, mirroring
how the Chapter 3 / Chapter 4 helpers work.
"""

from __future__ import annotations

import urllib.request

import numpy as np
import matplotlib.pyplot as plt


class Chapter9:
    """Data + visualization helpers for the attention / transformer chapter."""

    NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"

    # ------------------------------------------------------------------ data
    @staticmethod
    def load_names() -> list[str]:
        """Download Karpathy's names dataset (same one used in Chapter 3)."""
        response = urllib.request.urlopen(Chapter9.NAMES_URL)
        text = response.read().decode("utf-8")
        return [n.strip().lower() for n in text.strip().split("\n") if n.strip()]

    # ------------------------------------------------------------- vector viz
    @staticmethod
    def plot_vectors_2d(
        points: dict[str, tuple[float, float]],
        arrows_from: str | None = None,
        title: str = "Mixing embeddings slides the vector between meanings",
        figsize: tuple[int, int] = (7, 7),
    ) -> plt.Figure:
        """Scatter labelled 2-D vectors; optionally draw arrows from one point.

        Args:
            points: mapping label -> (x, y).
            arrows_from: if given, draw a faint arrow from this label to every
                label that starts with it plus ``+`` (e.g. ``bank`` -> ``bank+river``).
            title: figure title.
            figsize: figure size.
        """
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(points), 3)))
        for (label, (x, y)), c in zip(points.items(), colors):
            is_mix = "+" in label
            ax.scatter(
                x, y,
                s=260 if not is_mix else 150,
                color=c,
                edgecolor="black",
                linewidth=1.5,
                alpha=0.9 if not is_mix else 0.65,
                zorder=3,
            )
            ax.annotate(
                label, (x, y),
                textcoords="offset points", xytext=(10, 8),
                fontsize=11, weight="bold",
            )
        if arrows_from is not None and arrows_from in points:
            x0, y0 = points[arrows_from]
            for label, (x, y) in points.items():
                if label.startswith(arrows_from + "+"):
                    ax.annotate(
                        "", xy=(x, y), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=1.5, alpha=0.6), zorder=2,
                    )
        ax.axhline(0, color="gray", lw=0.6, alpha=0.4)
        ax.axvline(0, color="gray", lw=0.6, alpha=0.4)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("dimension 0", fontsize=10)
        ax.set_ylabel("dimension 1", fontsize=10)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()
        return fig

    # --------------------------------------------------------- attention viz
    @staticmethod
    def plot_attention(
        weights: np.ndarray,
        tokens: list[str] | None = None,
        title: str = "Attention weights",
        ax: plt.Axes | None = None,
        cmap: str = "viridis",
        annotate: bool = True,
        figsize: tuple[int, int] = (6, 5),
    ) -> plt.Axes:
        """Heatmap of an attention matrix. Row i = how much query i attends to
        each key j (rows sum to 1). Same visual language as the Chapter 3
        transition-matrix heatmap.
        """
        weights = np.asarray(weights)
        T = weights.shape[0]
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(weights, cmap=cmap, vmin=0, vmax=max(weights.max(), 1e-9))
        if tokens is None:
            tokens = [str(i) for i in range(T)]
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(tokens, fontsize=9, rotation=45, ha="right")
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_xlabel("key  (attended TO)", fontsize=10, weight="bold")
        ax.set_ylabel("query  (attending FROM)", fontsize=10, weight="bold")
        ax.set_title(title, fontsize=11, weight="bold")
        if annotate and T <= 12:
            for i in range(T):
                for j in range(weights.shape[1]):
                    v = weights[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v < 0.5 * weights.max() else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return ax

    @staticmethod
    def plot_attention_pair(
        left: np.ndarray,
        right: np.ndarray,
        tokens: list[str] | None = None,
        titles: tuple[str, str] = ("Masked (causal → GPT)", "Unmasked (→ BERT)"),
        figsize: tuple[int, int] = (13, 5),
    ) -> plt.Figure:
        """Two attention heatmaps side by side (e.g. masked vs unmasked)."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        Chapter9.plot_attention(left, tokens, titles[0], ax=axes[0])
        Chapter9.plot_attention(right, tokens, titles[1], ax=axes[1])
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_heads(
        head_weights: list[np.ndarray],
        tokens: list[str] | None = None,
        title: str = "Different heads attend to different things",
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """Grid of attention heatmaps, one per head."""
        h = len(head_weights)
        if figsize is None:
            figsize = (5.5 * h, 5)
        fig, axes = plt.subplots(1, h, figsize=figsize)
        if h == 1:
            axes = [axes]
        for i, (ax, W) in enumerate(zip(axes, head_weights)):
            Chapter9.plot_attention(W, tokens, f"Head {i}", ax=ax)
        fig.suptitle(title, fontsize=13, weight="bold", y=1.03)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_softmax_comparison(
        distributions: dict[str, np.ndarray],
        title: str = "Softmax outputs",
        figsize: tuple[int, int] = (11, 4),
    ) -> plt.Figure:
        """Bar plots of several probability vectors side by side (Part 4)."""
        n = len(distributions)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
        for ax, (name, dist) in zip(axes, distributions.items()):
            dist = np.asarray(dist)
            ax.bar(range(len(dist)), dist, color="#2E86AB",
                   edgecolor="black", linewidth=1.2, alpha=0.85)
            ax.set_title(f"{name}\nmax = {dist.max():.2f}", fontsize=11, weight="bold")
            ax.set_ylim(0, 1)
            ax.set_xlabel("key index", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("attention weight", fontsize=10, weight="bold")
        fig.suptitle(title, fontsize=13, weight="bold", y=1.04)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_loss(
        losses: list[float],
        baseline: float | None = None,
        baseline_label: str = "bigram floor",
        steps_per_point: int = 1,
        title: str = "Training loss",
        figsize: tuple[int, int] = (9, 5.5),
    ) -> plt.Figure:
        """Training loss curve with an optional horizontal baseline (Part 9)."""
        fig, ax = plt.subplots(figsize=figsize)
        xs = np.arange(len(losses)) * steps_per_point
        ax.plot(xs, losses, color="#2E86AB", lw=2, label="transformer (train)")
        if baseline is not None:
            ax.axhline(baseline, color="#C1121F", ls="--", lw=2,
                       label=f"{baseline_label} = {baseline:.3f}")
        ax.set_xlabel("training step", fontsize=11, weight="bold")
        ax.set_ylabel("cross-entropy loss (nats)", fontsize=11, weight="bold")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=11)
        plt.tight_layout()
        return fig

    @staticmethod
    def two_column_names(
        left: list[str], right: list[str],
        headers: tuple[str, str] = ("Bigram (Ch 3)", "Transformer (Ch 9)"),
    ) -> str:
        """Return a monospace two-column comparison of generated names."""
        w = max([len(headers[0])] + [len(s) for s in left]) + 4
        lines = [f"{headers[0]:<{w}}{headers[1]}",
                 f"{'-' * (w - 2):<{w}}{'-' * (len(headers[1]))}"]
        for a, b in zip(left, right):
            lines.append(f"{a:<{w}}{b}")
        return "\n".join(lines)
