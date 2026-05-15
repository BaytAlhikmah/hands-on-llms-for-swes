"""Visualization helpers for the course notebooks."""

import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def plot_matrix(matrix: list[list[float]] | npt.NDArray,
                row_labels: list[str] | None = None,
                col_labels: list[str] | None = None,
                title: str | None = None,
                fmt: str | None = None,
                cmap: str = 'Blues',
                figsize: tuple[int, int] = (8, 8),
                dpi: int = 200,
                xlabel: str = 'Column',
                ylabel: str = 'Row') -> None:
    """Plot an NxM matrix as a labeled heatmap.

    Args:
        matrix: 2D list or numpy array of values.
        row_labels: Labels for rows. Defaults to integer indices.
        col_labels: Labels for columns. Defaults to integer indices.
        title: Plot title.
        fmt: Format string for cell text (e.g. '.2f', 'd'). If None,
             auto-detects: 'd' for integers, '.2f' for floats.
        cmap: Matplotlib colormap name.
        figsize: Figure size tuple.
        dpi: Figure DPI.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
    """
    data = np.array(matrix)
    n_rows, n_cols = data.shape

    if row_labels is None:
        row_labels = [str(i) for i in range(n_rows)]
    if col_labels is None:
        col_labels = [str(i) for i in range(n_cols)]

    if fmt is None:
        fmt = 'd' if np.issubdtype(data.dtype, np.integer) else '.2f'

    # Scale font size to matrix dimensions
    font_size = max(2, min(8, 120 // max(n_rows, n_cols)))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(data, cmap=cmap)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, fontsize=font_size)
    ax.set_yticklabels(row_labels, fontsize=font_size)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    if title:
        ax.set_title(f'{title}\n', fontsize=11)

    threshold = data.max() * 0.5
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if val == 0:
                ax.text(j, i, '-', ha='center', va='center',
                        fontsize=font_size, color='lightgray')
            else:
                color = 'white' if val > threshold else 'black'
                ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                        fontsize=font_size, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()


class Chapter2:
    """Visualizations for Chapter 2: Entropy and Decision Trees."""

    @staticmethod
    def _draw_box(ax, x: float, y: float, text: str,
                  width: float = 1.5, height: float = 0.6,
                  color: str = 'lightblue', alpha: float = 1.0) -> None:
        """Draw a rounded box with text."""
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color,
                            linewidth=2, alpha=alpha)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11,
               weight='bold', alpha=alpha)

    @staticmethod
    def _draw_arrow(ax, x1: float, y1: float, x2: float, y2: float,
                    label: str = '', color: str = 'black',
                    alpha: float = 1.0, width: float = 2) -> None:
        """Draw an arrow with optional label."""
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=width, color=color, alpha=alpha)
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.3, label, ha='center', fontsize=10,
                   weight='bold', color=color, alpha=alpha)

    @classmethod
    def draw_uniform_tree(cls, figsize: tuple[int, int] = (14, 8)) -> plt.Figure:
        """Draw the full binary search tree for uniform distribution (1-8).

        Args:
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Root
        cls._draw_box(ax, 7, 7, '1-8\np=0.125 each', width=2, color='#FFE6E6')
        ax.text(7, 7.8, 'Q1: Is it > 4?', ha='center', fontsize=11,
               weight='bold', color='red')

        # Level 1
        cls._draw_arrow(ax, 6.5, 6.7, 3.5, 5.3, 'No', 'green')
        cls._draw_arrow(ax, 7.5, 6.7, 10.5, 5.3, 'Yes', 'red')

        cls._draw_box(ax, 3.5, 5, '1-4\np=0.125 each', width=1.8, color='#E6F3FF')
        ax.text(3.5, 5.7, 'Q2: Is it > 2?', ha='center', fontsize=10, color='red')

        cls._draw_box(ax, 10.5, 5, '5-8\np=0.125 each', width=1.8, color='#E6F3FF')
        ax.text(10.5, 5.7, 'Q2: Is it > 6?', ha='center', fontsize=10, color='red')

        # Level 2 - left side
        cls._draw_arrow(ax, 2.8, 4.7, 1.8, 3.3, 'No', 'green')
        cls._draw_arrow(ax, 4.2, 4.7, 5.2, 3.3, 'Yes', 'red')

        cls._draw_box(ax, 1.8, 3, '1-2\np=0.125 each', width=1.5, color='#FFF4E6')
        ax.text(1.8, 3.6, 'Q3: Is it 1?', ha='center', fontsize=9, color='red')

        cls._draw_box(ax, 5.2, 3, '3-4\np=0.125 each', width=1.5, color='#FFF4E6')
        ax.text(5.2, 3.6, 'Q3: Is it 3?', ha='center', fontsize=9, color='red')

        # Level 2 - right side
        cls._draw_arrow(ax, 9.8, 4.7, 8.8, 3.3, 'No', 'green')
        cls._draw_arrow(ax, 11.2, 4.7, 12.2, 3.3, 'Yes', 'red')

        cls._draw_box(ax, 8.8, 3, '5-6\np=0.125 each', width=1.5, color='#FFF4E6')
        ax.text(8.8, 3.6, 'Q3: Is it 5?', ha='center', fontsize=9, color='red')

        cls._draw_box(ax, 12.2, 3, '7-8\np=0.125 each', width=1.5, color='#FFF4E6')
        ax.text(12.2, 3.6, 'Q3: Is it 7?', ha='center', fontsize=9, color='red')

        # Level 3 - final answers
        y_final = 1.5
        finals = [
            (0.8, '1', 1.8), (2.8, '2', 1.8),
            (4.2, '3', 5.2), (5.8, '4', 5.2),
            (7.8, '5', 8.8), (9.8, '6', 8.8),
            (11.2, '7', 12.2), (12.8, '8', 12.2),
        ]

        for x, label, parent_x in finals:
            cls._draw_box(ax, x, y_final, label, width=0.6, height=0.5,
                         color='#90EE90')
            cls._draw_arrow(ax, parent_x, 2.7, x, y_final + 0.3, '', 'gray')

        # Summary
        ax.text(7, 0.5, 'All paths take exactly 3 questions → Entropy = log₂(8) = 3.0 bits',
               ha='center', fontsize=12, weight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.title('Uniform Distribution: Binary Search Tree',
                 fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        return fig

    @classmethod
    def draw_biased_tree(cls, figsize: tuple[int, int] = (14, 8)) -> plt.Figure:
        """Draw the optimized tree for biased distribution (50% is 1).

        Args:
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Root
        cls._draw_box(ax, 7, 7, '1-8', width=2.5, color='#FFE6E6')
        ax.text(7, 7.8, 'Q1: Is it 1?', ha='center', fontsize=11,
               weight='bold', color='red')
        ax.text(7, 6.3, '1: p=0.5, others: p≈0.0714 (= 0.5 / 7) each', ha='center',
               fontsize=9, color='purple', style='italic')

        # Level 1
        cls._draw_arrow(ax, 6, 6.7, 3, 5.3, 'Yes (p=0.5)', 'green')
        cls._draw_arrow(ax, 8, 6.7, 10.5, 5.3, 'No (p=0.5)', 'red')

        # Left branch - DONE!
        cls._draw_box(ax, 3, 5, '1', width=1.2, height=0.8, color='#90EE90')
        ax.text(3, 4.5, 'p=0.5', ha='center', fontsize=8,
               style='italic', color='purple')
        ax.text(3, 4, '✓ Done in 1 question!', ha='center', fontsize=10,
               weight='bold', color='green')
        ax.text(3, 3.5, 'Questions: 1', ha='center', fontsize=9,
               color='darkgreen',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # Right branch
        cls._draw_box(ax, 10.5, 5, '2-8\n(7 options)', width=2, color='#E6F3FF')
        ax.text(10.5, 4.3, 'p≈0.0714 (= 0.5 / 7) each', ha='center', fontsize=8,
               style='italic', color='purple')
        ax.text(10.5, 5.7, 'Q2: Is it > 5?', ha='center', fontsize=10, color='red')

        # Level 2
        cls._draw_arrow(ax, 9.8, 4.7, 8.5, 3.3, 'No', 'green')
        cls._draw_arrow(ax, 11.2, 4.7, 12.5, 3.3, 'Yes', 'red')

        cls._draw_box(ax, 8.5, 3, '2-5', width=1.5, color='#FFF4E6')
        ax.text(8.5, 3.6, 'Q3: Is it > 3?', ha='center', fontsize=9, color='red')

        cls._draw_box(ax, 12.5, 3, '6-8', width=1.5, color='#FFF4E6')
        ax.text(12.5, 3.6, 'Q3: Is it > 7?', ha='center', fontsize=9, color='red')

        # Show continuation
        ax.text(8.5, 1.8, '...continues\n', ha='center',
               fontsize=9, color='gray', style='italic')
        ax.text(12.5, 1.8, '...continues\n', ha='center',
               fontsize=9, color='gray', style='italic')

        # Summary
        summary_text = (
            'Expected questions = 0.5 × (1 question) + 0.5 × (1 + log₂(7) questions)\n'
            '                   = 0.5 × 1 + 0.5 × (1 + 2.807)\n'
            '                   = 0.5 + 0.5 × 3.807 = 2.4035 questions\n'
            'Entropy = 2.4035 bits (vs 3.0 for uniform)'
        )
        ax.text(7, 0.7, summary_text, ha='center', fontsize=10, weight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
               family='monospace')

        plt.title('Biased Distribution: Optimized Tree (Check Most Likely First)',
                 fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        return fig

    @classmethod
    def draw_comparison(cls, figsize: tuple[int, int] = (16, 6)) -> plt.Figure:
        """Draw side-by-side comparison of uniform vs biased trees.

        Args:
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Uniform - simplified
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        ax1.set_title('Uniform (p=0.125 each)', fontsize=14, weight='bold', pad=10)

        # Draw simplified tree for uniform
        levels = [
            [(5, 8.5, '1-8', 'lightcoral')],
            [(2.5, 6, '1-4', 'lightblue'), (7.5, 6, '5-8', 'lightblue')],
            [(1.2, 3.5, '1-2', 'lightyellow'), (3.8, 3.5, '3-4', 'lightyellow'),
             (6.2, 3.5, '5-6', 'lightyellow'), (8.8, 3.5, '7-8', 'lightyellow')],
            [(0.5, 1, '1', 'lightgreen'), (1.9, 1, '2', 'lightgreen'),
             (3.1, 1, '3', 'lightgreen'), (4.5, 1, '4', 'lightgreen'),
             (5.5, 1, '5', 'lightgreen'), (6.9, 1, '6', 'lightgreen'),
             (8.1, 1, '7', 'lightgreen'), (9.5, 1, '8', 'lightgreen')]
        ]

        for level in levels:
            for x, y, label, color in level:
                circle = plt.Circle((x, y), 0.4, color=color, ec='black', linewidth=2)
                ax1.add_patch(circle)
                ax1.text(x, y, label, ha='center', va='center',
                        fontsize=9, weight='bold')

        # Add depth markers
        for i, label in enumerate(['Depth 0', 'Depth 1', 'Depth 2', 'Depth 3']):
            y = 8.5 - i * 2.5
            ax1.text(9.5, y, label, fontsize=8, style='italic', color='gray')

        ax1.text(5, 0.2, 'All paths: 3 questions\nEntropy = 3.0 bits',
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        # Biased - simplified
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        ax2.set_title('Biased (p=0.5 is 1, rest p≈0.0714 (= 0.5 / 7))', fontsize=14,
                     weight='bold', pad=10)

        # Root
        circle = plt.Circle((5, 8.5), 0.4, color='lightcoral', ec='black', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(5, 8.5, '1-8', ha='center', va='center', fontsize=9, weight='bold')

        # Level 1 - answer 1 (p=0.5)
        circle = plt.Circle((2, 6), 0.5, color='lightgreen', ec='green', linewidth=3)
        ax2.add_patch(circle)
        ax2.text(2, 6, '1', ha='center', va='center', fontsize=12, weight='bold')
        ax2.text(2, 5.2, 'p=0.5\n1 question', ha='center', fontsize=9,
                color='green', weight='bold')
        ax2.arrow(4.7, 8.2, -2.3, -1.8, head_width=0.2, head_length=0.2,
                 fc='green', ec='green', linewidth=2)

        # Level 1 - rest (p=0.5 total)
        circle = plt.Circle((8, 6), 0.4, color='lightblue', ec='black', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(8, 6, '2-8', ha='center', va='center', fontsize=9, weight='bold')
        ax2.text(8, 5.2, 'p=0.5 total', ha='center', fontsize=8,
                style='italic', color='purple')
        ax2.arrow(5.3, 8.2, 2.3, -1.8, head_width=0.2, head_length=0.2,
                 fc='red', ec='red', linewidth=2)

        # Show continuation
        ax2.text(8, 3.5, '...', ha='center', fontsize=20, color='gray')
        ax2.text(8, 2.8, 'log₂(7) ≈ 2.807\nmore questions', ha='center',
                fontsize=9, style='italic', color='gray')
        ax2.text(8, 1.8, 'Total: 1 + 2.807 = 3.807', ha='center', fontsize=9,
                color='gray')

        # Depth markers
        ax2.text(9.5, 8.5, 'Depth 0', fontsize=8, style='italic', color='gray')
        ax2.text(9.5, 6, 'Depth 1', fontsize=8, style='italic', color='gray')

        # Summary
        summary = (
            'Average:\n'
            '0.5×1 + 0.5×3.807 = 2.4035\n'
            'Entropy = 2.4035 bits'
        )
        ax2.text(5, 0.5, summary, ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.tight_layout()
        return fig

    @classmethod
    def draw_biased_tree_step(cls, step: int = 0,
                             figsize: tuple[int, int] = (12, 8)) -> plt.Figure:
        """Draw the biased tree with progressive reveal (for interactive teaching).

        Args:
            step: Step number (0-4)
                0: Show initial state
                1: Ask Q1
                2: Show 50% path (done in 1)
                3: Show 50% path (continues)
                4: Show final calculation
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Root (always visible)
        cls._draw_box(ax, 6, 8, '1-8', width=2, color='#FFE6E6')
        ax.text(6, 8.8, 'Start: Numbers 1-8', ha='center', fontsize=12,
               weight='bold')
        ax.text(6, 7.3, '1: p=0.5  |  2-8: p≈0.0714 (= 0.5 / 7) each', ha='center',
               fontsize=9, style='italic', color='purple')

        if step >= 1:
            ax.text(6, 9.5, 'Q1: "Is it 1?"', ha='center', fontsize=13,
                   weight='bold', color='red',
                   bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

        if step >= 2:
            # Left branch
            cls._draw_arrow(ax, 5.5, 7.7, 3.5, 5.5, 'YES (p=0.5)', 'green', width=3)
            cls._draw_box(ax, 3, 5, '1', width=1.5, height=1, color='#90EE90')
            ax.text(3, 3.5, '✓ DONE!\n1 question', ha='center', fontsize=11,
                   weight='bold', color='green',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        if step >= 3:
            # Right branch
            cls._draw_arrow(ax, 6.5, 7.7, 9, 5.5, 'NO (p=0.5)', 'red', width=3)
            cls._draw_box(ax, 9, 5, '2-8\n(7 options)', width=2, color='#FFE6E6')
            ax.text(9, 3.5, 'Need log₂(7) ≈ 2.807\nmore questions\nTotal: 1 + 2.807 = 3.807',
                   ha='center', fontsize=10, style='italic', color='gray',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

        if step >= 4:
            # Final calculation
            calc_text = (
                'Expected Questions:\n\n'
                '0.5 × 1 question     = 0.5\n'
                '0.5 × 3.807 questions  = 1.9035\n'
                '―――――――――――――――――\n'
                'Average = 2.4035 questions\n\n'
                'Entropy = 2.4035 bits'
            )
            ax.text(6, 1.5, calc_text, ha='center', fontsize=11,
                   family='monospace', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.title(f'Biased Distribution - Step {step}/4',
                 fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_distributions(figsize: tuple[int, int] = (14, 5)) -> plt.Figure:
        """Plot side-by-side probability distributions with entropy values.

        Args:
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        numbers = list(range(1, 9))
        uniform_probs = [1/8] * 8
        biased_probs = [0.5] + [0.5/7] * 7

        # Uniform
        bars1 = ax1.bar(numbers, uniform_probs, color='steelblue',
                       edgecolor='black', linewidth=2)
        ax1.axhline(1/8, color='red', linestyle='--', linewidth=2,
                   label='p=0.125 baseline')
        ax1.set_xlabel('Number', fontsize=12, weight='bold')
        ax1.set_ylabel('Probability', fontsize=12, weight='bold')
        ax1.set_title('Uniform Distribution\nEntropy = 3.0 bits',
                     fontsize=13, weight='bold')
        ax1.set_ylim(0, 0.6)
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()

        # Add probability labels
        for i, (x, y) in enumerate(zip(numbers, uniform_probs)):
            ax1.text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9)

        # Biased
        colors = ['green'] + ['steelblue'] * 7
        bars2 = ax2.bar(numbers, biased_probs, color=colors,
                       edgecolor='black', linewidth=2)
        ax2.axhline(1/8, color='red', linestyle='--', linewidth=2,
                   alpha=0.5, label='p=0.125 (uniform)')
        ax2.set_xlabel('Number', fontsize=12, weight='bold')
        ax2.set_ylabel('Probability', fontsize=12, weight='bold')
        ax2.set_title('Biased Distribution\nEntropy = 2.4035 bits',
                     fontsize=13, weight='bold')
        ax2.set_ylim(0, 0.6)
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend()

        # Add probability labels
        for i, (x, y) in enumerate(zip(numbers, biased_probs)):
            ax2.text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9,
                    weight='bold' if i == 0 else 'normal')

        plt.tight_layout()
        return fig

    @staticmethod
    def print_entropy_comparison() -> None:
        """Print numerical entropy comparison between uniform and biased distributions."""
        # Uniform distribution
        uniform_probs = [1/8] * 8
        uniform_entropy = -sum(p * math.log2(p) for p in uniform_probs)

        # Biased distribution
        biased_probs = [0.5] + [0.5/7] * 7
        biased_entropy = -sum(p * math.log2(p) for p in biased_probs if p > 0)

        print("=" * 50)
        print("ENTROPY COMPARISON")
        print("=" * 50)
        print(f"\nUniform (each number p=0.125):")
        print(f"  Entropy = {uniform_entropy:.4f} bits")
        print(f"  Average questions needed: {uniform_entropy:.1f}")

        print(f"\nBiased (1 is p=0.5, rest p≈0.0714 (= 0.5 / 7) each):")
        print(f"  Entropy = {biased_entropy:.4f} bits")
        print(f"  Average questions needed: {biased_entropy:.1f}")

        print(f"\nDifference: {uniform_entropy - biased_entropy:.4f} bits")
        print(f"Reduction: {(1 - biased_entropy/uniform_entropy)*100:.1f}%")

        print("\n" + "=" * 50)
        print("KEY INSIGHT:")
        print("=" * 50)
        print("More predictable = Lower entropy = Fewer questions needed")
        print("=" * 50)

    @staticmethod
    def plot_compression_example(figsize: tuple[int, int] = (14, 10)) -> plt.Figure:
        """Visualize compression with frequency-based encoding.

        Args:
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1], hspace=0.4, wspace=0.3)

        # File visualization
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('File: AAAAAABBCD (10 characters)', fontsize=13, weight='bold', pad=10)

        # Draw file as blocks
        colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 'D': '#FFA07A'}
        file_content = 'AAAAAABBCD'
        for i, char in enumerate(file_content):
            rect = plt.Rectangle((i, 0.3), 0.9, 0.4, facecolor=colors[char],
                                edgecolor='black', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(i + 0.45, 0.5, char, ha='center', va='center',
                    fontsize=14, weight='bold', color='white')

        # Frequency bar chart
        ax2 = fig.add_subplot(gs[1, 0])
        chars = ['A', 'B', 'C', 'D']
        freqs = [6/10, 2/10, 1/10, 1/10]
        bars = ax2.bar(chars, freqs, color=[colors[c] for c in chars],
                      edgecolor='black', linewidth=2)
        ax2.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax2.set_title('Character Frequencies', fontsize=12, weight='bold')
        ax2.set_ylim(0, 0.7)
        ax2.grid(axis='y', alpha=0.3)

        # Add percentage labels on bars
        for bar, freq in zip(bars, freqs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{freq:.0%}', ha='center', va='bottom', fontsize=10, weight='bold')

        # Encoding tree
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)
        ax3.axis('off')
        ax3.set_title('Encoding Tree', fontsize=12, weight='bold')

        # Draw binary tree
        # Root
        root_circle = plt.Circle((5, 8), 0.4, color='lightgray', ec='black', linewidth=2)
        ax3.add_patch(root_circle)
        ax3.text(5, 8, 'Root', ha='center', va='center', fontsize=8, weight='bold')

        # A branch (left, 60%)
        ax3.plot([5, 2.5], [7.7, 5.3], 'k-', linewidth=2)
        ax3.text(3.5, 6.8, '0', ha='center', fontsize=10, weight='bold', color='green')
        a_circle = plt.Circle((2.5, 5), 0.5, color=colors['A'], ec='black', linewidth=2)
        ax3.add_patch(a_circle)
        ax3.text(2.5, 5, 'A\n60%', ha='center', va='center',
                fontsize=9, weight='bold', color='white')
        ax3.text(2.5, 3.8, 'Code: 0\n1 bit', ha='center', fontsize=8,
                style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow'))

        # Right branch splits into B, C, D
        ax3.plot([5, 7.5], [7.7, 5.3], 'k-', linewidth=2)
        ax3.text(6.5, 6.8, '1', ha='center', fontsize=10, weight='bold', color='red')
        inner_circle = plt.Circle((7.5, 5), 0.4, color='lightgray', ec='black', linewidth=2)
        ax3.add_patch(inner_circle)

        # B (left of inner)
        ax3.plot([7.5, 6], [4.7, 2.8], 'k-', linewidth=1.5)
        ax3.text(6.5, 4, '0', ha='center', fontsize=9, weight='bold', color='green')
        b_circle = plt.Circle((6, 2.5), 0.45, color=colors['B'], ec='black', linewidth=2)
        ax3.add_patch(b_circle)
        ax3.text(6, 2.5, 'B\n20%', ha='center', va='center',
                fontsize=8, weight='bold', color='white')
        ax3.text(6, 1.3, 'Code: 10\n2 bits', ha='center', fontsize=7,
                style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow'))

        # Right of inner splits to C and D
        ax3.plot([7.5, 9], [4.7, 2.8], 'k-', linewidth=1.5)
        ax3.text(8.5, 4, '1', ha='center', fontsize=9, weight='bold', color='red')
        inner2_circle = plt.Circle((9, 2.5), 0.35, color='lightgray', ec='black', linewidth=1.5)
        ax3.add_patch(inner2_circle)

        # C
        ax3.plot([9, 8.2], [2.2, 0.8], 'k-', linewidth=1.5)
        ax3.text(8.5, 1.6, '0', ha='center', fontsize=8, weight='bold', color='green')
        c_circle = plt.Circle((8.2, 0.5), 0.35, color=colors['C'], ec='black', linewidth=1.5)
        ax3.add_patch(c_circle)
        ax3.text(8.2, 0.5, 'C\n10%', ha='center', va='center',
                fontsize=7, weight='bold', color='white')
        ax3.text(7.2, -0.5, '110\n3 bits', ha='center', fontsize=7, style='italic')

        # D
        ax3.plot([9, 9.8], [2.2, 0.8], 'k-', linewidth=1.5)
        ax3.text(9.5, 1.6, '1', ha='center', fontsize=8, weight='bold', color='red')
        d_circle = plt.Circle((9.8, 0.5), 0.35, color=colors['D'], ec='black', linewidth=1.5)
        ax3.add_patch(d_circle)
        ax3.text(9.8, 0.5, 'D\n10%', ha='center', va='center',
                fontsize=7, weight='bold', color='white')
        ax3.text(9.8, -0.5, '111\n3 bits', ha='center', fontsize=7, style='italic')

        # Encoding comparison
        ax4 = fig.add_subplot(gs[2, :])
        ax4.set_xlim(0, 22)
        ax4.set_ylim(0, 3)
        ax4.axis('off')

        # Naive encoding
        ax4.text(0.5, 2.5, 'Naive (2 bits/char):', fontsize=11, weight='bold')
        naive_bits = '00 00 00 00 00 00 01 01 10 11'
        for i, bit_pair in enumerate(naive_bits.split()):
            x_pos = 6 + i * 1.5
            rect = plt.Rectangle((x_pos, 2.2), 1.3, 0.5, facecolor='#FFB6C1',
                                edgecolor='black', linewidth=1)
            ax4.add_patch(rect)
            ax4.text(x_pos + 0.65, 2.45, bit_pair, ha='center', va='center',
                    fontsize=8, family='monospace', weight='bold')
        ax4.text(21, 2.45, '= 20 bits', ha='left', va='center', fontsize=10, weight='bold')

        # Optimal encoding
        ax4.text(0.5, 1, 'Optimal (entropy):', fontsize=11, weight='bold')
        optimal_bits = ['0', '0', '0', '0', '0', '0', '10', '10', '110', '111']
        x_pos = 6
        for bits in optimal_bits:
            width = len(bits) * 0.6
            rect = plt.Rectangle((x_pos, 0.7), width, 0.5, facecolor='#90EE90',
                                edgecolor='black', linewidth=1)
            ax4.add_patch(rect)
            ax4.text(x_pos + width/2, 0.95, bits, ha='center', va='center',
                    fontsize=8, family='monospace', weight='bold')
            x_pos += width + 0.2
        ax4.text(21, 0.95, '= 16 bits', ha='left', va='center', fontsize=10, weight='bold')

        # Summary
        ax4.text(11, 0.1, 'Savings: 20% compression  |  Entropy: 1.57 bits → Actual: 1.6 bits/char',
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.suptitle('Compression via Entropy-Based Encoding', fontsize=15, weight='bold', y=0.98)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_surprise_curve(figsize: tuple[int, int] = (12, 7)) -> plt.Figure:
        """Plot probability vs surprise relationship.

        Args:
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Generate curve
        probs = np.linspace(0.001, 1, 1000)
        surprise = np.log2(1 / probs)

        # Plot main curve
        ax.plot(probs, surprise, linewidth=3, color='#2E86AB', label='Surprise = log₂(1/p)')
        ax.fill_between(probs, surprise, alpha=0.2, color='#2E86AB')

        # Mark key points
        key_points = [
            (0.99, 'Sun rises\np=0.99', 'green', 'below'),
            (0.50, 'Coin flip\np=0.5', 'orange', 'above'),
            (0.000001, 'Win lottery\np=0.000001', 'red', 'above'),
        ]

        for p, label, color, pos in key_points:
            s = math.log2(1/p)
            ax.scatter([p], [s], s=200, color=color, edgecolor='black',
                      linewidth=2, zorder=10, alpha=0.8)
            ax.axvline(p, color=color, linestyle='--', alpha=0.3, linewidth=1.5)
            ax.axhline(s, color=color, linestyle='--', alpha=0.3, linewidth=1.5)

            # Position label
            if pos == 'above':
                ax.annotate(f'{label}\nSurprise: {s:.2f} bits',
                           xy=(p, s), xytext=(p, s + 2.5),
                           ha='center', fontsize=10, weight='bold',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))
            else:
                ax.annotate(f'{label}\nSurprise: {s:.4f} bits',
                           xy=(p, s), xytext=(p + 0.15, s + 1),
                           ha='left', fontsize=10, weight='bold',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))

        # Styling
        ax.set_xlabel('Probability (p)', fontsize=13, weight='bold')
        ax.set_ylabel('Surprise (bits)', fontsize=13, weight='bold')
        ax.set_title('The Relationship Between Probability and Surprise',
                    fontsize=14, weight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-1, 22)

        # Add key insight box
        insight = (
            'Key Insight:\n'
            'Rare events (low probability) → High surprise\n'
            'Common events (high probability) → Low surprise\n'
            'Surprise = log₂(1/p) = -log₂(p)'
        )
        ax.text(0.5, 17, insight, ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
               weight='bold')

        ax.legend(loc='upper right', fontsize=11)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_surprise_comparison(figsize: tuple[int, int] = (12, 6)) -> plt.Figure:
        """Compare surprise levels for different events.

        Args:
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Events with probabilities
        events = [
            ('Sun rises tomorrow', 0.99, '#2ECC40'),
            ('Your birthday this year', 1/365, '#0074D9'),
            ('Coin flip is heads', 0.50, '#FF851B'),
            ('Roll a 6 on a die', 1/6, '#B10DC9'),
            ('Draw a specific card', 1/52, '#FF4136'),
            ('Win the lottery', 0.000001, '#85144b'),
        ]

        # Calculate surprises
        labels = [e[0] for e in events]
        probs = [e[1] for e in events]
        colors = [e[2] for e in events]
        surprises = [math.log2(1/p) for p in probs]

        # Sort by surprise
        sorted_indices = sorted(range(len(surprises)), key=lambda i: surprises[i])
        labels = [labels[i] for i in sorted_indices]
        probs = [probs[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        surprises = [surprises[i] for i in sorted_indices]

        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, surprises, color=colors, edgecolor='black', linewidth=2)

        # Add probability and surprise values
        for i, (bar, prob, surprise) in enumerate(zip(bars, probs, surprises)):
            width = bar.get_width()
            # Surprise value at end of bar
            ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                   f'{surprise:.2f} bits', va='center', fontsize=10, weight='bold')
            # Probability value inside/before bar
            if width > 3:
                ax.text(width - 0.5, bar.get_y() + bar.get_height()/2,
                       f'p={prob:.6f}' if prob < 0.01 else f'p={prob:.3f}',
                       va='center', ha='right', fontsize=9, color='white', weight='bold')
            else:
                ax.text(0.1, bar.get_y() + bar.get_height()/2,
                       f'p={prob:.6f}' if prob < 0.01 else f'p={prob:.3f}',
                       va='center', ha='left', fontsize=9, weight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Surprise (bits)', fontsize=12, weight='bold')
        ax.set_title('Surprise Levels for Different Events', fontsize=14, weight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(surprises) + 2)

        # Add interpretation
        ax.text(max(surprises)/2, -0.8,
               'Lower surprise = more expected | Higher surprise = more shocking',
               ha='center', fontsize=11, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        return fig
    @staticmethod
    def plot_binary_entropy(figsize: tuple[int, int] = (12, 7)) -> plt.Figure:
        """Plot the binary entropy function H(p) = -p·log₂(p) - (1-p)·log₂(1-p).

        This is the most fundamental entropy curve in information theory.

        Args:
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Generate the binary entropy curve
        p = np.linspace(0.001, 0.999, 1000)
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        # Plot the curve
        ax.plot(p, h, linewidth=4, color='#2E86AB', label='H(p) = -p·log₂(p) - (1-p)·log₂(1-p)')
        ax.fill_between(p, h, alpha=0.2, color='#2E86AB')

        # Calculate actual entropy values for marked points
        marked_points = [
            (0.5, 1.0, 'Coin flip\np=0.5\nH=1.0 bit\n(Maximum!)', 'red'),
            (0.1, -0.1 * math.log2(0.1) - 0.9 * math.log2(0.9), 'Very biased\np=0.1\nH≈0.47 bits', 'orange'),
            (0.9, -0.9 * math.log2(0.9) - 0.1 * math.log2(0.1), 'Very biased\np=0.9\nH≈0.47 bits', 'orange'),
        ]

        for p_val, h_val, label, color in marked_points:
            ax.scatter([p_val], [h_val], s=250, color=color, edgecolor='black',
                      linewidth=3, zorder=10, alpha=0.9)
            ax.axvline(p_val, color=color, linestyle='--', alpha=0.3, linewidth=1.5)
            ax.axhline(h_val, color=color, linestyle='--', alpha=0.3, linewidth=1.5)

            # Position labels to avoid overlap
            if p_val == 0.5:
                y_offset = -0.30
            elif p_val < 0.5:
                y_offset = -0.15
            else:
                y_offset = -0.15

            ax.annotate(label, xy=(p_val, h_val),
                       xytext=(p_val, h_val + y_offset),
                       ha='center', fontsize=9, weight='bold',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2) if abs(y_offset) > 0.05 else None)

        # Styling
        ax.set_xlabel('Probability (p)', fontsize=13, weight='bold')
        ax.set_ylabel('Entropy H(p) in bits', fontsize=13, weight='bold')
        ax.set_title('Binary Entropy Function: The Most Important Curve in Information Theory',
                    fontsize=14, weight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.15)

        # Add key insights
        insight = (
            'Key Insights:\n'
            '• Maximum entropy at p=0.5 (equal probabilities = maximum uncertainty)\n'
            '• Zero entropy at p=0 or p=1 (certain outcome = no uncertainty)\n'
            '• Symmetric around p=0.5\n'
            '• This curve is the foundation of all entropy calculations!'
        )
        ax.text(0.5, -0.35, insight, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4),
               transform=ax.transAxes)

        ax.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_ternary_entropy(figsize: tuple[int, int] = (12, 10)) -> plt.Figure:
        """Plot entropy for a 3-outcome distribution on a ternary/simplex plot.

        Shows how entropy varies across all possible probability distributions
        for 3 outcomes (e.g., Rock-Paper-Scissors).

        Args:
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create a grid of points in the probability simplex
        # p1 + p2 + p3 = 1, so we can parameterize with (p1, p2) and compute p3
        resolution = 100
        p1_vals = np.linspace(0, 1, resolution)
        p2_vals = np.linspace(0, 1, resolution)

        # Create meshgrid
        P1, P2 = np.meshgrid(p1_vals, p2_vals)

        # Compute p3 and entropy
        P3 = 1 - P1 - P2
        H = np.zeros_like(P1)

        for i in range(resolution):
            for j in range(resolution):
                p1, p2, p3 = P1[i, j], P2[i, j], P3[i, j]
                # Only valid if all probabilities are non-negative
                if p1 >= 0 and p2 >= 0 and p3 >= 0 and abs(p1 + p2 + p3 - 1) < 0.01:
                    # Calculate entropy
                    entropy_val = 0
                    for p in [p1, p2, p3]:
                        if p > 1e-10:  # Avoid log(0)
                            entropy_val -= p * math.log2(p)
                    H[i, j] = entropy_val
                else:
                    H[i, j] = np.nan  # Invalid region

        # Plot as contour/heatmap
        # Mask invalid regions
        H_masked = np.ma.masked_invalid(H)

        # Create the heatmap
        im = ax.contourf(P1, P2, H_masked, levels=20, cmap='RdYlGn_r', alpha=0.8)
        contours = ax.contour(P1, P2, H_masked, levels=10, colors='black',
                             alpha=0.3, linewidths=1)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

        # Draw the simplex boundary
        # Triangle vertices: (1,0), (0,1), (0,0)
        triangle = plt.Polygon([(1, 0), (0, 1), (0, 0)], fill=False,
                              edgecolor='black', linewidth=3)
        ax.add_patch(triangle)

        # Mark special points
        special_points = [
            (1/3, 1/3, 'Uniform\n(1/3, 1/3, 1/3)\nH=1.585 bits\nMaximum!', 'red', 1.585),
            (0.6, 0.3, 'Biased RPS\n(0.6, 0.3, 0.1)\nH≈1.30 bits', 'orange',
             -0.6*math.log2(0.6) - 0.3*math.log2(0.3) - 0.1*math.log2(0.1)),
            (1.0, 0.0, 'Always p₁\n(1, 0, 0)\nH=0 bits', 'blue', 0),
            (0.0, 1.0, 'Always p₂\n(0, 1, 0)\nH=0 bits', 'blue', 0),
            (0.5, 0.5, 'Two-way\n(0.5, 0.5, 0)\nH=1.0 bit', 'green', 1.0),
        ]

        for p1, p2, label, color, h_val in special_points:
            ax.scatter([p1], [p2], s=300, color=color, edgecolor='black',
                      linewidth=3, zorder=10, marker='*', alpha=0.9)
            # Position labels to avoid overlap
            if p1 == 1/3 and p2 == 1/3:
                offset = (0.05, 0.1)
            elif p1 == 1.0:
                offset = (0.05, -0.05)
            elif p2 == 1.0:
                offset = (-0.15, 0.02)
            elif p1 == 0.6:
                offset = (0.05, 0.05)
            else:
                offset = (0.05, -0.1)

            ax.annotate(label, xy=(p1, p2),
                       xytext=(p1 + offset[0], p2 + offset[1]),
                       ha='left', fontsize=9, weight='bold',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.4),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))

        # Labels for axes
        ax.set_xlabel('Probability p₁', fontsize=12, weight='bold')
        ax.set_ylabel('Probability p₂', fontsize=12, weight='bold')
        ax.set_title('Ternary Entropy: All Possible 3-Outcome Distributions\n(p₁ + p₂ + p₃ = 1)',
                    fontsize=14, weight='bold', pad=15)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Entropy (bits)', shrink=0.8)
        cbar.set_label('Entropy (bits)', fontsize=11, weight='bold')

        # Set axis limits to show the triangle
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')

        # Add constraint label
        ax.text(0.5, -0.15, 'Constraint: p₁ + p₂ + p₃ = 1  (inside triangle)\n' +
               'p₃ is determined by p₁ and p₂',
               ha='center', fontsize=10, style='italic',
               transform=ax.transAxes)

        # Add insight box
        insight = (
            'Key Insights:\n'
            '• Center (uniform) = Maximum entropy\n'
            '• Corners (certain) = Zero entropy\n'
            '• Perfect symmetry!\n'
            '• Any point in triangle = valid distribution'
        )
        ax.text(0.98, 0.02, insight, ha='right', va='bottom', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
               transform=ax.transAxes)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_ternary_entropy_3d_plotly() -> go.Figure:
        """Plot entropy for 3-outcome distribution as an interactive Plotly 3D surface.

        Works in Google Colab and Jupyter with no backend setup required.

        Returns:
            Plotly Figure with 3D surface.
        """
        resolution = 50
        p1_vals = np.linspace(0, 1, resolution)
        p2_vals = np.linspace(0, 1, resolution)
        P1, P2 = np.meshgrid(p1_vals, p2_vals)
        P3 = 1 - P1 - P2
        H = np.zeros_like(P1)

        for i in range(resolution):
            for j in range(resolution):
                p1, p2, p3 = P1[i, j], P2[i, j], P3[i, j]
                if p1 >= 0 and p2 >= 0 and p3 >= 0 and abs(p1 + p2 + p3 - 1) < 0.01:
                    entropy_val = 0
                    for p in [p1, p2, p3]:
                        if p > 1e-10:
                            entropy_val -= p * np.log2(p)
                    H[i, j] = entropy_val
                else:
                    H[i, j] = np.nan

        surface = go.Surface(
            x=P1, y=P2, z=H,
            colorscale='RdYlGn_r',
            customdata=np.stack([1 - P1 - P2], axis=-1),
            hovertemplate='p₁: %{x:.3f}<br>p₂: %{y:.3f}<br>p₃: %{customdata[0]:.3f}<br>H: %{z:.3f}<extra></extra>'
        )

        special_points = [
            (1/3, 1/3, -1/3*np.log2(1/3) - 1/3*np.log2(1/3) - 1/3*np.log2(1/3), 'Uniform (1/3,1/3,1/3) H=1.585', 'red'),
            (1.0, 0.0, 0.0, 'Certain p₁ (1,0,0) H=0', 'blue'),
            (0.0, 1.0, 0.0, 'Certain p₂ (0,1,0) H=0', 'blue'),
            (0.0, 0.0, 0.0, 'Certain p₃ (0,0,1) H=0', 'blue'),
        ]

        markers = go.Scatter3d(
            x=[p[0] for p in special_points],
            y=[p[1] for p in special_points],
            z=[p[2] for p in special_points],
            mode='markers+text',
            marker=dict(size=10, symbol='diamond', color=[p[4] for p in special_points], line=dict(width=2, color='black')),
            text=[p[3] for p in special_points],
            textposition='top center',
            textfont=dict(size=9),
            hovertemplate='%{text}<extra></extra>'
        )

        triangle_x = [0, 1, 0, 0]
        triangle_y = [0, 0, 1, 0]
        triangle_z = [0, 0, 0, 0]
        edges = go.Scatter3d(
            x=triangle_x, y=triangle_y, z=triangle_z,
            mode='lines',
            line=dict(color='black', width=5),
            hoverinfo='skip'
        )

        layout = go.Layout(
            title=dict(text='3D Entropy Surface for 3-Outcome Distributions<br><sup>Rotate: click & drag | Zoom: scroll</sup>', font=dict(size=16)),
            scene=dict(
                xaxis=dict(title='Probability p₁', range=[0, 1], backgroundcolor='rgb(240,240,240)'),
                yaxis=dict(title='Probability p₂', range=[0, 1], backgroundcolor='rgb(240,240,240)'),
                zaxis=dict(title='Entropy (bits)', range=[0, 1.7], backgroundcolor='rgb(250,250,250)'),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            ),
            margin=dict(l=0, r=0, b=0, t=60),
            annotations=[dict(
                 text='Constraint: p₁ + p₂ + p₃ = 1<br>Peak at (1/3,1/3,1/3) = Maximum entropy | Corners = Zero entropy',
                showarrow=False,
                x=0.05, y=0.95, xref='paper', yref='paper',
                bgcolor='yellow', opacity=0.7,
                borderpad=4
            )]
        )

        fig = go.Figure(data=[surface, markers, edges], layout=layout)
        return fig
    
    @staticmethod
    def draw_node(ax, x, y, text, color='lightblue', radius=0.4):
        """Draw a circular node."""
        circle = plt.Circle((x, y), radius, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')

    @staticmethod
    def draw_edge(ax, x1, y1, x2, y2, label='', color='black'):
        """Draw an edge between nodes."""
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.2, label, ha='center', fontsize=10, 
                    weight='bold', color=color)

    @staticmethod
    def draw_uniform_tree():
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Tree 2: Uniform Codebook Q (A=25%, B=25%, C=25%, D=25%)\n' +
                    'Used to encode P: Average = 2.0 bits = H(P,Q)', 
                    fontsize=14, weight='bold', pad=20, color='darkred')
        
        # Root
        Chapter2.draw_node(ax, 5, 8, 'Root', 'lightgray')
        
        # Level 1: Split A,B vs C,D
        Chapter2.draw_edge(ax, 4.7, 7.7, 3, 5.3, '0', 'green')
        Chapter2.draw_node(ax, 3, 5, '', 'lightgray', 0.35)
        
        Chapter2.draw_edge(ax, 5.3, 7.7, 7, 5.3, '1', 'red')
        Chapter2.draw_node(ax, 7, 5, '', 'lightgray', 0.35)
        
        # Level 2 left: A vs B
        Chapter2.draw_edge(ax, 2.7, 4.7, 2, 2.8, '0', 'green')
        Chapter2.draw_node(ax, 2, 2.5, 'A\n60%', '#FF6B6B', 0.45)
        ax.text(2, 1.3, 'Code: 00\n2 bits', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        Chapter2.draw_edge(ax, 3.3, 4.7, 4, 2.8, '1', 'red')
        Chapter2.draw_node(ax, 4, 2.5, 'B\n20%', '#4ECDC4', 0.45)
        ax.text(4, 1.3, 'Code: 01\n2 bits', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        # Level 2 right: C vs D
        Chapter2.draw_edge(ax, 6.7, 4.7, 6, 2.8, '0', 'green')
        Chapter2.draw_node(ax, 6, 2.5, 'C\n10%', '#45B7D1', 0.45)
        ax.text(6, 1.3, 'Code: 10\n2 bits', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        Chapter2.draw_edge(ax, 7.3, 4.7, 8, 2.8, '1', 'red')
        Chapter2.draw_node(ax, 8, 2.5, 'D\n10%', '#FFA07A', 0.45)
        ax.text(8, 1.3, 'Code: 11\n2 bits', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        # Summary with waste highlighted
        summary = ('Encoding AAAAAABBCD with uniform codes:\n' +
                '6×(2 bits) + 2×(2 bits) + 1×(2 bits) + 1×(2 bits) = 20 bits\n' +
                'Average: 20/10 = 2.0 bits per character\n' +
                '━━━━━━━━━━━━━━━━━━━━\n' +
                'WASTED: 2.0 - 1.6 = 0.4 bits per character!')
        ax.text(5, -1.5, summary, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='#FFB6C6', alpha=0.7),
            family='monospace', weight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def draw_optimal_tree():
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Tree 1: Optimal for P (A=60%, B=20%, C=10%, D=10%)\n' +
                    'Average: 1.6 bits = H(P)', fontsize=14, weight='bold', pad=20)
        
        # Root
        Chapter2.draw_node(ax, 5, 8, 'Root', 'lightgray')
        
        # Level 1: A (left) vs others (right)
        Chapter2.draw_edge(ax, 4.7, 7.7, 2.5, 5.3, '0', 'green')
        Chapter2.draw_node(ax, 2.5, 5, 'A\n60%', '#FF6B6B', 0.5)
        ax.text(2.5, 3.8, 'Code: 0\n1 bit', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        Chapter2.draw_edge(ax, 5.3, 7.7, 7.5, 5.3, '1', 'red')
        Chapter2.draw_node(ax, 7.5, 5, '', 'lightgray', 0.35)
        
        # Level 2: B vs (C,D)
        Chapter2.draw_edge(ax, 7.2, 4.7, 6, 2.8, '0', 'green')
        Chapter2.draw_node(ax, 6, 2.5, 'B\n20%', '#4ECDC4', 0.45)
        ax.text(6, 1.3, 'Code: 10\n2 bits', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        Chapter2.draw_edge(ax, 7.8, 4.7, 9, 2.8, '1', 'red')
        Chapter2.draw_node(ax, 9, 2.5, '', 'lightgray', 0.3)
        
        # Level 3: C vs D
        Chapter2.draw_edge(ax, 8.7, 2.2, 8.2, 0.8, '0', 'green')
        Chapter2.draw_node(ax, 8.2, 0.5, 'C\n10%', '#45B7D1', 0.35)
        ax.text(8.2, -0.5, '110\n3 bits', ha='center', fontsize=7)
        
        Chapter2.draw_edge(ax, 9.3, 2.2, 9.8, 0.8, '1', 'red')
        Chapter2.draw_node(ax, 9.8, 0.5, 'D\n10%', '#FFA07A', 0.35)
        ax.text(9.8, -0.5, '111\n3 bits', ha='center', fontsize=7)
        
        # Summary
        summary = ('Encoding AAAAAABBCD:\n' +
                '6×(1 bit) + 2×(2 bits) + 1×(3 bits) + 1×(3 bits) = 16 bits\n' +
                'Average: 16/10 = 1.6 bits per character')
        ax.text(5, -1.5, summary, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            family='monospace')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def draw_wrong_biased_tree():
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Tree 3: Wrong-Biased Codebook Q (A=25%, B=12.5%, C=50%, D=12.5%)\n' +
                    'Used to encode P: Average = 2.2 bits = H(P,Q)',
                    fontsize=14, weight='bold', pad=20, color='darkred')
        
        # Root
        Chapter2.draw_node(ax, 5, 8, 'Root', 'lightgray')
        
        # Level 1: C (left) vs others (right) - WRONG PRIORITY!
        Chapter2.draw_edge(ax, 4.7, 7.7, 2.5, 5.3, '0', 'green')
        Chapter2.draw_node(ax, 2.5, 5, 'C\n10%', '#45B7D1', 0.5)
        ax.text(2.5, 6, 'Wrong!\nShould be A', ha='center', fontsize=8,
            color='red', weight='bold', style='italic')
        ax.text(2.5, 3.8, 'Code: 0\n1 bit', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        Chapter2.draw_edge(ax, 5.3, 7.7, 7.5, 5.3, '1', 'red')
        Chapter2.draw_node(ax, 7.5, 5, '', 'lightgray', 0.35)
        
        # Level 2: A vs (B,D)
        Chapter2.draw_edge(ax, 7.2, 4.7, 6, 2.8, '0', 'green')
        Chapter2.draw_node(ax, 6, 2.5, 'A\n60%', '#FF6B6B', 0.45)
        ax.text(6, 3.5, 'Most frequent\nbut 2 bits!', ha='center', fontsize=7,
            color='red', weight='bold', style='italic')
        ax.text(6, 1.3, 'Code: 10\n2 bits', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

        Chapter2.draw_edge(ax, 7.8, 4.7, 9, 2.8, '1', 'red')
        Chapter2.draw_node(ax, 9, 2.5, '', 'lightgray', 0.3)

        # Level 3: B vs D
        Chapter2.draw_edge(ax, 8.7, 2.2, 8.2, 0.8, '0', 'green')
        Chapter2.draw_node(ax, 8.2, 0.5, 'B\n20%', '#4ECDC4', 0.35)
        ax.text(8.2, -0.5, '110\n3 bits', ha='center', fontsize=7)

        Chapter2.draw_edge(ax, 9.3, 2.2, 9.8, 0.8, '1', 'red')
        Chapter2.draw_node(ax, 9.8, 0.5, 'D\n10%', '#FFA07A', 0.35)
        ax.text(9.8, -0.5, '111\n3 bits', ha='center', fontsize=7)

        # Summary with even more waste
        summary = ('Encoding AAAAAABBCD with wrong-biased codes:\n' +
                '6×(2 bits) + 2×(3 bits) + 1×(1 bit) + 1×(3 bits) = 22 bits\n' +
                'Average: 22/10 = 2.2 bits per character\n' +
                '━━━━━━━━━━━━━━━━━━━━\n' +
                'WASTED: 2.2 - 1.6 = 0.6 bits per character!\n' +
                '(Worse than uniform!)')
        ax.text(5, -1.8, summary, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='#FF8888', alpha=0.7),
            family='monospace', weight='bold')
        
        plt.tight_layout()
        return fig
    
    def draw_worst_case_tree():
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Tree 4: Reversed-Priority Q (A=12.5%, B=12.5%, C=25%, D=50%)\n' +
                    'Used to encode P: Average = 2.7 bits = H(P,Q)',
                    fontsize=14, weight='bold', pad=20, color='darkred')
        
        # Root
        Chapter2.draw_node(ax, 5, 8, 'Root', 'lightgray')
        
        # Level 1: D (left) vs others (right) - COMPLETELY BACKWARDS!
        Chapter2.draw_edge(ax, 4.7, 7.7, 2.5, 5.3, '0', 'green')
        Chapter2.draw_node(ax, 2.5, 5, 'D\n10%', '#FFA07A', 0.5)
        ax.text(2.5, 6.2, 'DISASTER!\nLeast frequent\ngets 1 bit!', ha='center', fontsize=7, 
            color='red', weight='bold', style='italic')
        ax.text(2.5, 3.8, 'Code: 0\n1 bit', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        Chapter2.draw_edge(ax, 5.3, 7.7, 7.5, 5.3, '1', 'red')
        Chapter2.draw_node(ax, 7.5, 5, '', 'lightgray', 0.35)
        
        # Level 2: C vs (A,B)
        Chapter2.draw_edge(ax, 7.2, 4.7, 6, 2.8, '0', 'green')
        Chapter2.draw_node(ax, 6, 2.5, 'C\n10%', '#45B7D1', 0.45)
        ax.text(6, 3.5, '10% but 2 bits', ha='center', fontsize=7, 
            color='red', weight='bold', style='italic')
        ax.text(6, 1.3, 'Code: 10\n2 bits', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        Chapter2.draw_edge(ax, 7.8, 4.7, 9, 2.8, '1', 'red')
        Chapter2.draw_node(ax, 9, 2.5, '', 'lightgray', 0.3)
        
        # Level 3: A vs B - BOTH MOST FREQUENT GET LONGEST CODES!
        Chapter2.draw_edge(ax, 8.7, 2.2, 8.2, 0.8, '0', 'green')
        Chapter2.draw_node(ax, 8.2, 0.5, 'A\n60%', '#FF6B6B', 0.35)
        ax.text(9.0, 0.5, '\n60% but\n3 bits!', ha='center', fontsize=6, 
            color='red', weight='bold', style='italic')
        ax.text(8.2, -0.8, '110\n3 bits', ha='center', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='#FFE6E6'))
        
        Chapter2.draw_edge(ax, 9.3, 2.2, 9.8, 0.8, '1', 'red')
        Chapter2.draw_node(ax, 9.8, 0.5, 'B\n20%', '#4ECDC4', 0.35)
        ax.text(9.8, -0.8, '111\n3 bits', ha='center', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='#FFE6E6'))
        
        # Summary with reversed priorities
        summary = ('Encoding AAAAAABBCD with reversed-priority codes:\n' +
                '6×(3 bits) + 2×(3 bits) + 1×(2 bits) + 1×(1 bit) = 27 bits\n' +
                'Average: 27/10 = 2.7 bits per character\n' +
                '━━━━━━━━━━━━━━━━━━━━\n' +
                'WASTED: 2.7 - 1.6 = 1.1 bits per character!!!\n' +
                'That\'s 69% MORE bits than optimal!')
        ax.text(5, -2.2, summary, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='#FF5555', alpha=0.8),
            family='monospace', weight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def draw_all_four_comparison():
        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        
        titles = [
            'Optimal\nH(P) = 1.6 bits',
            'Uniform\nH(P,Q_unif) = 2.0\n+0.4 wasted',
            'Wrong-Biased\nH(P,Q_bias) = 2.2\n+0.6 wasted',
            'Reversed-Priority\nH(P,Q_rev) = 2.7\n+1.1 wasted!'
        ]
        colors = ['lightgreen', '#FFB6C6', '#FF8888', '#FF5555']
        
        for i, (ax, title, color) in enumerate(zip(axes, titles, colors)):
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            ax.set_title(title, fontsize=11, weight='bold', pad=10,
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
            
            # Simplified representation - just show the codes
            codes = [
                [('A', '0', 1), ('B', '10', 2), ('C', '110', 3), ('D', '111', 3)],
                [('A', '00', 2), ('B', '01', 2), ('C', '10', 2), ('D', '11', 2)],
                [('C', '0', 1), ('A', '10', 2), ('B', '110', 3), ('D', '111', 3)],
                [('D', '0', 1), ('C', '10', 2), ('A', '110', 3), ('B', '111', 3)]
            ]
            
            y_pos = 8
            for char, code, bits in codes[i]:
                # Highlight A in red if it's not getting 1 bit
                char_color = '#FF6B6B' if char == 'A' else 'black'
                if char == 'A' and bits != 1:
                    char_color = 'red'
                    ax.text(1, y_pos, 'х', fontsize=16)
                elif char == 'A' and bits == 1:
                    ax.text(1, y_pos, '✓', fontsize=16, color='green')
                
                ax.text(2, y_pos, f'{char} (60%)' if char == 'A' else 
                                f'{char} (20%)' if char == 'B' else f'{char} (10%)',
                    fontsize=10, weight='bold', color=char_color)
                ax.text(5, y_pos, f'→  {code}', fontsize=10, family='monospace')
                ax.text(6.8, y_pos, f'({bits} bit{"s" if bits > 1 else ""})', 
                    fontsize=9, style='italic')
                y_pos -= 1.5
            
            # Show calculation
            calcs = [
                '0.6×1 + 0.2×2\n+0.1×3 + 0.1×3\n= 1.6 bits',
                '0.6×2 + 0.2×2\n+0.1×2 + 0.1×2\n= 2.0 bits',
                '0.6×2 + 0.2×3\n+0.1×1 + 0.1×3\n= 2.2 bits',
                '0.6×3 + 0.2×3\n+0.1×2 + 0.1×1\n= 2.7 bits'
            ]
            ax.text(5, 0.8, calcs[i], ha='center', fontsize=9, 
                family='monospace', weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.suptitle('Cross-Entropy Spectrum: From Optimal to Reversed-Priority',
                    fontsize=16, weight='bold', y=0.98)
        plt.tight_layout()
        return fig


    @staticmethod
    def entropy(P_dist):
        """Calculate Shannon entropy H(P)."""
        return -sum(p * math.log2(p) for p in P_dist.values() if p > 0)


    @staticmethod
    def interactive_cross_entropy():
        import math
        from itertools import permutations

        def code_lengths_to_distribution(code_dict, symbols):
            """Derive implied distribution Q from code lengths.

            For Huffman codes, code length L implies Q(x) ∝ 2^(-L).
            We normalize to get a proper distribution.
            """
            unnormalized = {s: 2**(-code_dict[s]) for s in symbols}
            total = sum(unnormalized.values())
            return {s: unnormalized[s] / total for s in symbols}

        def actual_cross_entropy(P, Q, symbols):
            """Calculate actual cross-entropy H(P,Q) = -Σ P(x) log₂ Q(x)"""
            return -sum(P[s] * math.log2(Q[s]) for s in symbols)

        # True distribution
        P = {'A': 0.6, 'B': 0.2, 'C': 0.1, 'D': 0.1}
        symbols = ['A', 'B', 'C', 'D']
        probs = [P[s] for s in symbols]

        # Valid code length patterns for 4 symbols (satisfying Kraft inequality)
        # Pattern 1: (2,2,2,2) - uniform tree
        # Pattern 2: (1,2,3,3) - unbalanced tree
        valid_patterns = [
            (2, 2, 2, 2),
            (1, 2, 3, 3)
        ]

        # Generate all possible code assignments
        all_assignments = []
        for pattern in valid_patterns:
            # For each pattern, generate all permutations (which symbol gets which length)
            for perm in permutations(pattern):
                # Code length assignment
                code_dict = {symbols[i]: perm[i] for i in range(4)}

                # Average code length (Huffman practical value)
                avg_code_length = sum(probs[i] * perm[i] for i in range(4))

                # Derive implied distribution Q and calculate actual cross-entropy
                Q = code_lengths_to_distribution(code_dict, symbols)
                actual_ce = actual_cross_entropy(P, Q, symbols)

                # Create a readable description
                desc = ', '.join([f'{s}:{code_dict[s]}' for s in symbols])
                Q_desc = ', '.join([f'{s}:{Q[s]:.3f}' for s in symbols])

                all_assignments.append({
                    'assignment': code_dict,
                    'description': desc,
                    'avg_code_length': avg_code_length,
                    'actual_cross_entropy': actual_ce,
                    'Q_distribution': Q,
                    'Q_description': Q_desc,
                    'pattern': str(pattern)
                })

        # First, ensure our four special cases are included
        special_cases = [
            {'A': 1, 'B': 2, 'C': 3, 'D': 3},  # Optimal
            {'A': 2, 'B': 2, 'C': 2, 'D': 2},  # Uniform
            {'A': 2, 'B': 3, 'C': 1, 'D': 3},  # Wrong-biased
            {'A': 3, 'B': 3, 'C': 2, 'D': 1},  # Reversed
        ]

        # Add special cases first
        unique_assignments = []
        for special in special_cases:
            avg_code_length = sum(probs[i] * special[symbols[i]] for i in range(4))
            desc = ', '.join([f'{s}:{special[s]}' for s in symbols])
            Q = code_lengths_to_distribution(special, symbols)
            actual_ce = actual_cross_entropy(P, Q, symbols)
            Q_desc = ', '.join([f'{s}:{Q[s]:.3f}' for s in symbols])

            unique_assignments.append({
                'assignment': special,
                'description': desc,
                'avg_code_length': avg_code_length,
                'actual_cross_entropy': actual_ce,
                'Q_distribution': Q,
                'Q_description': Q_desc,
                'pattern': '(1, 2, 3, 3)' if avg_code_length != 2.0 else '(2, 2, 2, 2)'
            })

        # Then add other unique assignments (by average code length)
        special_ce_set = {round(a['avg_code_length'], 4) for a in unique_assignments}
        for a in all_assignments:
            ce_rounded = round(a['avg_code_length'], 4)
            if ce_rounded not in special_ce_set:
                # Check if we already have this code length
                if not any(round(ua['avg_code_length'], 4) == ce_rounded for ua in unique_assignments):
                    unique_assignments.append(a)

        # Sort by average code length
        unique_assignments.sort(key=lambda x: x['avg_code_length'])

        # Extract data for plotting (use average code length for x-axis)
        avg_code_lengths = [a['avg_code_length'] for a in unique_assignments]
        actual_ces = [a['actual_cross_entropy'] for a in unique_assignments]
        descriptions = [a['description'] for a in unique_assignments]
        Q_descriptions = [a['Q_description'] for a in unique_assignments]
        patterns = [a['pattern'] for a in unique_assignments]

        # Create colors based on our 4 examples
        # Check actual assignments, not just cross-entropy values
        colors = []
        special_labels = []
        for a in unique_assignments:
            assignment = a['assignment']

            # Optimal: A=1, B=2, C=3, D=3
            if assignment == {'A': 1, 'B': 2, 'C': 3, 'D': 3}:
                colors.append('#90EE90')  # Light green - optimal
                special_labels.append('⭐ OPTIMAL')
            # Uniform: all 2
            elif assignment == {'A': 2, 'B': 2, 'C': 2, 'D': 2}:
                colors.append('#FFB6C6')  # Pink - uniform
                special_labels.append('🔷 UNIFORM')
            # Wrong-biased: C=1, A=2, B=3, D=3
            elif assignment == {'A': 2, 'B': 3, 'C': 1, 'D': 3}:
                colors.append('#FF8888')  # Light red - wrong-biased
                special_labels.append('⚠️ WRONG-BIASED')
            # Reversed: D=1, C=2, A=3, B=3 (complete opposite of optimal)
            elif assignment == {'A': 3, 'B': 3, 'C': 2, 'D': 1}:
                colors.append('#FF5555')  # Red - reversed
                special_labels.append('🔄 REVERSED')
            else:
                colors.append('#B0C4DE')  # Light steel blue - others
                special_labels.append('')

        # Create interactive plot
        fig = go.Figure()

        # Add scatter plot for all assignments
        fig.add_trace(go.Scatter(
            x=avg_code_lengths,
            y=[0] * len(avg_code_lengths),  # All at same y-level
            mode='markers',
            marker=dict(
                size=16,
                color=colors,
                line=dict(width=2, color='black'),
                symbol='circle'
            ),
            text=descriptions,
            customdata=[[avg_len, actual_ce, desc, Q_desc, label, pattern]
                       for avg_len, actual_ce, desc, Q_desc, label, pattern in
                       zip(avg_code_lengths, actual_ces, descriptions, Q_descriptions, special_labels, patterns)],
            hovertemplate='<b>%{customdata[4]}</b><br>' +
                        '<b>Average Code Length:</b> %{customdata[0]:.2f} bits<br>' +
                        '<b>Actual H(P,Q):</b> %{customdata[1]:.3f} bits<br>' +
                        '─────────────────────<br>' +
                        '<b>Code Lengths:</b> %{customdata[2]}<br>' +
                        '<b>Implied Q:</b> {%{customdata[3]}}<br>' +
                        '<i>(Q derived from code lengths via Q(x) ∝ 2^(-length))</i><br>' +
                        '<extra></extra>',
            showlegend=False
        ))

        # Add H(P) baseline
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, opacity=0.3)
        
        H_P = Chapter2.entropy(P)
        # Add vertical line at H(P)
        fig.add_vline(x=H_P, line_dash="dash", line_color="blue", line_width=3,
                    annotation_text=f"H(P) = {H_P:.2f} bits<br>(theoretical minimum)",
                    annotation_position="top")

        # Add annotations for special points
        annotations_data = [
            (1.6, "⭐ OPTIMAL", 0.08),
            (2.0, "🔷 UNIFORM", 0.06),
            (2.2, "⚠️ WRONG", 0.04),
            (2.7, "🔄 REVERSED", 0.02)
        ]

        for x_pos, label, y_pos in annotations_data:
            fig.add_annotation(
                x=x_pos, y=y_pos,
                text=label,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=0, ay=-40,
                font=dict(size=11, color='black'),
                bgcolor='white',
                bordercolor='black',
                borderwidth=2
            )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Complete Coding Spectrum: All Possible Code Assignments<br>' +
                        '<sub>For P = {A:60%, B:20%, C:10%, D:10%} with 4-symbol binary codes<br>' +
                        'Hover to see both Average Code Length (Huffman practical) and Actual H(P,Q) (theoretical)</sub>',
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title='Average Code Length (bits per symbol)',
            yaxis_title='',
            xaxis=dict(
                range=[1.5, 2.8],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showticklabels=False,
                range=[-0.05, 0.15]
            ),
            height=400,
            hovermode='closest',
            plot_bgcolor='white'
        )

        # Add range annotation
        fig.add_annotation(
            x=2.15, y=0.12,
            text=f'Total Range: [{H_P:.2f}, 2.70] bits<br>' +
                f'Span: {2.7 - H_P:.2f} bits ({((2.7/H_P - 1)*100):.0f}% variation)',
            showarrow=False,
            bgcolor='lightyellow',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        )
        return fig
    
    @staticmethod
    def plot_cross_entropy_spectrum(avg_optimal, avg_uniform, avg_biased, avg_worst, H_P):
        fig, ax = plt.subplots(figsize=(14, 8))
        scenarios = ['Optimal\n(Q=P)', 'Uniform\nCodebook', 'Wrong-Biased\nCodebook', 'Reversed-Priority\n(Q_reversed)']
        avg_bits_list = [avg_optimal, avg_uniform, avg_biased, avg_worst]
        colors = ['lightgreen', '#FFB6C6', '#FF8888', '#FF5555']

        bars = ax.bar(scenarios, avg_bits_list, color=colors, edgecolor='black', linewidth=2)

        # Add entropy baseline
        ax.axhline(H_P, color='blue', linestyle='--', linewidth=3, 
                label=f'H(P) = {H_P:.4f} bits (theoretical minimum)')

        # Add values on bars
        for bar, bits in zip(bars, avg_bits_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.08,
                f'{bits:.2f} bits', ha='center', va='bottom', 
                fontsize=13, weight='bold')
            
            # Show waste
            waste = bits - H_P
            percent = ((bits/H_P - 1) * 100)
            if waste > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'+{waste:.2f}\n({percent:.0f}%)', ha='center', va='center',
                    fontsize=11, weight='bold', color='darkred',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

        ax.set_ylabel('Average Bits per Character', fontsize=14, weight='bold')
        ax.set_title('Cross-Entropy Spectrum: From Optimal to Reversed-Priority',
                    fontsize=16, weight='bold', pad=20)
        ax.set_ylim(0, 3.0)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=12, loc='upper left')

        # Add interpretation
        interpretation = (
            'When encoding data with distribution P:\n'
            f'• BEST case (Q=P): H(P) = {H_P:.2f} bits\n'
            f'• Reversed-priority (Q=reversed): H(P,Q_rev) = {avg_worst:.2f} bits\n'
            f'• Range with code lengths (1,2,3,3): [{H_P:.2f}, {avg_worst:.2f}] bits\n'
            f'• KL(P||Q) = H(P,Q) - H(P) = wasted bits due to mismatch'
        )
        ax.text(0.5, -0.28, interpretation, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            transform=ax.transAxes)
        