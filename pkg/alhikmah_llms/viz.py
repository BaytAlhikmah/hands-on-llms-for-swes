"""Visualization helpers for the course notebooks."""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


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
