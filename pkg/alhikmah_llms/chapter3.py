"""Visualization helpers for Chapter 3: Cross-Entropy and KL Divergence."""

import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class Chapter3:
    """Visualizations for Chapter3 3: Cross-Entropy, KL Divergence, and Neural Networks."""

    @staticmethod
    def plot_cross_entropy_intuition(
        true_probs: list[float] | None = None,
        predicted_bad: list[float] | None = None,
        predicted_good: list[float] | None = None,
        labels: list[str] | None = None,
        figsize: tuple[int, int] = (15, 4)
    ) -> plt.Figure:
        """Side-by-side: true vs predicted distribution.

        Shows how cross-entropy measures the gap between true and predicted distributions.

        Args:
            true_probs: True probability distribution. Defaults to [0.7, 0.2, 0.1].
            predicted_bad: Bad model's predictions. Defaults to uniform [0.33, 0.33, 0.34].
            predicted_good: Good model's predictions. Defaults to [0.65, 0.25, 0.1].
            labels: Category labels. Defaults to ['u', 'a', 'e'].
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        if true_probs is None:
            true_probs = [0.7, 0.2, 0.1]
        if predicted_bad is None:
            predicted_bad = [0.33, 0.33, 0.34]
        if predicted_good is None:
            predicted_good = [0.65, 0.25, 0.1]
        if labels is None:
            labels = ['u', 'a', 'e']

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: True distribution
        axes[0].bar(labels, true_probs, color='green', alpha=0.7, edgecolor='black', linewidth=2)
        axes[0].set_title('True Distribution P\n(actual data)', fontsize=12, weight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].set_ylabel('Probability', fontsize=11, weight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Add probability labels
        for i, (label, p) in enumerate(zip(labels, true_probs)):
            axes[0].text(i, p + 0.03, f'{p:.2f}', ha='center', fontsize=10, weight='bold')

        # Plot 2: Bad prediction
        axes[1].bar(labels, predicted_bad, color='red', alpha=0.7, edgecolor='black', linewidth=2)
        h_bad = -sum(p * np.log(q) if q > 0 else 0 for p, q in zip(true_probs, predicted_bad))
        axes[1].set_title(f'Bad Model Q\nH(P,Q) = {h_bad:.3f} nats', fontsize=12, weight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].set_ylabel('Probability', fontsize=11, weight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        for i, (label, p) in enumerate(zip(labels, predicted_bad)):
            axes[1].text(i, p + 0.03, f'{p:.2f}', ha='center', fontsize=10)

        # Plot 3: Good prediction
        axes[2].bar(labels, predicted_good, color='blue', alpha=0.7, edgecolor='black', linewidth=2)
        h_good = -sum(p * np.log(q) if q > 0 else 0 for p, q in zip(true_probs, predicted_good))
        axes[2].set_title(f'Good Model Q\nH(P,Q) = {h_good:.3f} nats', fontsize=12, weight='bold')
        axes[2].set_ylim([0, 1])
        axes[2].set_ylabel('Probability', fontsize=11, weight='bold')
        axes[2].grid(axis='y', alpha=0.3)

        for i, (label, p) in enumerate(zip(labels, predicted_good)):
            axes[2].text(i, p + 0.03, f'{p:.2f}', ha='center', fontsize=10)

        plt.suptitle('Cross-Entropy: Comparing True and Predicted Distributions',
                     fontsize=14, weight='bold', y=1.02)

        # Add insight box
        fig.text(0.5, -0.05, 'Key Insight: Lower cross-entropy = predicted distribution closer to true distribution',
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_surprise_breakdown(
        true_probs: list[float] | None = None,
        predicted: list[float] | None = None,
        labels: list[str] | None = None,
        figsize: tuple[int, int] = (12, 4)
    ) -> plt.Figure:
        """Show which outcomes contribute most to cross-entropy.

        Args:
            true_probs: True probability distribution.
            predicted: Model's predictions.
            labels: Category labels.
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        if true_probs is None:
            true_probs = [0.7, 0.2, 0.05, 0.05]
        if predicted is None:
            predicted = [0.5, 0.3, 0.1, 0.1]
        if labels is None:
            labels = ['u', 'a', 'e', 'i']

        # Surprise for each outcome: -P(x) * log Q(x)
        surprises = [-p * np.log(q) if q > 0 else 0 for p, q in zip(true_probs, predicted)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar chart of contributions
        bars = ax1.bar(labels, surprises, color='orange', alpha=0.8,
                      edgecolor='black', linewidth=2)
        ax1.set_title('Cross-Entropy Contribution by Outcome', fontsize=12, weight='bold')
        ax1.set_ylabel('-P(x) log Q(x)', fontsize=11, weight='bold')
        ax1.set_xlabel('Outcome', fontsize=11, weight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Annotate with probabilities
        for i, (bar, p, q, s) in enumerate(zip(bars, true_probs, predicted, surprises)):
            ax1.text(i, s + max(surprises) * 0.02,
                    f'P={p:.2f}\nQ={q:.2f}',
                    ha='center', fontsize=9)

        # Cumulative contribution
        total_ce = sum(surprises)
        n_outcomes = len(labels)
        random_ce = np.log(n_outcomes)

        ax2.bar(['Model\nCross-Entropy'], [total_ce], color='darkred',
               alpha=0.8, edgecolor='black', linewidth=2)
        ax2.axhline(y=random_ce, color='gray', linestyle='--', linewidth=2,
                   label=f'Random guess = ln({n_outcomes}) = {random_ce:.3f}')
        ax2.set_title(f'Total Cross-Entropy', fontsize=12, weight='bold')
        ax2.set_ylabel('Nats (natural log units)', fontsize=11, weight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)

        # Add value on bar
        ax2.text(0, total_ce + random_ce * 0.02, f'{total_ce:.3f}',
                ha='center', fontsize=12, weight='bold')

        plt.suptitle('Where Does the Cross-Entropy Come From?',
                     fontsize=14, weight='bold')

        # Key insight
        fig.text(0.5, -0.05,
                '🎯 Key Observation: High-probability outcomes (where P is large) contribute most!',
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_entropy_decomposition(
        true_probs: list[float] | None = None,
        predicted: list[float] | None = None,
        figsize: tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """Visualize H(P,Q) = H(P) + KL(P||Q).

        Args:
            true_probs: True probability distribution.
            predicted: Model's predictions.
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        if true_probs is None:
            true_probs = [0.7, 0.2, 0.1]
        if predicted is None:
            predicted = [0.5, 0.3, 0.2]

        # Calculate components
        H_P = -sum(p * np.log(p) for p in true_probs if p > 0)
        H_P_Q = -sum(p * np.log(q) for p, q in zip(true_probs, predicted) if q > 0)
        KL = H_P_Q - H_P

        fig, ax = plt.subplots(figsize=figsize)

        # Stacked bar
        bar1 = ax.bar(['Cross-Entropy\nH(P,Q)'], [H_P],
                     label=f'Entropy H(P) = {H_P:.4f}\n(irreducible - property of data)',
                     color='lightblue', edgecolor='black', linewidth=2)
        bar2 = ax.bar(['Cross-Entropy\nH(P,Q)'], [KL], bottom=[H_P],
                     label=f'KL Divergence = {KL:.4f}\n(what we minimize in training)',
                     color='orange', edgecolor='black', linewidth=2, alpha=0.8)

        ax.axhline(y=H_P_Q, color='red', linestyle='--', linewidth=2,
                  label=f'Total = {H_P_Q:.4f}')
        ax.set_ylabel('Nats (natural log units)', fontsize=12, weight='bold')
        ax.set_title('Cross-Entropy Decomposition\nH(P,Q) = H(P) + KL(P||Q)',
                    fontsize=13, weight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, H_P_Q * 1.2)

        # Annotations on bars
        ax.text(0, H_P/2, "Can't reduce\n(property of data)",
               ha='center', va='center', fontsize=11, color='darkblue', weight='bold')
        ax.text(0, H_P + KL/2, "Optimize this\nduring training!",
               ha='center', va='center', fontsize=11, color='darkred', weight='bold')

        # Key insight box
        fig.text(0.5, -0.08,
                '🎯 Key Insight: Entropy H(P) is constant (property of true data).\n'
                'Training only minimizes the KL divergence term!',
                ha='center', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_training_trajectory(
        true_probs: list[float] | None = None,
        steps: list[list[float]] | None = None,
        labels: list[str] | None = None,
        figsize: tuple[int, int] = (18, 3)
    ) -> plt.Figure:
        """Show Q moving toward P during training.

        Args:
            true_probs: True probability distribution.
            steps: List of predicted distributions at different training steps.
            labels: Category labels.
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        if true_probs is None:
            true_probs = [0.7, 0.2, 0.1]
        if steps is None:
            steps = [
                [0.33, 0.33, 0.34],  # step 0: random
                [0.45, 0.35, 0.20],  # step 50
                [0.55, 0.28, 0.17],  # step 100
                [0.62, 0.24, 0.14],  # step 150
                [0.68, 0.21, 0.11],  # step 200: converged
            ]
        if labels is None:
            labels = ['u', 'a', 'e']

        fig, axes = plt.subplots(1, len(steps), figsize=figsize)

        for i, (ax, pred) in enumerate(zip(axes, steps)):
            # Overlay true (green) and predicted (blue) distributions
            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width/2, true_probs, width, label='True P',
                  color='green', alpha=0.6, edgecolor='black', linewidth=1.5)
            ax.bar(x + width/2, pred, width, label='Model Q',
                  color='blue', alpha=0.6, edgecolor='black', linewidth=1.5)

            # Calculate and show loss
            ce = -sum(p * np.log(q) if q > 0 else 0 for p, q in zip(true_probs, pred))
            ax.set_title(f'Step {i*50}\nLoss = {ce:.3f}', fontsize=11, weight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylim([0, 0.8])
            ax.grid(axis='y', alpha=0.3)

            if i == 0:
                ax.legend(fontsize=9, loc='upper right')
                ax.set_ylabel('Probability', fontsize=10, weight='bold')

        plt.suptitle('Model Learning: Q converges to P → Cross-Entropy decreases',
                     fontsize=14, weight='bold', y=1.05)

        # Key insight
        fig.text(0.5, -0.08,
                '🎯 Training = moving predicted distribution closer to true distribution!',
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_kl_asymmetry(
        P: list[float] | None = None,
        Q: list[float] | None = None,
        labels: list[str] | None = None,
        figsize: tuple[int, int] = (15, 4)
    ) -> plt.Figure:
        """Show KL(P||Q) ≠ KL(Q||P).

        Args:
            P: First distribution (peaked).
            Q: Second distribution (flat).
            labels: Category labels.
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        if P is None:
            P = [0.8, 0.15, 0.05]  # peaked
        if Q is None:
            Q = [0.4, 0.3, 0.3]     # flat
        if labels is None:
            labels = ['A', 'B', 'C']

        KL_PQ = sum(p * np.log(p/q) if p > 0 and q > 0 else 0 for p, q in zip(P, Q))
        KL_QP = sum(q * np.log(q/p) if p > 0 and q > 0 else 0 for p, q in zip(P, Q))

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot distributions
        x = np.arange(len(labels))
        width = 0.35
        axes[0].bar(x - width/2, P, width, label='P (peaked)',
                   color='red', alpha=0.7, edgecolor='black', linewidth=2)
        axes[0].bar(x + width/2, Q, width, label='Q (flat)',
                   color='blue', alpha=0.7, edgecolor='black', linewidth=2)
        axes[0].set_title('Two Distributions', fontsize=12, weight='bold')
        axes[0].legend(fontsize=10)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, fontsize=11)
        axes[0].set_ylim(0, 0.9)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylabel('Probability', fontsize=11, weight='bold')

        # Add probability labels
        for i, (p_val, q_val) in enumerate(zip(P, Q)):
            axes[0].text(i - width/2, p_val + 0.03, f'{p_val:.2f}',
                        ha='center', fontsize=9, weight='bold')
            axes[0].text(i + width/2, q_val + 0.03, f'{q_val:.2f}',
                        ha='center', fontsize=9, weight='bold')

        # Forward KL: KL(P||Q)
        axes[1].bar(['KL(P||Q)'], [KL_PQ], color='purple', alpha=0.8,
                   edgecolor='black', linewidth=2)
        axes[1].set_title(f'Forward KL = {KL_PQ:.4f}\n(mode-seeking)',
                         fontsize=12, weight='bold')
        axes[1].set_ylabel('Divergence (nats)', fontsize=11, weight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].text(0, KL_PQ + KL_PQ * 0.05, f'{KL_PQ:.4f}',
                    ha='center', fontsize=12, weight='bold')

        # Reverse KL: KL(Q||P)
        axes[2].bar(['KL(Q||P)'], [KL_QP], color='orange', alpha=0.8,
                   edgecolor='black', linewidth=2)
        axes[2].set_title(f'Reverse KL = {KL_QP:.4f}\n(mode-covering)',
                         fontsize=12, weight='bold')
        axes[2].set_ylabel('Divergence (nats)', fontsize=11, weight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].text(0, KL_QP + KL_QP * 0.05, f'{KL_QP:.4f}',
                    ha='center', fontsize=12, weight='bold')

        plt.suptitle('KL Divergence is NOT Symmetric!',
                     fontsize=14, weight='bold', y=1.02)

        # Key insight
        fig.text(0.5, -0.08,
                '🎯 Key Insight: KL(P||Q) penalizes differently than KL(Q||P) - matters for model training!',
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_cross_entropy_surface(
        true_probs: list[float] | None = None,
        figsize: tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """2D distribution space → cross-entropy heatmap.

        For a 3-outcome distribution, shows how cross-entropy varies
        as the predicted distribution changes.

        Args:
            true_probs: True probability distribution [p1, p2, p3].
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        if true_probs is None:
            true_probs = [0.7, 0.2, 0.1]

        # Vary predicted Q = [q1, q2, 1-q1-q2]
        q1_range = np.linspace(0.05, 0.95, 80)
        q2_range = np.linspace(0.05, 0.95, 80)

        ce_grid = np.zeros((len(q2_range), len(q1_range)))

        for i, q2 in enumerate(q2_range):
            for j, q1 in enumerate(q1_range):
                q3 = 1 - q1 - q2
                if q3 > 0:  # valid probability
                    Q = [q1, q2, q3]
                    ce = -sum(p * np.log(q) if q > 0 else 10
                             for p, q in zip(true_probs, Q))
                    ce_grid[i, j] = ce
                else:
                    ce_grid[i, j] = np.nan

        fig, ax = plt.subplots(figsize=figsize)

        # Contour plot
        im = ax.contourf(q1_range, q2_range, ce_grid, levels=25, cmap='viridis_r')
        contours = ax.contour(q1_range, q2_range, ce_grid, levels=10,
                             colors='white', alpha=0.4, linewidths=1)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

        # Mark the true distribution (optimal point)
        ax.plot(true_probs[0], true_probs[1], 'r*', markersize=25,
               label=f'True P = [{true_probs[0]:.1f}, {true_probs[1]:.1f}, {true_probs[2]:.1f}]',
               markeredgecolor='white', markeredgewidth=2)

        ax.set_xlabel('Q[0] - predicted probability for outcome 1',
                     fontsize=11, weight='bold')
        ax.set_ylabel('Q[1] - predicted probability for outcome 2',
                     fontsize=11, weight='bold')
        ax.set_title('Cross-Entropy Loss Surface H(P,Q)\n(darker = lower loss)',
                    fontsize=13, weight='bold', pad=15)

        # Draw valid region boundary (triangle)
        triangle_x = [0, 1, 0, 0]
        triangle_y = [0, 0, 1, 0]
        ax.plot(triangle_x, triangle_y, 'k--', linewidth=2, alpha=0.5,
               label='Valid region (q1+q2≤1)')

        cbar = plt.colorbar(im, ax=ax, label='Cross-Entropy (nats)')
        cbar.set_label('Cross-Entropy (nats)', fontsize=11, weight='bold')

        ax.legend(fontsize=10, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Key insight
        fig.text(0.5, -0.05,
                '🎯 Loss surface has unique minimum at Q = P\n'
                '(but optimization in neural net parameter space is non-convex!)',
                ha='center', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_bigram_cross_entropy_example(
        true_counts: list[int] | None = None,
        predictions: dict[str, list[float]] | None = None,
        labels: list[str] | None = None,
        figsize: tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """Show cross-entropy for actual bigram predictions.

        Example: After 'q', what comes next?

        Args:
            true_counts: Observed counts for each next character.
            predictions: Dict mapping training stage to predicted distribution.
            labels: Character labels.
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        if true_counts is None:
            true_counts = [850, 5, 2, 1, 1]  # 'qu' dominates
        if labels is None:
            labels = ['u', 'a', 'e', 'i', 'other']
        if predictions is None:
            predictions = {
                'Random (step 0)': [0.2, 0.2, 0.2, 0.2, 0.2],
                'After 10 epochs': [0.6, 0.15, 0.1, 0.1, 0.05],
                'After 100 epochs': [0.85, 0.06, 0.04, 0.03, 0.02],
                'After 500 epochs': [0.99, 0.003, 0.002, 0.003, 0.002],
            }

        # Normalize true counts to probabilities
        true_probs = [c/sum(true_counts) for c in true_counts]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for ax, (name, pred) in zip(axes, predictions.items()):
            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width/2, true_probs, width, label='True',
                  color='green', alpha=0.6, edgecolor='black', linewidth=1.5)
            ax.bar(x + width/2, pred, width, label='Predicted',
                  color='blue', alpha=0.6, edgecolor='black', linewidth=1.5)

            ce = -sum(p * np.log(q) if q > 0 else 10
                     for p, q in zip(true_probs, pred))

            ax.set_title(f'{name}\nCross-Entropy = {ce:.4f}',
                        fontsize=11, weight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9)
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylabel('Probability', fontsize=10, weight='bold')

        plt.suptitle('Cross-Entropy During Training: Predicting "q" → ?',
                     fontsize=14, weight='bold', y=0.995)

        # Key insight
        fig.text(0.5, 0.02,
                '🎯 As the model learns, predicted distribution approaches true distribution,\n'
                'and cross-entropy decreases toward the entropy of the true distribution!',
                ha='center', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_loss_curve(
        losses: list[float] | None = None,
        val_losses: list[float] | None = None,
        figsize: tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """Plot training and validation loss curves.

        Args:
            losses: Training losses over epochs.
            val_losses: Validation losses over epochs (optional).
            figsize: Figure size tuple.

        Returns:
            Matplotlib figure.
        """
        if losses is None:
            # Generate example loss curve
            epochs = 100
            losses = [2.5 * np.exp(-0.03 * i) + 0.5 + np.random.normal(0, 0.02)
                     for i in range(epochs)]

        if val_losses is None and losses is not None:
            # Generate example validation curve (slightly higher, more variance)
            val_losses = [l + 0.1 + np.random.normal(0, 0.03) for l in losses]

        fig, ax = plt.subplots(figsize=figsize)

        epochs_range = range(len(losses))
        ax.plot(epochs_range, losses, linewidth=2, label='Training Loss',
               color='#2E86AB', alpha=0.8)

        if val_losses is not None:
            ax.plot(epochs_range, val_losses, linewidth=2, label='Validation Loss',
                   color='#FF6B6B', alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12, weight='bold')
        ax.set_ylabel('Cross-Entropy Loss (nats)', fontsize=12, weight='bold')
        ax.set_title('Training Progress: Loss Decreasing Over Time',
                    fontsize=13, weight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='upper right')

        # Mark final values
        if val_losses is not None:
            final_train = losses[-1]
            final_val = val_losses[-1]
            ax.axhline(y=final_train, color='#2E86AB', linestyle='--',
                      alpha=0.3, linewidth=1)
            ax.axhline(y=final_val, color='#FF6B6B', linestyle='--',
                      alpha=0.3, linewidth=1)
            ax.text(len(losses) * 0.7, final_train - 0.05,
                   f'Final train: {final_train:.3f}',
                   fontsize=10, color='#2E86AB', weight='bold')
            ax.text(len(losses) * 0.7, final_val + 0.05,
                   f'Final val: {final_val:.3f}',
                   fontsize=10, color='#FF6B6B', weight='bold')

        plt.tight_layout()
        return fig
