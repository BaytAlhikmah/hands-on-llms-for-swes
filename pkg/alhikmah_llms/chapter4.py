import numpy as np
import matplotlib.pyplot as plt


class Chapter4:
    """Visualization utilities for Chapter 4: Neural Networks from Scratch."""

    # ========== Part 1: Basic Neuron Visualizations ==========

    @staticmethod
    def visualize_neuron_variants(neuron_func, x_values, configs):
        """
        Visualize how different weight/bias combinations affect neuron output.

        Args:
            neuron_func: Function that takes (x, w, b) and returns output
            x_values: Array of input values to plot
            configs: List of (w, b, label) tuples to visualize
        """
        plt.figure(figsize=(12, 8))

        for i, (w, b, label) in enumerate(configs, 1):
            y_values = [neuron_func(x, w, b) for x in x_values]

            plt.subplot(2, 2, i)
            plt.plot(x_values, y_values, linewidth=2)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            plt.title(label, fontsize=12)
            plt.xlabel('Input (x)')
            plt.ylabel('Output (y)')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_activation_functions(sigmoid_func, relu_func, z_values):
        """
        Visualize sigmoid and ReLU activation functions side by side.

        Args:
            sigmoid_func: Sigmoid activation function
            relu_func: ReLU activation function
            z_values: Array of input values to plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Sigmoid
        axes[0].plot(z_values, sigmoid_func(z_values), 'g-', linewidth=2)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        axes[0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
        axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_xlabel('Input (z)', fontsize=12)
        axes[0].set_ylabel('Output', fontsize=12)
        axes[0].set_title('Sigmoid: Squashes to (0, 1)', fontsize=14, fontweight='bold')
        axes[0].set_ylim(-0.1, 1.1)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].text(0, 0.85, 'Output is ALWAYS\nbetween 0 and 1',
                    fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))

        # ReLU
        axes[1].plot(z_values, relu_func(z_values), 'r-', linewidth=2)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_xlabel('Input (z)', fontsize=12)
        axes[1].set_ylabel('Output', fontsize=12)
        axes[1].set_title('ReLU: Cuts off negative part', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].text(5, 8, 'Output can be ANY\nnon-negative number',
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_line_to_decision_boundary(w, b, numbers, labels):
        """
        Visualize how the neuron's line (y = wx + b) becomes a decision boundary point.

        Shows two perspectives:
        1. The neuron as a function: y = wx + b (a line in 2D)
        2. The decision boundary: where y = 0 (a point on the x-axis)

        Args:
            w: weight
            b: bias
            numbers: input data points
            labels: true labels
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: The neuron as a function (line in 2D)
        ax1 = axes[0]
        x_range = np.linspace(0, 11, 200)
        y_range = w * x_range + b

        # Plot the line y = wx + b
        ax1.plot(x_range, y_range, 'b-', linewidth=3, label=f'y = {w}x + {b}')

        # Mark y = 0 threshold
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2,
                   label='Decision threshold (y = 0)')

        # Shade regions
        ax1.fill_between(x_range, -10, 0, alpha=0.2, color='red',
                        label='y < 0 → Predict class 0')
        ax1.fill_between(x_range, 0, 20, alpha=0.2, color='blue',
                        label='y ≥ 0 → Predict class 1')

        # Find decision boundary (where y = 0)
        boundary_x = -b / w if w != 0 else 0
        boundary_y = 0

        # Mark the decision boundary point
        ax1.scatter([boundary_x], [boundary_y], color='green', s=400,
                   marker='*', edgecolors='black', linewidths=2, zorder=10,
                   label=f'Decision boundary\n(x = {boundary_x:.2f}, y = 0)')

        # Draw vertical line from decision point
        ax1.axvline(x=boundary_x, color='green', linestyle=':', linewidth=2, alpha=0.5)

        # Add annotation
        ax1.annotate('Solving y = 0:\n' + f'{w}x + {b} = 0\n' + f'x = {boundary_x:.2f}',
                    xy=(boundary_x, 0), xytext=(boundary_x + 2, 5),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))

        ax1.set_xlabel('Input (x)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Neuron Output (y)', fontsize=14, fontweight='bold')
        ax1.set_title('Neuron as a Function: y = wx + b', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 11)
        ax1.set_ylim(-10, 15)

        # Right plot: Decision boundary on number line (input space)
        ax2 = axes[1]

        # Shade regions
        ax2.fill_between([0, boundary_x], -0.5, 0.5, alpha=0.3, color='red',
                        label='Predict class 0 (y < 0)')
        ax2.fill_between([boundary_x, 11], -0.5, 0.5, alpha=0.3, color='blue',
                        label='Predict class 1 (y ≥ 0)')

        # Draw decision boundary
        ax2.axvline(x=boundary_x, color='green', linestyle='--', linewidth=4,
                   label=f'Decision boundary (x = {boundary_x:.2f})')

        # Plot data points
        for num, label in zip(numbers, labels):
            color = 'red' if label == 0 else 'blue'
            marker = 'o' if label == 0 else 's'
            y_val = w * num + b
            predicted = 1 if y_val >= 0 else 0
            correct = (predicted == label)
            edge_color = 'green' if correct else 'black'
            edge_width = 3 if correct else 2

            ax2.scatter([num], [0], c=color, s=300, marker=marker,
                       edgecolors=edge_color, linewidths=edge_width, zorder=5)
            ax2.text(num, -0.15, str(num), ha='center', va='top',
                    fontsize=10, fontweight='bold')

        ax2.set_xlim(0, 11)
        ax2.set_ylim(-0.4, 0.4)
        ax2.set_xlabel('Input (x)', fontsize=14, fontweight='bold')
        ax2.set_yticks([])
        ax2.set_title('Decision Boundary on Number Line', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')

        # Add big arrow connecting the two
        fig.text(0.48, 0.5, '→', fontsize=60, ha='center', va='center',
                color='green', fontweight='bold')
        fig.text(0.48, 0.38, 'Project to\nx-axis', fontsize=11, ha='center', va='top',
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow'))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_decision_attempts(numbers, labels, attempts):
        """
        Visualize decision boundary attempts on a number line.

        Args:
            numbers: List of input numbers
            labels: True labels (0 or 1)
            attempts: List of (w, b, description) tuples
        """
        fig, axes = plt.subplots(len(attempts), 1, figsize=(14, 4 * len(attempts)))

        # Handle single attempt case
        if len(attempts) == 1:
            axes = [axes]

        for ax, (w, b, description) in zip(axes, attempts):
            # Calculate decision boundary (where w*x + b = 0)
            boundary = -b / w

            # Plot the number line regions
            x_range = np.linspace(0, 11, 1000)

            # Shade regions based on prediction
            # Left of boundary: predict class 0 (red)
            # Right of boundary: predict class 1 (blue)
            ax.fill_between([0, boundary], -0.5, 0.5, alpha=0.3, color='red',
                           label='Predict "small" (class 0)')
            ax.fill_between([boundary, 11], -0.5, 0.5, alpha=0.3, color='blue',
                           label='Predict "big" (class 1)')

            # Draw decision boundary line
            ax.axvline(x=boundary, color='green', linestyle='--', linewidth=3,
                      label=f'Decision boundary (x={boundary:.2f})')

            # Plot the actual data points
            for num, label in zip(numbers, labels):
                color = 'red' if label == 0 else 'blue'
                marker = 'o' if label == 0 else 's'
                # Check if prediction is correct
                predicted = 1 if num >= boundary else 0
                correct = (predicted == label)
                edge_color = 'green' if correct else 'black'
                edge_width = 3 if correct else 2

                ax.scatter([num], [0], c=color, s=300, marker=marker,
                          edgecolors=edge_color, linewidths=edge_width, zorder=5)

                # Add label showing the number
                ax.text(num, -0.15, str(num), ha='center', va='top',
                       fontsize=10, fontweight='bold')

            # Formatting
            ax.set_xlim(0, 11)
            ax.set_ylim(-0.4, 0.4)
            ax.set_xlabel('Input Value (x)', fontsize=12)
            ax.set_yticks([])
            ax.set_title(f'{description}: w={w}, b={b} (boundary at x={boundary:.2f})',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3, axis='x')

            # Add accuracy annotation
            correct_count = sum(1 for num, label in zip(numbers, labels)
                              if ((num >= boundary) == label))
            accuracy = correct_count / len(numbers) * 100
            ax.text(0.98, 0.95, f'Accuracy: {accuracy:.0f}%',
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   ha='right', va='top')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_sigmoid_on_problem(w, b, numbers, labels, neuron_func, sigmoid_func):
        """
        Visualize how sigmoid transforms the linear neuron output into probabilities.

        Shows three panels:
        1. Linear neuron output: y = wx + b
        2. After sigmoid: probability = σ(wx + b)
        3. Final classification with 0.5 threshold

        Args:
            w: weight
            b: bias
            numbers: input data points
            labels: true labels
            neuron_func: neuron function (x, w, b) -> output
            sigmoid_func: sigmoid activation function
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        x_smooth = np.linspace(0, 11, 300)

        # Panel 1: Linear neuron output (before sigmoid)
        ax1 = axes[0]
        y_linear = w * x_smooth + b

        ax1.plot(x_smooth, y_linear, 'b-', linewidth=3, label=f'y = {w}x + ({b})')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2,
                   label='Threshold: y = 0')

        # Shade regions
        ax1.fill_between(x_smooth, -15, 0, alpha=0.2, color='red',
                        label='y < 0 (will predict 0)')
        ax1.fill_between(x_smooth, 0, 15, alpha=0.2, color='blue',
                        label='y ≥ 0 (will predict 1)')

        # Plot actual outputs for data points
        for num, label in zip(numbers, labels):
            y_val = neuron_func(num, w, b)
            color = 'red' if label == 0 else 'blue'
            marker = 'o' if label == 0 else 's'
            ax1.scatter([num], [y_val], c=color, s=150, marker=marker,
                       edgecolors='k', linewidth=2, zorder=5)

        ax1.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Raw Output (y)', fontsize=12, fontweight='bold')
        ax1.set_title('Step 1: Linear Neuron\ny = wx + b',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 11)
        ax1.set_ylim(-10, 12)

        # Panel 2: After sigmoid (probabilities)
        ax2 = axes[1]
        probs = sigmoid_func(y_linear)

        ax2.plot(x_smooth, probs, 'g-', linewidth=3, label='p = σ(wx + b)')
        ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2,
                   label='Threshold: p = 0.5')

        # Mark decision boundary
        boundary_x = -b / w
        ax2.axvline(x=boundary_x, color='orange', linestyle=':', linewidth=2,
                   alpha=0.7, label=f'Decision at x={boundary_x:.1f}')

        # Shade regions
        ax2.fill_between(x_smooth, 0, 0.5, alpha=0.2, color='red',
                        label='p < 0.5 (predict 0)')
        ax2.fill_between(x_smooth, 0.5, 1, alpha=0.2, color='blue',
                        label='p ≥ 0.5 (predict 1)')

        # Plot actual probabilities for data points
        for num, label in zip(numbers, labels):
            y_val = neuron_func(num, w, b)
            prob = sigmoid_func(y_val)
            color = 'red' if label == 0 else 'blue'
            marker = 'o' if label == 0 else 's'
            ax2.scatter([num], [prob], c=color, s=150, marker=marker,
                       edgecolors='k', linewidth=2, zorder=5)

        ax2.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax2.set_title('Step 2: Apply Sigmoid\nσ(y) = 1/(1+e⁻ʸ)',
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 11)
        ax2.set_ylim(-0.05, 1.05)

        # Panel 3: Final classification on number line
        ax3 = axes[2]

        # Shade regions
        ax3.fill_between([0, boundary_x], -0.5, 0.5, alpha=0.3, color='red',
                        label='Predict class 0')
        ax3.fill_between([boundary_x, 11], -0.5, 0.5, alpha=0.3, color='blue',
                        label='Predict class 1')

        # Decision boundary
        ax3.axvline(x=boundary_x, color='green', linestyle='--', linewidth=4,
                   label=f'Decision boundary\n(x={boundary_x:.1f})')

        # Plot data points with predictions
        for num, label in zip(numbers, labels):
            y_val = neuron_func(num, w, b)
            prob = sigmoid_func(y_val)
            predicted = 1 if prob >= 0.5 else 0

            color = 'red' if label == 0 else 'blue'
            marker = 'o' if label == 0 else 's'
            correct = (predicted == label)
            edge_color = 'green' if correct else 'black'
            edge_width = 3 if correct else 2

            ax3.scatter([num], [0], c=color, s=300, marker=marker,
                       edgecolors=edge_color, linewidths=edge_width, zorder=5)

            # Add probability label
            ax3.text(num, 0.25, f'{prob:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax3.set_xlim(0, 11)
        ax3.set_ylim(-0.4, 0.4)
        ax3.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
        ax3.set_yticks([])
        ax3.set_title('Step 3: Classification\nif p ≥ 0.5: class 1, else: class 0',
                     fontsize=13, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='x')

        # Add arrows between panels
        fig.text(0.31, 0.5, '→', fontsize=40, ha='center', va='center',
                color='darkgreen', fontweight='bold')
        fig.text(0.31, 0.35, 'Apply\nσ(·)', fontsize=11, ha='center', va='top',
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen'))

        fig.text(0.645, 0.5, '→', fontsize=40, ha='center', va='center',
                color='darkgreen', fontweight='bold')
        fig.text(0.645, 0.35, 'Threshold\nat 0.5', fontsize=11, ha='center', va='top',
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen'))

        plt.tight_layout()
        plt.show()

    # ========== Part 2: Gradient Descent Visualizations ==========

    @staticmethod
    def visualize_2d_loss_landscape(w_range, b_range, compute_loss_func):
        """
        Visualize 2D loss landscape as a contour plot.

        Args:
            w_range: tuple (w_min, w_max, num_points)
            b_range: tuple (b_min, b_max, num_points)
            compute_loss_func: function(w, b) -> loss
        """
        w_min, w_max, w_points = w_range
        b_min, b_max, b_points = b_range

        w_values = np.linspace(w_min, w_max, w_points)
        b_values = np.linspace(b_min, b_max, b_points)

        W, B = np.meshgrid(w_values, b_values)
        losses = np.zeros_like(W)

        for i in range(len(b_values)):
            for j in range(len(w_values)):
                losses[i, j] = compute_loss_func(w_values[j], b_values[i])

        # Find global minimum
        min_idx = np.unravel_index(np.argmin(losses), losses.shape)
        w_opt = w_values[min_idx[1]]
        b_opt = b_values[min_idx[0]]
        loss_opt = losses[min_idx]

        plt.figure(figsize=(12, 8))

        # Contour plot
        levels = np.linspace(losses.min(), losses.min() + 0.5, 30)
        contour = plt.contour(W, B, losses, levels=levels, cmap='viridis', alpha=0.6)
        contourf = plt.contourf(W, B, losses, levels=levels, cmap='viridis', alpha=0.4)
        plt.colorbar(contourf, label='Loss (Binary Cross-Entropy)')

        # Mark the global minimum
        plt.scatter([w_opt], [b_opt], color='red', s=400, marker='*',
                   edgecolors='white', linewidths=2, zorder=10,
                   label=f'Better B&W\nw={w_opt:.3f}, b={b_opt:.3f}\nloss={loss_opt:.4f}')

        # Mark the constrained minimum (b=-5)
        b_constrained = -5.0
        if b_min <= b_constrained <= b_max:
            # Find best w at b=-5
            b_idx = np.argmin(np.abs(b_values - b_constrained))
            losses_at_b = losses[b_idx, :]
            w_constrained_idx = np.argmin(losses_at_b)
            w_constrained = w_values[w_constrained_idx]
            loss_constrained = losses_at_b[w_constrained_idx]

            plt.scatter([w_constrained], [b_constrained], color='orange', s=300,
                       marker='o', edgecolors='white', linewidths=2, zorder=9,
                       label=f'Constrained (b=-5)\nw={w_constrained:.3f}\nloss={loss_constrained:.4f}')

            # Draw horizontal line at b=-5
            plt.axhline(y=b_constrained, color='orange', linestyle='--',
                       linewidth=2, alpha=0.5, label='Constraint: b = -5')

        plt.xlabel('Weight (w)', fontsize=14, fontweight='bold')
        plt.ylabel('Bias (b)', fontsize=14, fontweight='bold')
        plt.title('2D Loss Landscape: Optimizing Both w and b', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return w_opt, b_opt, loss_opt

    @staticmethod
    def visualize_sigmoids_at_different_points(points_list, numbers, labels, neuron_func, sigmoid_func):
        """
        Show sigmoid curves at different (w, b) points from the loss landscape.

        Args:
            points_list: List of (w, b, loss, label, color) tuples
            numbers, labels: Data
            neuron_func, sigmoid_func: Functions
        """
        n_points = len(points_list)
        fig, axes = plt.subplots(1, n_points, figsize=(6*n_points, 5))

        if n_points == 1:
            axes = [axes]

        x_smooth = np.linspace(0, 11, 300)

        for ax, (w, b, loss, label, color) in zip(axes, points_list):
            # Compute sigmoid curve
            y_vals = w * x_smooth + b
            probs = sigmoid_func(y_vals)

            # Plot sigmoid
            ax.plot(x_smooth, probs, color=color, linewidth=3,
                   label=f'σ({w:.2f}x + {b:.2f})')
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

            # Decision boundary
            boundary = -b / w
            ax.axvline(x=boundary, color='green', linestyle='--', linewidth=2,
                      alpha=0.7, label=f'Boundary: x={boundary:.2f}')

            # Shade regions
            ax.fill_between(x_smooth, 0, 0.5, alpha=0.1, color='red')
            ax.fill_between(x_smooth, 0.5, 1, alpha=0.1, color='blue')

            # Plot data points with probabilities
            for num, true_label in zip(numbers, labels):
                y_val = neuron_func(num, w, b)
                prob = sigmoid_func(y_val)
                marker_color = 'red' if true_label == 0 else 'blue'
                marker = 'o' if true_label == 0 else 's'
                ax.scatter([num], [prob], c=marker_color, s=100, marker=marker,
                          edgecolors='k', linewidth=1.5, zorder=5, alpha=0.7)

            ax.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
            ax.set_title(f'{label}\nw={w:.2f}, b={b:.2f}\nLoss={loss:.4f}',
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 11)
            ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_constrained_vs_global(w_const, b_const, loss_const,
                                     w_global, b_global, loss_global,
                                     numbers, labels, neuron_func, sigmoid_func):
        """
        Compare constrained optimization vs global optimization side-by-side.

        Args:
            w_const, b_const, loss_const: Constrained optimum
            w_global, b_global, loss_global: Global optimum
            numbers, labels: Data
            neuron_func, sigmoid_func: Functions
        """
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        x_smooth = np.linspace(0, 11, 300)

        # Top row: Sigmoid curves comparison
        for col, (w, b, loss, title, color) in enumerate([
            (w_const, b_const, loss_const, 'Constrained\n(b=-5.0 fixed)', 'orange'),
            (w_global, b_global, loss_global, 'Global Optimum\n(both w, b free)', 'red')
        ]):
            ax = fig.add_subplot(gs[0, col])

            # Compute sigmoid curve
            y_vals = w * x_smooth + b
            probs = sigmoid_func(y_vals)

            ax.plot(x_smooth, probs, color=color, linewidth=3,
                   label=f'σ({w:.3f}x + {b:.3f})')
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)

            # Decision boundary
            boundary = -b / w
            ax.axvline(x=boundary, color='green', linestyle='--', linewidth=2,
                      label=f'Boundary: x={boundary:.2f}')

            # Plot data points
            for num, label in zip(numbers, labels):
                y_val = neuron_func(num, w, b)
                prob = sigmoid_func(y_val)
                marker_color = 'red' if label == 0 else 'blue'
                marker = 'o' if label == 0 else 's'
                ax.scatter([num], [prob], c=marker_color, s=120, marker=marker,
                          edgecolors='k', linewidth=1.5, zorder=5)

            ax.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
            ax.set_title(f'{title}\nLoss: {loss:.4f}', fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 11)
            ax.set_ylim(-0.05, 1.05)

        # Middle row: Number lines with probabilities
        for col, (w, b, title, color) in enumerate([
            (w_const, b_const, 'Constrained', 'orange'),
            (w_global, b_global, 'Global Optimum', 'red')
        ]):
            ax = fig.add_subplot(gs[1, col])

            boundary = -b / w

            # Shade regions
            ax.fill_between([0, boundary], -0.5, 0.5, alpha=0.2, color='red')
            ax.fill_between([boundary, 11], -0.5, 0.5, alpha=0.2, color='blue')

            # Decision boundary
            ax.axvline(x=boundary, color='green', linestyle='--', linewidth=3)

            # Plot points with probability labels
            for num, label in zip(numbers, labels):
                y_val = neuron_func(num, w, b)
                prob = sigmoid_func(y_val)
                predicted = 1 if prob >= 0.5 else 0

                marker_color = 'red' if label == 0 else 'blue'
                marker = 'o' if label == 0 else 's'
                correct = (predicted == label)
                edge_color = 'green' if correct else 'black'
                edge_width = 3 if correct else 2

                ax.scatter([num], [0], c=marker_color, s=250, marker=marker,
                          edgecolors=edge_color, linewidths=edge_width, zorder=5)

                # Probability label
                ax.text(num, 0.25, f'{prob:.2f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            ax.set_xlim(0, 11)
            ax.set_ylim(-0.4, 0.4)
            ax.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
            ax.set_yticks([])
            ax.set_title(f'{title} Classification', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

        # Bottom row: Comparison table
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')

        # Create comparison data
        table_data = [
            ['Metric', 'Constrained (b=-5 fixed)', 'Global Optimum', 'Difference'],
            ['Weight (w)', f'{w_const:.4f}', f'{w_global:.4f}', f'{w_global-w_const:+.4f}'],
            ['Bias (b)', f'{b_const:.4f}', f'{b_global:.4f}', f'{b_global-b_const:+.4f}'],
            ['Loss', f'{loss_const:.6f}', f'{loss_global:.6f}',
             f'{loss_global-loss_const:.6f} ({(loss_const-loss_global)/loss_const*100:.1f}% better!)'],
            ['Decision Boundary', f'{-b_const/w_const:.4f}', f'{-b_global/w_global:.4f}',
             f'{-b_global/w_global + b_const/w_const:+.4f}'],
        ]

        table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                              bbox=[0.1, 0.3, 0.8, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style data rows
        for i in range(1, 5):
            for j in range(4):
                if j == 3:  # Difference column
                    table[(i, j)].set_facecolor('#FFF9C4')
                else:
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

        ax_table.text(0.5, 0.15, '💡 Key Insight: Larger w means steeper sigmoid = more confident predictions = lower loss!',
                     ha='center', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                     transform=ax_table.transAxes)

        plt.show()

    @staticmethod
    def visualize_loss_landscape(weight_values, losses, b_fixed):
        """
        Visualize 1D loss landscape showing the valley.

        Args:
            weight_values: Array of weight values
            losses: Corresponding loss values
            b_fixed: The fixed bias value used
        """
        plt.figure(figsize=(10, 6))
        plt.plot(weight_values, losses, linewidth=2, color='blue')
        plt.xlabel('Weight (w)', fontsize=12)
        plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
        plt.title(f'Loss Landscape (bias fixed at {b_fixed})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Mark the minimum
        best_idx = np.argmin(losses)
        best_w = weight_values[best_idx]
        best_loss = losses[best_idx]
        plt.scatter([best_w], [best_loss], color='red', s=200, zorder=5,
                   label=f'Minimum: w={best_w:.2f}, loss={best_loss:.4f}')
        plt.legend(fontsize=12)

        plt.show()

    @staticmethod
    def visualize_gradient_tangent(weight_values, losses, w_current, loss_current, grad_w_current):
        """
        Visualize gradient as a tangent line on the loss curve.

        Args:
            weight_values: Array of weight values for loss curve
            losses: Corresponding loss values
            w_current: Current weight position
            loss_current: Current loss value
            grad_w_current: Current gradient value
        """
        plt.figure(figsize=(12, 6))
        plt.plot(weight_values, losses, linewidth=2, label='Loss curve', color='blue', alpha=0.6)

        # Plot current position
        plt.scatter([w_current], [loss_current], c='red', s=200, zorder=5, label='Current position')

        # Plot tangent line (shows gradient)
        tangent_w = np.linspace(w_current - 0.8, w_current + 0.8, 100)
        tangent_loss = loss_current + grad_w_current * (tangent_w - w_current)
        plt.plot(tangent_w, tangent_loss, 'r--', linewidth=3,
                label=f'Tangent (slope = gradient = {grad_w_current:.3f})')

        # Arrow showing direction to move
        arrow_length = 0.3
        if grad_w_current > 0:
            plt.annotate('', xy=(w_current - arrow_length, loss_current),
                        xytext=(w_current, loss_current),
                        arrowprops=dict(arrowstyle='->', color='green', lw=3))
            plt.text(w_current - 0.6, loss_current + 0.05, 'Go this way!\n(decrease w)',
                    fontsize=11, color='green', fontweight='bold', ha='center')
        else:
            plt.annotate('', xy=(w_current + arrow_length, loss_current),
                        xytext=(w_current, loss_current),
                        arrowprops=dict(arrowstyle='->', color='green', lw=3))
            plt.text(w_current + 0.6, loss_current + 0.05, 'Go this way!\n(increase w)',
                    fontsize=11, color='green', fontweight='bold', ha='center')

        plt.xlabel('Weight (w)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Gradient = Slope of the Loss Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(w_current - 1, w_current + 1)

        plt.show()

    @staticmethod
    def visualize_training_progress(loss_history, title='Gradient Descent: Loss Decreasing Over Time'):
        """
        Visualize loss decreasing over training steps.

        Args:
            loss_history: List of loss values over training
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, 'r-', linewidth=2)
        plt.xlabel('Training Step', fontsize=14)
        plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ========== Part 2.5: Bridging 1D to 2D ==========

    @staticmethod
    def visualize_2d_dataset(inputs_2d, labels, feature_names=('Feature 1', 'Feature 2')):
        """
        Visualize 2D dataset as a scatter plot.

        Args:
            inputs_2d: Array of shape (n, 2) with 2D input points
            labels: Array of shape (n,) with binary labels
            feature_names: Tuple of (x_label, y_label) for axis labels
        """
        plt.figure(figsize=(10, 8))

        # Separate by class
        class_0 = inputs_2d[labels == 0]
        class_1 = inputs_2d[labels == 1]

        # Plot class 0
        plt.scatter(class_0[:, 0], class_0[:, 1], c='red', s=300, marker='o',
                   alpha=0.7, edgecolors='k', linewidth=2, label='Class 0', zorder=5)

        # Plot class 1
        plt.scatter(class_1[:, 0], class_1[:, 1], c='blue', s=300, marker='s',
                   alpha=0.7, edgecolors='k', linewidth=2, label='Class 1', zorder=5)

        # Add labels to points
        for i, (point, label) in enumerate(zip(inputs_2d, labels)):
            plt.text(point[0], point[1], f'{i+1}',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        plt.xlabel(feature_names[0], fontsize=14, fontweight='bold')
        plt.ylabel(feature_names[1], fontsize=14, fontweight='bold')
        plt.title('2D Dataset: Moving from 1D to 2D', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_2d_learned_boundary(inputs_2d, labels, w1, w2, b,
                                     neuron_func, sigmoid_func,
                                     feature_names=('Feature 1', 'Feature 2')):
        """
        Visualize learned decision boundary in 2D space.

        Args:
            inputs_2d: Array of shape (n, 2) with 2D input points
            labels: Array of shape (n,) with binary labels
            w1, w2, b: Learned weights and bias
            neuron_func: Neuron function (x1, x2, w1, w2, b) -> output
            sigmoid_func: Sigmoid activation function
            feature_names: Tuple of (x_label, y_label) for axis labels
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Create mesh for probability surface
        x1_min, x1_max = inputs_2d[:, 0].min() - 1, inputs_2d[:, 0].max() + 1
        x2_min, x2_max = inputs_2d[:, 1].min() - 1, inputs_2d[:, 1].max() + 1

        x1_range = np.linspace(x1_min, x1_max, 300)
        x2_range = np.linspace(x2_min, x2_max, 300)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        # Compute probabilities over the mesh
        Z = w1 * X1 + w2 * X2 + b
        probs = sigmoid_func(Z)

        # Left plot: Probability heatmap
        ax1 = axes[0]
        im = ax1.contourf(X1, X2, probs, levels=20, cmap='RdBu_r', alpha=0.8)
        ax1.contour(X1, X2, probs, levels=[0.5], colors='green', linewidths=4)
        plt.colorbar(im, ax=ax1, label='P(class 1)')

        # Plot data points
        for point, label in zip(inputs_2d, labels):
            color = 'red' if label == 0 else 'blue'
            marker = 'o' if label == 0 else 's'
            ax1.scatter(point[0], point[1], c=color, s=300, marker=marker,
                       alpha=0.9, edgecolors='k', linewidth=2, zorder=5)

        ax1.set_xlabel(feature_names[0], fontsize=13, fontweight='bold')
        ax1.set_ylabel(feature_names[1], fontsize=13, fontweight='bold')
        ax1.set_title('Probability Landscape\n(Green line = Decision Boundary)',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Right plot: Decision boundary line
        ax2 = axes[1]

        # Shade regions
        ax2.contourf(X1, X2, probs, levels=[0, 0.5], colors=['red'], alpha=0.2)
        ax2.contourf(X1, X2, probs, levels=[0.5, 1], colors=['blue'], alpha=0.2)

        # Draw decision boundary (where w1*x1 + w2*x2 + b = 0)
        # Rearranging: x2 = -(w1*x1 + b) / w2
        if abs(w2) > 1e-6:
            x1_line = np.linspace(x1_min, x1_max, 100)
            x2_line = -(w1 * x1_line + b) / w2
            ax2.plot(x1_line, x2_line, 'g-', linewidth=4,
                    label=f'Decision Line: {w1:.2f}·x1 + {w2:.2f}·x2 + {b:.2f} = 0')

        # Plot data points with predictions
        for point, label in zip(inputs_2d, labels):
            y = w1 * point[0] + w2 * point[1] + b
            prob = sigmoid_func(y)
            predicted = 1 if prob >= 0.5 else 0

            color = 'red' if label == 0 else 'blue'
            marker = 'o' if label == 0 else 's'
            correct = (predicted == label)
            edge_color = 'green' if correct else 'black'
            edge_width = 3 if correct else 2

            ax2.scatter(point[0], point[1], c=color, s=300, marker=marker,
                       alpha=0.9, edgecolors=edge_color, linewidths=edge_width, zorder=5)

            # Show probability
            ax2.text(point[0], point[1] - 0.5, f'{prob:.2f}',
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

        ax2.set_xlabel(feature_names[0], fontsize=13, fontweight='bold')
        ax2.set_ylabel(feature_names[1], fontsize=13, fontweight='bold')
        ax2.set_title('Decision Boundary: The Line Separating Classes',
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(x1_min, x1_max)
        ax2.set_ylim(x2_min, x2_max)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_gradient_descent_2d(inputs_2d, labels, w1_history, w2_history,
                                     b_history, loss_history, compute_loss_func):
        """
        Visualize gradient descent in 2D parameter space (w1, w2).

        Args:
            inputs_2d: Training data
            labels: Training labels
            w1_history, w2_history, b_history: Parameter histories
            loss_history: Loss history
            compute_loss_func: Function(w1, w2, b) -> loss
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Loss over time
        ax1 = axes[0]
        ax1.plot(loss_history, 'b-', linewidth=2)
        ax1.set_xlabel('Training Step', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Loss (Binary Cross-Entropy)', fontsize=13, fontweight='bold')
        ax1.set_title('Loss Decreasing Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Annotate start and end
        ax1.scatter([0], [loss_history[0]], c='red', s=200, zorder=5,
                   label=f'Start: loss={loss_history[0]:.4f}')
        ax1.scatter([len(loss_history)-1], [loss_history[-1]], c='green', s=200, zorder=5,
                   label=f'End: loss={loss_history[-1]:.4f}')
        ax1.legend(fontsize=11)

        # Right: Parameter evolution in 3D bar chart
        ax2 = axes[1]

        # Show how parameters changed
        param_changes = {
            'w1': (w1_history[0], w1_history[-1]),
            'w2': (w2_history[0], w2_history[-1]),
            'b': (b_history[0], b_history[-1])
        }

        x_pos = np.arange(len(param_changes))
        initial_vals = [v[0] for v in param_changes.values()]
        final_vals = [v[1] for v in param_changes.values()]

        width = 0.35
        ax2.bar(x_pos - width/2, initial_vals, width, label='Initial', color='red', alpha=0.7)
        ax2.bar(x_pos + width/2, final_vals, width, label='Final', color='green', alpha=0.7)

        ax2.set_xlabel('Parameter', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Value', fontsize=13, fontweight='bold')
        ax2.set_title('Parameter Evolution', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(param_changes.keys(), fontsize=12, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.show()

    # ========== Part 3: Decision Boundary Visualizations ==========

    @staticmethod
    def visualize_single_neuron_decision_boundary(numbers, labels, w, b, neuron_func, sigmoid_func):
        """
        Visualize decision boundary learned by a single neuron.

        Args:
            numbers: Input data points
            labels: True labels (0 or 1)
            w: Learned weight
            b: Learned bias
            neuron_func: Neuron function (x, w, b) -> output
            sigmoid_func: Sigmoid activation function
        """
        # Create a smooth range for visualization
        x_range = np.linspace(0, 11, 300)

        # Compute predictions
        z_range = w * x_range + b
        probs = sigmoid_func(z_range)

        # Decision boundary is where sigmoid(w*x + b) = 0.5
        # This occurs when w*x + b = 0, so x = -b/w
        boundary_x = -b / w

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Probability curve with decision boundary
        axes[0].plot(x_range, probs, 'b-', linewidth=3, label='Predicted Probability P(big)')
        axes[0].axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
        axes[0].axvline(x=boundary_x, color='green', linestyle='--', linewidth=2,
                        label=f'Decision Boundary (x={boundary_x:.2f})')

        # Shade regions
        axes[0].fill_between(x_range, 0, 1, where=(x_range < boundary_x),
                            alpha=0.2, color='red', label='Predict "small" (class 0)')
        axes[0].fill_between(x_range, 0, 1, where=(x_range >= boundary_x),
                            alpha=0.2, color='blue', label='Predict "big" (class 1)')

        # Plot actual data points
        for num, label in zip(numbers, labels):
            z_val = neuron_func(num, w, b)
            prob = sigmoid_func(z_val)
            color = 'red' if label == 0 else 'blue'
            marker = 'o' if label == 0 else 's'
            axes[0].scatter([num], [prob], c=color, s=150, marker=marker,
                          edgecolors='k', linewidth=2, zorder=5)

        axes[0].set_xlabel('Input Value (x)', fontsize=12)
        axes[0].set_ylabel('Predicted Probability P(big)', fontsize=12)
        axes[0].set_title('Decision Boundary: The Line the Neuron Learned', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-0.05, 1.05)

        # Right: Data points with decision line
        added_labels = set()

        for num, label in zip(numbers, labels):
            color = 'red' if label == 0 else 'blue'
            marker = 'o' if label == 0 else 's'
            label_text = 'Small (class 0)' if label == 0 else 'Big (class 1)'

            # Add label only for first occurrence of each class
            if label not in added_labels:
                axes[1].scatter([num], [0], c=color, s=200, marker=marker,
                              edgecolors='k', linewidth=2, label=label_text)
                added_labels.add(label)
            else:
                axes[1].scatter([num], [0], c=color, s=200, marker=marker,
                              edgecolors='k', linewidth=2)

        axes[1].axvline(x=boundary_x, color='green', linestyle='--', linewidth=3,
                        label=f'Decision Line (x={boundary_x:.2f})')

        axes[1].fill_between([0, boundary_x], -0.5, 0.5, alpha=0.2, color='red')
        axes[1].fill_between([boundary_x, 11], -0.5, 0.5, alpha=0.2, color='blue')

        axes[1].set_xlabel('Input Value (x)', fontsize=12)
        axes[1].set_ylabel('(Arbitrary)', fontsize=10)
        axes[1].set_title('The Neuron Drew a Line to Separate Classes', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[1].set_ylim(-0.3, 0.3)
        axes[1].set_yticks([])
        axes[1].set_xlim(0, 11)

        plt.tight_layout()
        plt.show()

    # ========== Part 4: XOR Problem Visualizations ==========

    @staticmethod
    def visualize_network_architectures():
        """
        Visualize the difference between single neuron and 2-layer network architectures.

        Shows:
        - Left: Single neuron (can only draw one line)
        - Right: 2-layer network (can combine multiple lines)
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Single Neuron Architecture
        ax1 = axes[0]
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')

        # Input layer
        input_y = [7, 3]
        for i, y in enumerate(input_y):
            circle = plt.Circle((1, y), 0.4, color='lightblue', ec='black', linewidth=2, zorder=3)
            ax1.add_patch(circle)
            ax1.text(1, y, f'x{i+1}', ha='center', va='center', fontsize=12, fontweight='bold')
            ax1.text(0.2, y, f'Input\nx{i+1}', ha='right', va='center', fontsize=10)

        # Output neuron
        output_x, output_y = 5, 5
        circle = plt.Circle((output_x, output_y), 0.5, color='lightcoral', ec='black', linewidth=2, zorder=3)
        ax1.add_patch(circle)
        ax1.text(output_x, output_y, 'σ', ha='center', va='center', fontsize=16, fontweight='bold')
        ax1.text(output_x + 1.5, output_y, 'Output\n(0 or 1)', ha='left', va='center', fontsize=10)

        # Connections
        for y in input_y:
            ax1.plot([1.4, output_x - 0.5], [y, output_y], 'k-', linewidth=2, alpha=0.6, zorder=1)
            # Add weight labels
            mid_x, mid_y = (1.4 + output_x - 0.5) / 2, (y + output_y) / 2
            ax1.text(mid_x, mid_y + 0.3, 'w', fontsize=10, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Add bias
        ax1.text(output_x, output_y - 1.2, 'bias (b)', ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

        ax1.set_title('Single Neuron\n(Can only draw ONE line)', fontsize=14, fontweight='bold', pad=20)
        ax1.text(5, 0.5, 'y = w₁·x₁ + w₂·x₂ + b\nOne decision boundary',
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Right: 2-Layer Network Architecture
        ax2 = axes[1]
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')

        # Input layer
        for i, y in enumerate(input_y):
            circle = plt.Circle((1, y), 0.4, color='lightblue', ec='black', linewidth=2, zorder=3)
            ax2.add_patch(circle)
            ax2.text(1, y, f'x{i+1}', ha='center', va='center', fontsize=12, fontweight='bold')
            ax2.text(0.2, y, f'Input\nx{i+1}', ha='right', va='center', fontsize=10)

        # Hidden layer (2 neurons)
        hidden_y = [7, 3]
        hidden_x = 4
        for i, y in enumerate(hidden_y):
            circle = plt.Circle((hidden_x, y), 0.4, color='lightgreen', ec='black', linewidth=2, zorder=3)
            ax2.add_patch(circle)
            ax2.text(hidden_x, y, 'ReLU', ha='center', va='center', fontsize=9, fontweight='bold')
            ax2.text(hidden_x, y - 0.8, f'h{i+1}', ha='center', va='top', fontsize=10)

        ax2.text(hidden_x, 9, 'Hidden Layer\n(learns features)', ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

        # Output layer
        output_x, output_y = 7.5, 5
        circle = plt.Circle((output_x, output_y), 0.5, color='lightcoral', ec='black', linewidth=2, zorder=3)
        ax2.add_patch(circle)
        ax2.text(output_x, output_y, 'σ', ha='center', va='center', fontsize=16, fontweight='bold')
        ax2.text(output_x + 1.5, output_y, 'Output\n(0 or 1)', ha='left', va='center', fontsize=10)

        # Connections: Input to Hidden
        for input_y_val in input_y:
            for hidden_y_val in hidden_y:
                ax2.plot([1.4, hidden_x - 0.4], [input_y_val, hidden_y_val],
                        'k-', linewidth=1.5, alpha=0.4, zorder=1)

        # Label input->hidden weights
        ax2.text(2.5, 8, 'W₁', fontsize=11, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Connections: Hidden to Output
        for hidden_y_val in hidden_y:
            ax2.plot([hidden_x + 0.4, output_x - 0.5], [hidden_y_val, output_y],
                    'k-', linewidth=2, alpha=0.6, zorder=1)

        # Label hidden->output weights
        ax2.text(5.7, 6.5, 'W₂', fontsize=11, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax2.set_title('2-Layer Network\n(Can combine MULTIPLE lines)', fontsize=14, fontweight='bold', pad=20)
        ax2.text(5, 0.5, 'Hidden neurons learn features\nOutput combines them',
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add arrow showing progression
        fig.text(0.5, 0.95, 'From Simple to Powerful →', ha='center', fontsize=16,
                fontweight='bold', color='darkblue')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_xor_problem(xor_inputs, xor_labels):
        """
        Visualize the XOR problem in 2D space.

        Args:
            xor_inputs: Array of shape (4, 2) with XOR input points
            xor_labels: Array of shape (4,) with XOR labels
        """
        plt.figure(figsize=(8, 8))

        for (x1, x2), label in zip(xor_inputs, xor_labels):
            color = 'red' if label == 0 else 'blue'
            plt.scatter(x1, x2, c=color, s=500, alpha=0.6, edgecolors='k', linewidth=3)
            plt.text(x1, x2, f"({x1}, {x2})\n→ {label}",
                    ha='center', va='center', fontsize=12, fontweight='bold')

        plt.xlabel('x1', fontsize=14)
        plt.ylabel('x2', fontsize=14)
        plt.title('XOR Problem: Can You Draw ONE Line to Separate Red from Blue?',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.show()

    @staticmethod
    def visualize_xor_sigmoid_attempts(xor_inputs, xor_labels, sigmoid_func):
        """
        Visualize sigmoid probabilities for three line attempts on XOR.

        Shows that even with sigmoid, a single line can't solve XOR.

        Args:
            xor_inputs: XOR input points
            xor_labels: XOR labels
            sigmoid_func: Sigmoid activation function
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Define the three line attempts with their weights
        attempts = [
            {
                'name': 'Horizontal Line',
                'equation': 'x2 = 0.5',
                'w1': 0.0, 'w2': 1.0, 'b': -0.5,
                'description': 'Separates by x2 value'
            },
            {
                'name': 'Vertical Line',
                'equation': 'x1 = 0.5',
                'w1': 1.0, 'w2': 0.0, 'b': -0.5,
                'description': 'Separates by x1 value'
            },
            {
                'name': 'Diagonal Line',
                'equation': 'x1 + x2 = 1',
                'w1': 1.0, 'w2': 1.0, 'b': -1.0,
                'description': 'Separates diagonally'
            }
        ]

        # Create mesh for probability surface
        x1_range = np.linspace(-0.5, 1.5, 200)
        x2_range = np.linspace(-0.5, 1.5, 200)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        for ax, attempt in zip(axes, attempts):
            # Compute sigmoid probabilities over the entire space
            Z = attempt['w1'] * X1 + attempt['w2'] * X2 + attempt['b']
            probs = sigmoid_func(Z)

            # Plot probability heatmap
            im = ax.contourf(X1, X2, probs, levels=20, cmap='RdBu_r', alpha=0.8)
            ax.contour(X1, X2, probs, levels=[0.5], colors='green', linewidths=3)

            # Add colorbar
            plt.colorbar(im, ax=ax, label='P(class 1)')

            # Plot XOR data points
            for (x1, x2), label in zip(xor_inputs, xor_labels):
                color = 'red' if label == 0 else 'blue'
                marker = 'o' if label == 0 else 's'

                # Compute probability for this point
                z = attempt['w1'] * x1 + attempt['w2'] * x2 + attempt['b']
                prob = sigmoid_func(z)
                predicted = 1 if prob >= 0.5 else 0
                correct = (predicted == label)

                edge_color = 'green' if correct else 'black'
                edge_width = 3 if correct else 2

                ax.scatter(x1, x2, c=color, s=400, marker=marker,
                          edgecolors=edge_color, linewidths=edge_width, zorder=5)

                # Show probability value
                ax.text(x1, x2-0.15, f'{prob:.2f}', ha='center', va='top',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

            ax.set_xlabel('x1', fontsize=12)
            ax.set_ylabel('x2', fontsize=12)
            ax.set_title(f'{attempt["name"]}\n{attempt["equation"]}',
                        fontsize=12, fontweight='bold')
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Sigmoid Probabilities: All Single Lines Fail on XOR!',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_xor_line_attempts(xor_inputs, xor_labels):
        """
        Visualize three different line attempts to separate XOR classes.

        Args:
            xor_inputs: Array of shape (4, 2) with XOR input points
            xor_labels: Array of shape (4,) with XOR labels
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Define three different line attempts
        lines = [
            {
                'name': 'Horizontal Line',
                'equation': 'x2 = 0.5',
                'x1_range': np.linspace(-0.5, 1.5, 100),
                'x2_func': lambda x1: np.full_like(x1, 0.5),
                'description': 'Separates by x2 value'
            },
            {
                'name': 'Vertical Line',
                'equation': 'x1 = 0.5',
                'x1_range': np.full(100, 0.5),
                'x2_func': lambda _: np.linspace(-0.5, 1.5, 100),
                'description': 'Separates by x1 value'
            },
            {
                'name': 'Diagonal Line',
                'equation': 'x1 + x2 = 1',
                'x1_range': np.linspace(-0.5, 1.5, 100),
                'x2_func': lambda x1: 1 - x1,
                'description': 'Separates diagonally'
            }
        ]

        for idx, (ax, line_info) in enumerate(zip(axes, lines)):
            # Plot XOR data points
            for (x1, x2), label in zip(xor_inputs, xor_labels):
                color = 'red' if label == 0 else 'blue'
                marker = 'o' if label == 0 else 's'
                size = 400
                ax.scatter(x1, x2, c=color, s=size, alpha=0.7, edgecolors='k', linewidth=2, zorder=5)
                ax.text(x1, x2, f'({x1},{x2})', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white')

            # Plot the line
            x1_line = line_info['x1_range']
            if callable(line_info['x2_func']):
                x2_line = line_info['x2_func'](x1_line)
            else:
                x2_line = line_info['x2_func']

            ax.plot(x1_line, x2_line, 'g-', linewidth=3, label=line_info['equation'])

            # Shade regions on either side of the line
            if idx == 0:  # Horizontal
                ax.fill_between([-0.5, 1.5], -0.5, 0.5, alpha=0.2, color='red')
                ax.fill_between([-0.5, 1.5], 0.5, 1.5, alpha=0.2, color='blue')
            elif idx == 1:  # Vertical
                ax.fill_betweenx([-0.5, 1.5], -0.5, 0.5, alpha=0.2, color='red')
                ax.fill_betweenx([-0.5, 1.5], 0.5, 1.5, alpha=0.2, color='blue')
            else:  # Diagonal
                ax.fill_between(x1_line, -0.5, x2_line, where=(x2_line >= -0.5), alpha=0.2, color='red')
                ax.fill_between(x1_line, x2_line, 1.5, where=(x2_line <= 1.5), alpha=0.2, color='blue')

            ax.set_xlabel('x1', fontsize=12)
            ax.set_ylabel('x2', fontsize=12)
            ax.set_title(f'{line_info["name"]}\n{line_info["equation"]}',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)
            ax.set_aspect('equal')

        plt.suptitle('Attempting to Separate XOR with Single Lines - All Fail!',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_xor_decision_boundary(W1, b1, W2, b2, forward_func, xor_inputs, xor_labels,
                                       title="XOR Network Decision Boundary"):
        """
        Visualize 2D decision boundary learned by XOR network.

        Args:
            W1, b1: Hidden layer weights and biases
            W2, b2: Output layer weights and bias
            forward_func: Forward pass function (X, W1, b1, W2, b2) -> (predictions, hidden)
            xor_inputs: XOR input points
            xor_labels: XOR labels
            title: Plot title
        """
        # Create mesh grid
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))

        # Compute predictions for each point in the mesh
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z, _ = forward_func(grid_points, W1, b1, W2, b2)
        Z = Z.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(10, 8))

        # Plot decision boundary as filled contours
        contourf = plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
        plt.colorbar(contourf, label='Predicted Probability')

        # Plot decision line at 0.5
        contour = plt.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=3)
        plt.clabel(contour, inline=True, fontsize=10)

        # Plot data points
        for (x1, x2), label in zip(xor_inputs, xor_labels):
            color = 'red' if label == 0 else 'blue'
            marker = 'o' if label == 0 else 's'
            plt.scatter([x1], [x2], c=color, s=300, marker=marker,
                       edgecolors='k', linewidth=2, zorder=5)
            plt.text(x1, x2 + 0.15, f'({x1},{x2})→{label}',
                    ha='center', fontsize=10, fontweight='bold')

        plt.xlabel('x1', fontsize=14)
        plt.ylabel('x2', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_hidden_neuron_activations(W1, b1, xor_inputs, xor_labels):
        """
        Visualize what each hidden neuron detects in the XOR network.

        Args:
            W1, b1: Hidden layer weights and biases
            xor_inputs: XOR input points
            xor_labels: XOR labels
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for i in range(2):
            ax = axes[i]

            # Create mesh
            xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                                np.linspace(-0.5, 1.5, 200))
            grid_points = np.c_[xx.ravel(), yy.ravel()]

            # Compute hidden layer activations
            Z1 = grid_points @ W1.T + b1
            H = np.maximum(0, Z1)  # ReLU
            neuron_activation = H[:, i].reshape(xx.shape)

            # Plot
            contourf = ax.contourf(xx, yy, neuron_activation, levels=20, cmap='viridis', alpha=0.8)
            ax.contour(xx, yy, neuron_activation, levels=10, colors='white', linewidths=0.5, alpha=0.3)
            plt.colorbar(contourf, ax=ax, label='Activation')

            # Plot data points
            for (x1, x2), label in zip(xor_inputs, xor_labels):
                color = 'red' if label == 0 else 'blue'
                ax.scatter([x1], [x2], c=color, s=150, marker='o',
                          edgecolors='white', linewidth=2, zorder=5)

            ax.set_xlabel('x1', fontsize=12)
            ax.set_ylabel('x2', fontsize=12)
            ax.set_title(f'Hidden Neuron {i+1} Activation Pattern', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)

        plt.suptitle('What Each Hidden Neuron Detects', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()