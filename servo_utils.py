import numpy as np
from typing import List
import matplotlib.pyplot as plt


def plot_error_and_increment(errors: List[float], increments: List[tuple]) -> None:
    """
    Plot the error norm and XYZ increments over time.

    Args:
        errors (list of float): List of error norms at each time step.
        increments (list of tuple): List of XYZ increments at each time step, where each element is a tuple (dx, dy, dz).
    Returns:
        None
    """

    plt.figure(figsize=(14, 6))
    
    # Error Curve
    plt.subplot(1, 2, 1)
    plt.plot(errors, color='red', lw=2)
    plt.title('Error Norm over Time', fontsize=16)
    plt.xlabel('Time step', fontsize=12)
    plt.ylabel('Error Norm (px or mm)', fontsize=12)
    plt.grid(True)
    plt.tight_layout(pad=3.0)

    # Increment Curve
    plt.subplot(1, 2, 2)
    increments = np.array(increments)
    plt.plot(increments[:, 0], label='X Increment', color='r', lw=2)
    plt.plot(increments[:, 1], label='Y Increment', color='g', lw=2)
    plt.plot(increments[:, 2], label='Z Increment', color='b', lw=2)
    
    plt.title('XYZ Increment over Time', fontsize=16)
    plt.xlabel('Time step', fontsize=12)
    plt.ylabel('Increment (mm)', fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout(pad=3.0)

    plt.show()

