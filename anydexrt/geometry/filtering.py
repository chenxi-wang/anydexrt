"""
Author: Chenxi Wang, Ying Feng
"""
import time
import numpy as np

class BatchOneEuroFilter:
    """
    Batch version of 1Euro filter for N-dimensional signals.
    Applies the same 1Euro equations element-wise across all dimensions.
    """

    def __init__(self, ndim, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        """
        Args:
            ndim: Number of signal dimensions.
            mincutoff: Minimum cutoff frequency (Hz). Scalar or array-like [ndim].
            beta: Speed coefficient. Scalar or array-like [ndim].
            dcutoff: Cutoff frequency for derivative (Hz). Scalar or array-like [ndim].
        """
        if ndim <= 0:
            raise ValueError(f"ndim must be positive, got {ndim}")

        self.ndim = int(ndim)
        self.mincutoff = np.asarray(mincutoff, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self.dcutoff = np.asarray(dcutoff, dtype=float)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def reset(self):
        """Reset the filter state."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t=None):
        """
        Filter an N-dimensional value.

        Args:
            x: Input value as list or numpy array with size `ndim`.
            t: Timestamp (seconds). If None, uses time.time().

        Returns:
            Filtered value as numpy array with shape [ndim].
        """
        if t is None:
            t = time.time()

        x = np.asarray(x, dtype=float)
        if x.size != self.ndim:
            raise ValueError(
                f"Input size must match ndim={self.ndim}, got size={x.size}"
            )
        x = x.reshape(self.ndim)

        if self.x_prev is None:
            # First value, no filtering
            self.x_prev = x.copy()
            self.dx_prev = np.zeros(self.ndim, dtype=float)
            self.t_prev = t
            return x.copy()

        # Compute time delta
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev.copy()

        # Estimate derivative (velocity)
        dx = (x - self.x_prev) / dt

        # Filter the derivative
        edx = self._smooth(dx, self.dx_prev, dt, self.dcutoff)
        self.dx_prev = edx

        # Compute adaptive cutoff frequency
        cutoff = self.mincutoff + self.beta * np.abs(edx)

        # Filter the signal
        filtered_x = self._smooth(x, self.x_prev, dt, cutoff)

        # Update state
        self.x_prev = filtered_x.copy()
        self.t_prev = t

        return filtered_x

    def _smooth(self, x, x_prev, dt, cutoff):
        """Exponential smoothing filter (element-wise)."""
        if dt <= 0:
            return x

        cutoff = np.asarray(cutoff, dtype=float)
        rate = 2.0 * np.pi * dt * cutoff
        alpha = np.divide(rate, 1.0 + rate)
        alpha = np.where(cutoff > 0, alpha, 1.0)
        return alpha * x + (1.0 - alpha) * x_prev
