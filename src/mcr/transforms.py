"""Media transforms shared by the simulator and the analysis (no parameters live here)."""
import numpy as np


def adstock(x: np.ndarray, lam: float) -> np.ndarray:
    """Geometric carry-over: A_t = x_t + lam * A_{t-1}."""
    out = np.empty(len(x))
    carry = 0.0
    for t, v in enumerate(np.asarray(x, dtype=float)):
        carry = v + lam * carry
        out[t] = carry
    return out


def adbudg_shape(a: np.ndarray, c: float, half: float) -> np.ndarray:
    """ADBUDG / Hill response in [0, 1): a^c / (half^c + a^c). S-shaped when c > 1."""
    a = np.maximum(np.asarray(a, dtype=float), 0.0)
    return a ** c / (half ** c + a ** c)


def adbudg_inflection(c: float, half: float) -> float:
    """Input level where returns switch from increasing to diminishing (only exists for c > 1)."""
    return half * ((c - 1) / (c + 1)) ** (1 / c) if c > 1 else 0.0
