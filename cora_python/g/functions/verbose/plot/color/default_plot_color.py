"""
default_plot_color - returns next color according to the colororder of the current axis

This function provides the next color in the matplotlib color cycle.
Uses a per-axes counter so that repeated calls (e.g. I1.plot(); I2.plot(); I3.plot())
advance through the cycle even when options are read before patches are added.

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       24-March-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# Per-axes index for the default color cycle (so each axes gets its own sequence)
_axes_color_index: dict = {}
# Fallback counter when axes lookup fails (so we still cycle colors)
_fallback_color_index: int = 0

# Default matplotlib-like color cycle (hex) when rcParams/axes API unavailable
_DEFAULT_COLOR_ORDER = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def _get_color_order():
    """Get the default color cycle (works with matplotlib 3.8+ where prop_cycler is internal)."""
    try:
        # Matplotlib 3.8+: axes._get_lines.prop_cycler is no longer public
        cycle = plt.rcParams.get('axes.prop_cycle')
        if cycle is not None:
            by_key = cycle.by_key()
            if 'color' in by_key:
                return list(by_key['color'])
    except Exception:
        pass
    return _DEFAULT_COLOR_ORDER


def _color_to_rgb(color) -> np.ndarray:
    """Convert a single color (hex string, name, or array) to RGB array."""
    if isinstance(color, str):
        if color.startswith('#'):
            color = color[1:]
            return np.array([int(color[i:i+2], 16)/255.0 for i in (0, 2, 4)])
        from matplotlib.colors import to_rgb
        return np.array(to_rgb(color))
    return np.asarray(color)


def default_plot_color() -> np.ndarray:
    """
    Returns the next color in the colororder of the current axis.
    Repeated calls advance through the default color cycle (lines, patches,
    collections). Uses a per-axes counter so sequencing is correct when
    options are read before artists are added to the axes.
    Returns:
        RGB color triple as numpy array
    """
    try:
        ax = plt.gca()
        ax_id = id(ax)
        color_order = _get_color_order()

        # Per-axes counter so each call gets the next color (works before patches are added)
        if ax_id not in _axes_color_index:
            n_elems = len(ax.lines) + len(ax.patches) + len(ax.collections)
            _axes_color_index[ax_id] = n_elems
        color_index = _axes_color_index[ax_id]
        _axes_color_index[ax_id] = color_index + 1

        color = color_order[color_index % len(color_order)]
        return _color_to_rgb(color)
    except Exception:
        # Fallback: advance module counter so we don't return blue three times
        global _fallback_color_index
        idx = _fallback_color_index
        _fallback_color_index = idx + 1
        color = _DEFAULT_COLOR_ORDER[idx % len(_DEFAULT_COLOR_ORDER)]
        return _color_to_rgb(color)
