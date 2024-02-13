"""
This module contains helper functions for the orbital radar simulator.
"""

import numpy as np


def db2li(x):
    """
    Conversion from dB to linear.

    Parameters
    ----------
    x : float
        Any value or array to be converted from dB to linear unit
    """
    return 10 ** (0.1 * x)


def li2db(x):
    """
    Conversion from linear to dB.

    Parameters
    ----------
    x : float
        Any value or array to be converted from linear to dB unit
    """
    return 10 * np.log10(x)
