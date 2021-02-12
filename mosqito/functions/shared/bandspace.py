# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:01:07 2021

@author: pc
"""

# Standard library imports
import numpy as np
import math

# Local import
from mosqito.functions.shared.conversion import freq2erb, freq2bark

def bandspace(scale,fmin, fmax, bw, hitme=[]):
    """
    computes a vector containing values scaled between frequencies fmin and fmax 
    on the selected auditory scale.  All frequencies are specified in Hz.
    The distance between two consecutive values is bw on the selected scale,
    and the points will be centered on the scale between fmin and fmax.

    Parameters
    ----------
    scale : string
        scale type 'bark', 'erb', 'Hertz'
    fmin : float
        minimum frequency.
    fmax : float
        maximum frequency.
    bw : float
        frequency step in the unit corresponding to the scale chosen.

    Returns
    -------
    freqs : numpy.array
        list of frequencies between fmin and fmax spaced by bw
    n : integer
        number of frequencies in the scale

    """
   
    if scale == 'erb':
        # Convert the frequency limits to ERB
        freq_limits = freq2erb([fmin,fmax])
            
    if scale == 'bark':
        # Convert the frequency limits to Bark
        freq_limits = freq2bark([fmin,fmax])
        
    # Frequency range
    freq_range  = freq_limits[1] - freq_limits[0]

    # Number of points, excluding final point
    n = math.floor(freq_range/bw)
            
    # calculation in order to center the points correctly between fmin and fmax.
    center = freq_range - n * bw
        
    freq_points = freq_limits[0] + np.arange(0,n) * bw + center/2
    # Add the final point
    n = n+1
  
    return freq_points, n
