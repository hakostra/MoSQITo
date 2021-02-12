# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:20:05 2021

@author: wantysal
"""

# Standard library imports
import numpy as np
from scipy.signal import butter, lfilter

def butterworth_filter(signal, fs, order,filter_type, cutoff, cascade):
    """
        Use a Buuterworth filter on the input signal with a given order,
        filter type, cut-off frequency and cascade number
        

    Parameters
    ----------
    signal : numpy.array
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    order : TYPE
        DESCRIPTION.
    filter_type : TYPE
        'lowpass', 'highpass','bandpass','bandstop'
    cutoff : TYPE
        DESCRIPTION.
    cascade : TYPE
        DESCRIPTION.


    Returns
    -------
    outsig : TYPE
        DESCRIPTION.

    """
    
    #  The cutoff frequency is normalized from 0 to 1 where 1 is the Nyquist frequency
    [b,a] = butter(order,cutoff*(2/fs),filter_type)
    

    # Apply the filter 'cascade' times
    for i in range(0,cascade):
        outsig = lfilter(b, a, signal)

            
     # Correction for complex-valued transfer function filters
    if np.iscomplex(outsig.all()):
         outsig = 2 * np.real(outsig)
    
    return outsig