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
        acoustic signal to be filtered.
    fs : integer
        sampling frequency.
    order : integer
        order of the filter.
    filter_type : string
        type of the filter to use : 'lowpass', 'highpass','bandpass','bandstop'
    cutoff : float
         cut-off frequency of the filter.
    cascade : integer
        to cascade the filter, it is played cascade times.


    Returns
    -------
    outsig : numpy.array
        filtered signal.

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