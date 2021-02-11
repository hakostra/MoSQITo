# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:02:02 2021

@author: wantysal
"""

# Standard library imports
import numpy as np
from scipy.signal import lfilter

def outmidear_filter(signal,fs,ref):
    """     
        Simulates the outer- and middle ear transfer function
        
    References
    ----------
    [0] J. Breebaart, S. van de Par, and A. Kohlrausch. Binaural processing 
    model based on contralateral inhibition. I. Model structure. J. 
    Acoust. Soc. Am., August 2001.
    [1] M. Pflueger, R. Hoeldrich, W. Riedler (1998): A nonlinear model of 
    the peripheral auditory system

    Parameters
    ----------
    in : numpy.array
        input acoustic signal.
    fs : integer
        sampling frequency.
    ref : string
        'breebaart' ref[0] / 'pflueger'ref[1]

    Returns
    -------
    filtered signal

    """
    
    if ref == 'breebaart':

        # Coefficients
        q = 2 - np.cos(2*np.pi*4000/fs) - np.sqrt((np.cos(2*np.pi*4000/fs)-2)**2-1)
        r = 2 - np.cos(2*np.pi*1000/fs) - np.sqrt((np.cos(2*np.pi*1000/fs)-2)**2-1)
        
        # Initialization
        n=len(signal) + 2
        y = np.zeros((n))  
        
        # Define y and x
        x = np.zeros((n))    
        x[2:n] = signal    
        
        # Calculation
        for m in range(2,n):
            y[m]=(1-q)*r*x[m] - (1-q)*r*x[m-1] + (q+r)*y[m-1] - q*r*y[m-2]        
        outsig = y[2:]
    
    if ref== 'pflueger':

        # Highpass component
        b = [0.109, 0.109]
        a= [1, -2.5359, 3.9295, -4.7532, 4.7251, -3.5548, 2.139, -0.9879, 0.2836]
        
        # Lowpass component
        d = [1, -2, 1]
        c = [1, -2*0.989, 0.989**2]
        
        # Calculation
        outsig = lfilter(np.convolve(b, d), np.convolve(a, c), signal)

    
    return outsig