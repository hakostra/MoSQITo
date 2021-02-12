# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:12:05 2021

@author: wantysal
"""

import numpy as np

def bandwidth(fc, bandtype):
    """
        Calculate the critical bandwidth of the Bark or ERB band centered on fc
        
        For Bark scale, the relation is bw = bw = 25 + 75 ( 1+1.4e-06} fc**2 )**0.69
        ref [1]
        
        For ERb scale, the relation is bw = 24.7 + fc/9.265
        ref[2]
        
      References :
          
      [1] E. Zwicker and E. Terhardt. Analytical expressions for criticalband
     rate and critical bandwidth as a function of frequency. The Journal of
     the Acoustical Society of America, 68(5):1523--1525, 1980.
        
      [2] B. R. Glasberg and B. Moore. Derivation of auditory filter shapes from
     notched-noise data. Hearing Research, 47(1-2):103, 1990.
    
    Parameters
    ----------
    fc : list on float
        center frequencies of the band to evaluate
    type : string
        'bark' or 'erb'
        
    Returns
    -------
    bw : float
        bandwidth of the band centered on fc
        
        
    """
    
    # Check the inputs
    if len(fc) > 1:
        fc = np.array(fc)        
        if fc.any() < 0:
            raise ValueError('Center frequencies must be > 0')
    else:
        if fc < 0 :
            raise ValueError('Center frequency must be > 0')

    if bandtype == 'erb':
        bw = 24.7 + fc/9.265
    
    if bandtype == 'bark':
        bw = 25 + 75  *(1 + 1.4e-06*fc**2)**0.69
        
    return bw

