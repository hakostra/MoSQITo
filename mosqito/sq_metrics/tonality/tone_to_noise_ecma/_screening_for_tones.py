# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:23:01 2020

@author: wantysal
"""
# Standard library imports
import numpy as np

# Mosqito functions import
from mosqito.sq_metrics.tonality.tone_to_noise_ecma._spectrum_smoothing import _spectrum_smoothing
from mosqito.sq_metrics.tonality.tone_to_noise_ecma._LTH import _LTH
from mosqito.sq_metrics.tonality.tone_to_noise_ecma._critical_band import _critical_band


def _screening_for_tones(freqs, spec_db, method, low_freq, high_freq):
    """
        Screening function to find the tonal candidates in a spectrum

        The 'smoothed' method is the one described by Bray W and Caspary G in :
        Automating prominent tone evaluations and accounting for time-varying
        conditions, Sound Quality Symposium, SQS 2008, Detroit, 2008.

        The 'not-smoothed' method is the one used by Aures and Terhardt

        The criteria of tonal width comes from Wade Bray in 'Methods for automating
        prominent tone evaluation and for considering variations with time or other
        reference quantities'

    Parameters
    ----------
    freqs : numpy.array
        frequency axis (n blocks x frequency axis)
    spec_db : numpy.array
        spectrum in dB (n block x spectrum)
    method : string
        the method chosen to find the tones 'Sottek'
    low_freq : float
        lowest frequency of interest
    high_freq : float
        highest frequency of interest


    Returns
    -------
    tones : list
        list of index corresponding to the potential tonal components

    """

    ###############################################################################
    # Detection of the tonal candidates according to their level

    # Creation of the smoothed spectrum
    smooth_spec = _spectrum_smoothing(freqs, spec_db.T, 24, low_freq, high_freq, freqs).T
    
    
    n = spec_db.shape[0]
    if len(spec_db.shape)>1:
        m = spec_db.shape[1] 
        stop = np.arange(1,n+1) * m -1
        
    else:
        m = spec_db.shape[0]
        n = 1
        stop = [m]
    
    
    smooth_spec = smooth_spec.ravel()
    spec_db = spec_db.ravel()
    freqs = freqs.ravel()
 
    if method == "smoothed":

        # Criteria 1 : the level of the spectral line is higher than the level of
        # the two neighboring lines
        maxima = (np.diff(np.sign(np.diff(spec_db))) < 0).nonzero()[0] + 1

        # Criteria 2 : the level of the spectral line exceeds the corresponding lines
        # of the 1/24 octave smoothed spectrum by at least 6 dB
        indexx = np.where(spec_db[maxima] > smooth_spec[maxima] + 6)[0]

        # Criteria 3 : the level of the spectral line exceeds the threshold of hearing
        threshold = _LTH(freqs)
        audible = np.where(spec_db[maxima][indexx] > threshold[maxima][indexx] + 10)[0]

        index = np.arange(0, len(spec_db))[maxima][indexx][audible]

    if method == "not-smoothed":
        # Criteria 1 : the level of the spectral line is higher than the level of
        # the two neighboring lines
        maxima = (
            np.diff(np.sign(np.diff(spec_db[3 : len(spec_db) - 3]))) < 0
        ).nonzero()[
            0
        ] + 1  # local max

        # Criteria 2 : the level of the spectral line is at least 7 dB higher than its
        # +/- 2,3 neighbors
        indexx = np.where(
            (spec_db[maxima] > (spec_db[maxima + 2] + 7))
            & (spec_db[maxima] > (spec_db[maxima - 2] + 7))
            & (spec_db[maxima] > (spec_db[maxima + 3] + 7))
            & (spec_db[maxima] > (spec_db[maxima - 3] + 7))
        )[0]

        # Criteria 3 : the level of the spectral line exceeds the threshold of hearing
        threshold = _LTH(freqs)
        audible = np.where(spec_db[maxima][indexx] > threshold[maxima][indexx] + 10)[0]

        index = np.arange(0, len(spec_db))[maxima][indexx][audible]

    ###############################################################################
    # Check of the tones width : a candidate is discarded if its width is greater
    # than half the critical bandwidth

    if n == 1:
        tones = []
    else:   
        tones = [[]for i in range(n)]
        
    # Each candidate is evaluated
    while len(index) > 0:
        # Index of the candidate
        peak_index = index[0]
        
        for i in range(n):
            if (peak_index<stop[i]) & (peak_index>(stop[i]-m)):
                block = i
                

        # Lower and higher limits of the tone width
        low_limit = peak_index
        high_limit = peak_index

        # Screen the right points of the peak
        temp = peak_index + 1

        # As long as the level decreases or remains above the smoothed spectrum,
        while (spec_db[temp] > smooth_spec[temp] + 6) and (temp + 1 < (block+1)*m):
            # if a highest spectral line is found, it becomes the candidate
            if spec_db[temp] > spec_db[peak_index]:
                peak_index = temp
            high_limit += 1
            temp += 1

        # Screen the left points of the peak
        temp = peak_index - 1
        # As long as the level decreases,
        while (spec_db[temp] > smooth_spec[temp] + 6) and (temp +1 > (block)*m):
            # if a highest spectral line is found, it becomes the candidate
            if spec_db[temp] > spec_db[peak_index]:
                peak_index = temp
            low_limit -= 1
            temp -= 1

        # Critical bandwidth
        f1, f2 = _critical_band(freqs[peak_index])
        cb_width = f2 - f1

        # Tonal width
        t_width = freqs[high_limit] - freqs[low_limit]

        if t_width < cb_width:
            if n>1:
                tones[block] = np.append(tones[block], peak_index - block*m)
            else:
                tones = np.append(tones, peak_index )

        # All the candidates already screened are deleted from the list
        sup = np.where(index <= high_limit)[0]
        index = np.delete(index, sup)
    
    tones = np.asarray(tones, dtype=object)


    return tones
