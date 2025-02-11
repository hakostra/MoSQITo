# -*- coding: utf-8 -*-

# Optional package import
try:
    import pytest
except ImportError:
    raise RuntimeError(
        "In order to perform the tests you need the 'pytest' package."
    )

# External import
import numpy as np

# Local application imports
from mosqito.utils import load
from mosqito.sound_level_meter.spectrum import spectrum
from mosqito.sq_metrics import tnr_ecma_freq

@pytest.mark.tnr_freq  # to skip or run TNR test
def test_tnr_ecma_freq():
    """Test function for the prominence ratio calculation of an audio signal

    Validation function for the "tnr_ecma_freq" function with complex spectrum array
    as input. The input signal was generated using audacity, and then the spectrum is computed using mosqito.

    Parameters
    ----------
    None

    Outputs
    -------
    None
    """
    # Test signal as input for prominence ratio calculation
    # signals generated using audacity : white noise + tones at 200 and 2000 Hz


    signal =     {
            "is_stationary": True,
            "data_file": "tests/input/white_noise_442_1768_Hz_stationary.wav"
        }
    

    # Load signal
    audio, fs = load(signal["data_file"], wav_calib=0.01)
    # convert to frequency domain
    spec, freqs = spectrum(audio, fs, window='hanning', db=False)

    # 1D input
    # Compute tone-to-noise ratio
    t_tnr, tnr, promi, freq = tnr_ecma_freq(spec, freqs=freqs, prominence=False )

    # 2D spectrum, 1D freq axis
    spec = np.tile(spec, (4,1)).T
    
    t_tnr, tnr, promi, freq = tnr_ecma_freq(spec, freqs=freqs, prominence=False )

    # 2D spectrum, 2D freq axis
    freqs = np.tile(freqs,(4,1)).T
    t_tnr, tnr, promi, freq = tnr_ecma_freq(spec, freqs=freqs, prominence=False )


if __name__ == "__main__":
    test_tnr_ecma_freq()
