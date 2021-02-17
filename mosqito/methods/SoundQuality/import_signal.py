# -*- coding: utf-8 -*-

from mosqito.functions.shared.load import load
from SciDataTool import DataTime, DataLinspace


def import_signal(self, is_stationary, file, calib=1, mat_signal="", mat_fs=""):
    """Method to load the signal from a .wav .mat or .uff file

    Parameters
    ----------
    self : SoundQuality object
        Object frol the SoundQuality class
    is_stationary : boolean
        TRUE if the signal is stationary, FALSE if it is time-varying
    file : string
        string path to the signal file
    calib : float
        calibration factor for the signal to be in [pa]
    mat_signal : string
        in case of a .mat file, name of the signal variable
    mat_fs : string
        in case of a .mat file, name of the sampling frequency variable


    Outputs
    -------
    signal : numpy.array
        time signal values
    fs : integer
        sampling frequency

    """

    values, fs = load(
        is_stationary,
        file,
        calib=calib,
        mat_signal=mat_signal,
        mat_fs=mat_fs,
    )

    time_axis = DataLinspace(
        name="time",
        unit="s",
        initial=0,
        final=(len(values) - 1) / fs,
        number=len(values),
        include_endpoint=True,
    )

    self.fs = fs
    self.is_stationary = is_stationary
    self.signal = DataTime(
        name="Audio signal",
        symbol="x",
        unit="Pa",
        axes=[time_axis],
        values=values,
    )
