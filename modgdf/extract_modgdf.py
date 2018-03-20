"""Module for extracting phase features
"""

import argparse

import numpy as np
from scipy.fftpack import dct
import scipy.io.wavfile as wav
from python_speech_features.sigproc import preemphasis, framesig

from plot import plot_data

NFFT = 512
PREEMPH = 0.97
HAMMING_WINFUNC = np.hamming
LIFTER = 6
ALPHA = 0.4
GAMMA = 0.9


def cepstrally_smoothing(spec):
    """Return cepstrally smoothed spec
    """
    _spec = np.where(spec == 0, np.finfo(float).eps, spec)
    log_spec = np.log(_spec)
    ceps = np.fft.irfft(log_spec, NFFT)
    win = (np.arange(ceps.shape[-1]) < LIFTER).astype(np.float)
    win[LIFTER] = 0.5
    return np.absolute(np.fft.rfft(ceps * win, NFFT))


def get_complex_spec(wav_, winstep, winlen, with_time_scaled=False):
    """Return complex spec
    """
    rate, sig = wav.read(wav_)

    sig = preemphasis(sig, PREEMPH)
    frames = framesig(sig, winlen * rate, winstep * rate, HAMMING_WINFUNC)
    complex_spec = np.fft.rfft(frames, NFFT)

    time_scaled_complex_spec = None
    if with_time_scaled:
        time_scaled_frames = np.arange(frames.shape[-1]) * frames
        time_scaled_complex_spec = np.fft.rfft(time_scaled_frames, NFFT)

    return complex_spec, time_scaled_complex_spec


def get_mag_spec(complex_spec):
    """Return mag spec
    """
    return np.absolute(complex_spec)


def get_phase_spec(complex_spec):
    """Return phase spec
    """
    return np.angle(complex_spec)


def get_real_spec(complex_spec):
    """Return real spec
    """
    return np.real(complex_spec)


def get_imag_spec(complex_spec):
    """Return imag spec
    """
    return np.imag(complex_spec)


def get_modgdf(complex_spec, complex_spec_time_scaled):
    """Get Modified Group-Delay Feature
    """
    mag_spec = get_mag_spec(complex_spec)
    cepstrally_smoothed_mag_spec = cepstrally_smoothing(mag_spec)
    plot_data(cepstrally_smoothed_mag_spec,
              "cepstrally_smoothed_mag_spec.png",
              "cepstrally_smoothed_mag_spec")

    real_spec = get_real_spec(complex_spec)
    imag_spec = get_imag_spec(complex_spec)
    real_spec_time_scaled = get_real_spec(complex_spec_time_scaled)
    imag_spec_time_scaled = get_imag_spec(complex_spec_time_scaled)

    __divided = real_spec * real_spec_time_scaled \
            + imag_spec * imag_spec_time_scaled
    __tao = __divided / (cepstrally_smoothed_mag_spec ** (2. * GAMMA))
    __abs_tao = np.absolute(__tao)
    __sign = 2. * (__tao == __abs_tao).astype(np.float) - 1.
    return dct(__sign * (__abs_tao ** ALPHA), type=2, axis=1, norm='ortho')


def main():
    """Main
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default="LDC93S1.wav")
    parser.add_argument("--winstep", type=float, default=0.01)
    parser.add_argument("--winlen", type=float, default=0.025)
    parser.add_argument("--debug", type=float, default=True)

    args = parser.parse_args()
    complex_spec, complex_spec_time_scaled = get_complex_spec(
        args.wav, args.winstep,
        args.winlen, with_time_scaled=True)

    if args.debug:
        mag_spec = get_mag_spec(complex_spec)
        phase_spec = get_phase_spec(complex_spec)
        mag_spec_time_scaled = get_mag_spec(complex_spec_time_scaled)
        phase_spec_time_scaled = get_phase_spec(complex_spec_time_scaled)

        plot_data(mag_spec, "mag.png", "mag")
        plot_data(phase_spec, "orig_phase.png", "phase")
        plot_data(mag_spec_time_scaled, "mag_spec_time_scaled.png", "mag_spec_time_scaled")
        plot_data(phase_spec_time_scaled, "phase_spec_time_scaled.png", "phase_spec_time_scaled")

    modgdf = get_modgdf(complex_spec, complex_spec_time_scaled)
    plot_data(modgdf, "modgdf.png", "modgdf")
    plot_data(np.absolute(modgdf), "abs_modgdf.png", "abs_modgdf")

if __name__ == "__main__":
    main()
