#!/usr/bin/env python3
"""
BB60D Device Interface for FM Reception

Based on Signal Hound Python API bindings.
Modified for Linux and focused on FM demodulation functionality.
"""

from ctypes import *
import numpy as np
import os

# Find the library - check common locations
LIB_PATHS = [
    "libbb_api.so.5",  # System-installed (via ldconfig)
    "libbb_api.so",
    "/usr/local/lib/libbb_api.so.5",
    "/usr/local/lib/libbb_api.so",
    "/usr/lib/libbb_api.so",
    "./signal_hound_sdk/device_apis/bb_series/lib/linux_x64/Ubuntu 18.04/libbb_api.so.5.0.9",
]

bblib = None
for path in LIB_PATHS:
    try:
        bblib = CDLL(path)
        break
    except OSError:
        continue

if bblib is None:
    raise RuntimeError(
        "Could not load BB60D library. Please install the libraries:\n"
        "  sudo cp signal_hound_sdk/device_apis/bb_series/lib/linux_x64/Ubuntu\\ 18.04/libbb_api.so.5.0.9 /usr/local/lib/libbb_api.so\n"
        "  sudo cp signal_hound_sdk/device_apis/bb_series/lib/linux_x64/Ubuntu\\ 18.04/libftd2xx.so /usr/local/lib/\n"
        "  sudo ldconfig"
    )

# Constants
BB_TRUE = 1
BB_FALSE = 0

# Device types
BB_DEVICE_NONE = 0
BB_DEVICE_BB60A = 1
BB_DEVICE_BB60C = 2
BB_DEVICE_BB60D = 3

BB_MAX_DEVICES = 8

# Frequency range (Hz)
BB_MIN_FREQ = 9.0e3
BB_MAX_FREQ = 6.4e9

# Gain/Atten settings
BB_AUTO_ATTEN = -1
BB_AUTO_GAIN = -1

# Initiate modes
BB_IDLE = -1
BB_SWEEPING = 0
BB_REAL_TIME = 1
BB_STREAMING = 4
# BB_AUDIO_DEMOD = 7  # Removed - using software FM demodulation instead

# Demodulation types (removed - using software FM demodulation instead)
# BB_DEMOD_AM = 0
# BB_DEMOD_FM = 1
# BB_DEMOD_USB = 2
# BB_DEMOD_LSB = 3
# BB_DEMOD_CW = 4

# Streaming flags
BB_STREAM_IQ = 0x0

# API function mappings
bbOpenDevice = bblib.bbOpenDevice
bbCloseDevice = bblib.bbCloseDevice
bbGetSerialNumber = bblib.bbGetSerialNumber
bbGetDeviceType = bblib.bbGetDeviceType
bbGetFirmwareVersion = bblib.bbGetFirmwareVersion
bbGetDeviceDiagnostics = bblib.bbGetDeviceDiagnostics

bbConfigureRefLevel = bblib.bbConfigureRefLevel
bbConfigureGainAtten = bblib.bbConfigureGainAtten
bbConfigureIQCenter = bblib.bbConfigureIQCenter
bbConfigureIQ = bblib.bbConfigureIQ
# bbConfigureDemod = bblib.bbConfigureDemod  # Removed - using software FM demodulation

bbInitiate = bblib.bbInitiate
bbAbort = bblib.bbAbort

bbQueryIQParameters = bblib.bbQueryIQParameters
bbGetIQUnpacked = bblib.bbGetIQUnpacked
bbGetIQUnpacked.argtypes = [
    c_int,
    np.ctypeslib.ndpointer(np.complex64, ndim=1, flags='C'),
    c_int,
    POINTER(c_int),
    c_int,
    c_int,
    POINTER(c_int),
    POINTER(c_int),
    POINTER(c_int),
    POINTER(c_int)
]

# bbFetchAudio removed - using software FM demodulation instead
# bbFetchAudio = bblib.bbFetchAudio
# bbFetchAudio.argtypes = [
#     c_int,
#     np.ctypeslib.ndpointer(np.float32, ndim=1, flags='C')
# ]

bbGetAPIVersion = bblib.bbGetAPIVersion
bbGetAPIVersion.restype = c_char_p
bbGetErrorString = bblib.bbGetErrorString
bbGetErrorString.restype = c_char_p


def check_status(status, func_name):
    """Check API status and raise exception on error."""
    if status < 0:
        error_msg = bbGetErrorString(status).decode('utf-8')
        raise RuntimeError(f"BB60D Error in {func_name}: {error_msg} (code {status})")
    elif status > 0:
        error_msg = bbGetErrorString(status).decode('utf-8')
        print(f"BB60D Warning in {func_name}: {error_msg} (code {status})")


class BB60D:
    """
    BB60D spectrum analyzer interface for FM reception.

    Supports both hardware FM demodulation and IQ streaming.
    """

    # FM broadcast band limits (North America)
    FM_MIN_FREQ = 88.0e6
    FM_MAX_FREQ = 108.0e6
    FM_STEP = 100e3  # 100 kHz step

    # Full device frequency range (BB60D supports 9 kHz to 6.4 GHz)
    MIN_FREQ = BB_MIN_FREQ  # 9 kHz
    MAX_FREQ = BB_MAX_FREQ  # 6.4 GHz

    # Default frequency
    DEFAULT_FREQ = 89.9e6

    # Hardware audio demod constants removed - using software FM demodulation
    # AUDIO_SAMPLE_RATE = 32000  # 32 kHz
    # AUDIO_BLOCK_SIZE = 4096

    def __init__(self):
        self.handle = None
        self.frequency = self.DEFAULT_FREQ
        self.streaming_mode = None
        self.total_sample_loss = 0
        self.recent_sample_loss = 0

    def open(self):
        """Open connection to BB60D device."""
        device = c_int(-1)
        status = bbOpenDevice(byref(device))
        check_status(status, "bbOpenDevice")
        self.handle = device.value

        # Get device info
        serial = c_uint32(0)
        bbGetSerialNumber(self.handle, byref(serial))

        dev_type = c_int(0)
        bbGetDeviceType(self.handle, byref(dev_type))

        version = c_int(0)
        bbGetFirmwareVersion(self.handle, byref(version))

        type_names = {
            BB_DEVICE_BB60A: "BB60A",
            BB_DEVICE_BB60C: "BB60C",
            BB_DEVICE_BB60D: "BB60D"
        }
        type_name = type_names.get(dev_type.value, f"Unknown ({dev_type.value})")

        print(f"Connected to {type_name}, Serial: {serial.value}, Firmware: {version.value}")
        return self.handle

    def close(self):
        """Close connection to device."""
        if self.handle is not None:
            bbAbort(self.handle)
            status = bbCloseDevice(self.handle)
            check_status(status, "bbCloseDevice")
            self.handle = None

    # configure_fm_demod() removed - using software FM demodulation instead
    # The hardware FM demodulator is no longer used; all FM processing
    # is done via IQ streaming + FMStereoDecoder

    def get_effective_sample_rate(self, sample_rate=250000):
        """
        Calculate the effective sample rate that would be used for a given request.

        This allows callers to know the actual rate before starting IQ streaming,
        which is useful for pre-initializing signal processing components.

        Args:
            sample_rate: Desired sample rate in Hz

        Returns:
            Actual sample rate that would be used (BB60D uses 40MHz / power-of-2)
        """
        base_rate = 40e6
        decimation = int(base_rate / sample_rate)
        # Round to nearest power of 2
        decimation = max(1, min(8192, 2 ** int(np.log2(decimation) + 0.5)))
        return base_rate / decimation

    def configure_iq_streaming(self, freq=None, sample_rate=250000):
        """
        Configure BB60D for IQ streaming (for software demodulation).

        Args:
            freq: Center frequency in Hz
            sample_rate: Desired sample rate (will be rounded to nearest available)
        """
        if freq is not None:
            self.frequency = freq

        # Clamp to full device range (not just FM band)
        self.frequency = max(self.MIN_FREQ, min(self.MAX_FREQ, self.frequency))

        # Calculate decimation factor
        # BB60D base rate is 40 MHz, decimation must be power of 2
        base_rate = 40e6
        # Find nearest power of 2 decimation
        decimation = int(base_rate / sample_rate)
        # Round to nearest power of 2
        decimation = max(1, min(8192, 2 ** int(np.log2(decimation) + 0.5)))
        actual_rate = base_rate / decimation
        # Bandwidth should be <= 0.5 * sample_rate for the API
        # Use 0.48 to maximize bandwidth while staying within Nyquist
        # FM needs at least ±75kHz for full deviation, ±60kHz minimum for RDS at 57kHz
        bandwidth = actual_rate * 0.48  # ~150 kHz at 312.5kHz sample rate

        # Configure
        # RefLevel was c_double(-20.0)
        status = bbConfigureRefLevel(self.handle, c_double(-30.0))
        check_status(status, "bbConfigureRefLevel")

        status = bbConfigureGainAtten(self.handle, BB_AUTO_GAIN, BB_AUTO_ATTEN)
        check_status(status, "bbConfigureGainAtten")

        status = bbConfigureIQCenter(self.handle, c_double(self.frequency))
        check_status(status, "bbConfigureIQCenter")

        status = bbConfigureIQ(self.handle, decimation, c_double(bandwidth))
        check_status(status, "bbConfigureIQ")

        # Initiate streaming
        status = bbInitiate(self.handle, BB_STREAMING, BB_STREAM_IQ)
        check_status(status, "bbInitiate")

        # Query actual parameters
        rate = c_double(0)
        bw = c_double(0)
        bbQueryIQParameters(self.handle, byref(rate), byref(bw))

        self.iq_sample_rate = rate.value
        self.iq_bandwidth = bw.value
        self.streaming_mode = "iq"

        print(f"IQ Streaming: {self.iq_sample_rate/1e6:.6f} MS/s ({self.iq_sample_rate:.0f} Hz), BW: {self.iq_bandwidth/1e3:.1f} kHz")

    # fetch_audio() removed - using software FM demodulation instead

    def fetch_iq(self, num_samples=8192):
        """
        Fetch IQ samples for software demodulation.

        Args:
            num_samples: Number of IQ samples to fetch

        Returns:
            numpy array of complex64 IQ samples
        """
        if self.streaming_mode != "iq":
            raise RuntimeError("Device not in IQ streaming mode")

        iq_data = np.zeros(num_samples, dtype=np.complex64)
        data_remaining = c_int(0)
        sample_loss = c_int(0)
        sec = c_int(0)
        nano = c_int(0)

        status = bbGetIQUnpacked(
            self.handle,
            iq_data,
            num_samples,
            None,  # triggers
            0,     # trigger_count
            BB_FALSE,  # purge
            byref(data_remaining),
            byref(sample_loss),
            byref(sec),
            byref(nano)
        )
        check_status(status, "bbGetIQUnpacked")

        if sample_loss.value > 0:
            grace = getattr(self, '_flush_grace', 0)
            if grace > 0:
                self._flush_grace = grace - 1
            else:
                self.total_sample_loss += sample_loss.value
                self.recent_sample_loss = sample_loss.value

        return iq_data

    def flush_iq(self):
        """Flush stale IQ data from the BB60D internal buffer.

        Call after a long init delay to discard samples that accumulated
        while the host was busy. Also resets the sample loss counters so
        startup losses aren't reported.
        """
        if self.streaming_mode != "iq":
            return

        n = 8192
        iq_data = np.zeros(n, dtype=np.complex64)
        data_remaining = c_int(0)
        sample_loss = c_int(0)
        sec = c_int(0)
        nano = c_int(0)

        # Purge call discards buffered data
        bbGetIQUnpacked(
            self.handle,
            iq_data,
            n,
            None,
            0,
            BB_TRUE,  # purge — discard buffered data
            byref(data_remaining),
            byref(sample_loss),
            byref(sec),
            byref(nano)
        )

        # Reset loss counters so init-time losses aren't visible.
        # The BB60D may continue reporting sample loss for one or more
        # fetches after a purge; _flush_grace absorbs those reports.
        self.total_sample_loss = 0
        self.recent_sample_loss = 0
        self._flush_grace = 3  # ignore sample_loss on next N fetches

    def set_frequency(self, freq):
        """
        Change the tuned frequency.

        Args:
            freq: New frequency in Hz
        """
        # Clamp to full device range (not just FM band)
        self.frequency = max(self.MIN_FREQ, min(self.MAX_FREQ, freq))

        # Reconfigure IQ streaming with new frequency
        if self.streaming_mode == "iq":
            bbAbort(self.handle)
            status = bbConfigureIQCenter(self.handle, c_double(self.frequency))
            check_status(status, "bbConfigureIQCenter")
            status = bbInitiate(self.handle, BB_STREAMING, BB_STREAM_IQ)
            check_status(status, "bbInitiate")

    def tune_up(self, step=None):
        """Increase frequency by step (default 100 kHz)."""
        step = step or self.FM_STEP
        self.set_frequency(self.frequency + step)

    def tune_down(self, step=None):
        """Decrease frequency by step (default 100 kHz)."""
        step = step or self.FM_STEP
        self.set_frequency(self.frequency - step)

    @property
    def frequency_mhz(self):
        """Current frequency in MHz."""
        return self.frequency / 1e6

    def get_diagnostics(self):
        """Get device temperature and USB status."""
        temp = c_float(0)
        usb_v = c_float(0)
        usb_i = c_float(0)
        bbGetDeviceDiagnostics(self.handle, byref(temp), byref(usb_v), byref(usb_i))
        return {
            "temperature_c": temp.value,
            "usb_voltage": usb_v.value,
            "usb_current": usb_i.value
        }

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def get_api_version():
    """Get BB API version string."""
    return bbGetAPIVersion().decode('utf-8')


if __name__ == "__main__":
    # Test basic functionality
    print(f"BB API Version: {get_api_version()}")

    with BB60D() as device:
        device.configure_iq_streaming(89.9e6)
        print(f"Tuned to {device.frequency_mhz:.1f} MHz")

        # Fetch a few IQ blocks
        for i in range(3):
            iq = device.fetch_iq(8192)
            print(f"Block {i+1}: {len(iq)} IQ samples, power: {10*np.log10(np.mean(np.abs(iq)**2)):.1f} dB")
