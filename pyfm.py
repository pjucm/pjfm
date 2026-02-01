#!/usr/bin/env python3
"""
pyfm - FM Radio Receiver for SignalHound BB60D
Copyright (c) 2026 Phil Jensen <philj@philandamy.org>
All rights reserved.

A command-line FM radio application that:
- Receives FM broadcast signals (88-108 MHz) via BB60D
- Receives NOAA Weather Radio (162.400-162.550 MHz) in NBFM mode
- Demodulates FM audio in real-time
- Plays audio through the default audio device
- Allows frequency tuning via arrow keys

Usage:
    ./pyfm.py [frequency_mhz]

Controls:
    Left/Right arrows: Tune down/up by 100 kHz (FM) or 25 kHz (Weather)
    Up/Down arrows: Volume up/down
    1-5: Recall frequency preset (FM mode)
    1-7: Recall WX channel (Weather mode)
    Shift+1-5 (!@#$%): Set preset to current frequency (FM mode)
    w: Toggle Weather radio mode (NBFM for NWS)
    r: Toggle RDS decoder (FM mode only)
    b: Toggle bass boost
    t: Toggle treble boost
    a: Toggle spectrum analyzer
    G: Toggle GPU demodulator (FM mode only)
    Q: Toggle squelch
    d: Toggle RDS diagnostics (press twice to dump to /tmp/rds_diag.txt)
    B: Toggle debug stats display
    q: Quit
"""

import sys
import threading
import argparse
import configparser
import os
import numpy as np
import time

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich import box
except ImportError:
    print("Error: rich not installed. Run: pip install rich")
    sys.exit(1)

try:
    from bb60d import BB60D, get_api_version as bb60d_get_api_version
except ImportError as e:
    BB60D = None
    bb60d_get_api_version = None

try:
    from icom_r8600 import IcomR8600, get_api_version as r8600_get_api_version
except ImportError as e:
    IcomR8600 = None
    r8600_get_api_version = None

from demodulator import FMStereoDecoder, NBFMDecoder
from gpu import GPUFMDemodulator, GPUResampler, GPUFIRBank, GPUFIRFilter
from rds_decoder import RDSDecoder, pi_to_callsign


def enable_realtime_mode():
    """
    Enable real-time scheduling for improved timing determinism.

    This helps reduce jitter in audio processing and RDS decoding by:
    1. Setting SCHED_FIFO real-time scheduling policy
    2. Locking memory to prevent paging delays
    3. Disabling Python GC during critical sections (optional)

    Requires root privileges or CAP_SYS_NICE capability.
    Run with: sudo ./pyfm.py --realtime [frequency]
    Or grant capability: sudo setcap cap_sys_nice+ep /usr/bin/python3

    Returns:
        dict with status of each operation
    """
    import ctypes

    results = {
        'sched_fifo': False,
        'mlockall': False,
        'priority': None,
        'errors': []
    }

    # Try to set SCHED_FIFO with priority 50 (range is 1-99)
    try:
        priority = 50
        param = os.sched_param(priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        results['sched_fifo'] = True
        results['priority'] = priority
    except PermissionError:
        results['errors'].append("SCHED_FIFO: Permission denied (need root or CAP_SYS_NICE)")
    except Exception as e:
        results['errors'].append(f"SCHED_FIFO: {e}")

    # Try to lock all memory (prevent paging)
    try:
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        MCL_CURRENT = 1
        MCL_FUTURE = 2
        if libc.mlockall(MCL_CURRENT | MCL_FUTURE) == 0:
            results['mlockall'] = True
        else:
            errno = ctypes.get_errno()
            if errno == 1:  # EPERM
                results['errors'].append("mlockall: Permission denied (need root or CAP_IPC_LOCK)")
            else:
                results['errors'].append(f"mlockall: Failed with errno {errno}")
    except Exception as e:
        results['errors'].append(f"mlockall: {e}")

    return results


# NOAA Weather Radio channels (NWS standard)
WX_CHANNELS = {
    1: 162.550e6,  # WX1
    2: 162.400e6,  # WX2
    3: 162.475e6,  # WX3
    4: 162.425e6,  # WX4
    5: 162.450e6,  # WX5
    6: 162.500e6,  # WX6
    7: 162.525e6,  # WX7
}


def dbm_to_s_meter(dbm):
    """
    Convert dBm to S-meter reading (VHF/UHF standard).

    VHF/UHF standard: S9 = -93 dBm, 6 dB per S-unit.

    Returns:
        tuple: (s_units, db_over_s9) where s_units is 1-9 and db_over_s9 is dB above S9
    """
    S9_DBM = -93.0  # S9 reference for VHF/UHF
    DB_PER_S = 6.0  # 6 dB per S-unit

    if dbm >= S9_DBM:
        # Above S9, report as S9+dB
        db_over = dbm - S9_DBM
        return (9, db_over)
    else:
        # Below S9, calculate S-units
        s_units = 9 + (dbm - S9_DBM) / DB_PER_S
        s_units = max(0, min(9, s_units))
        return (s_units, 0)


def format_s_meter(dbm):
    """Format S-meter reading as string."""
    s_units, db_over = dbm_to_s_meter(dbm)

    if db_over > 0:
        return f"S9+{db_over:.0f}dB"
    elif s_units >= 1:
        return f"S{s_units:.0f}"
    else:
        return "S0"


class SpectrumAnalyzer:
    """
    Audio spectrum analyzer with ModPlug-style bar rendering.

    Computes FFT on audio samples and displays 16 frequency bands
    as vertical bars with peak hold.
    """

    NUM_BANDS = 16
    # Unicode block characters for bar rendering (1/8 to 8/8)
    BLOCKS = ' ▁▂▃▄▅▆▇█'

    def __init__(self, sample_rate=48000, fft_size=2048):
        """
        Initialize spectrum analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
            fft_size: FFT size (determines frequency resolution)
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size

        # Logarithmic frequency bands from ~60Hz to ~16kHz
        # (Starting at 60Hz ensures each band spans at least one FFT bin
        # given 48kHz sample rate and 2048-point FFT = 23.4Hz resolution)
        min_freq = 60
        max_freq = 16000
        # Generate logarithmically spaced band edges
        self.band_edges = np.logspace(
            np.log10(min_freq),
            np.log10(max_freq),
            self.NUM_BANDS + 1
        )

        # Convert frequencies to FFT bin indices
        freq_per_bin = sample_rate / fft_size
        self.band_bins = (self.band_edges / freq_per_bin).astype(int)
        self.band_bins = np.clip(self.band_bins, 0, fft_size // 2)

        # Current levels (0.0 to 1.0) and peak levels
        self.levels = np.zeros(self.NUM_BANDS)
        self.peaks = np.zeros(self.NUM_BANDS)

        # Smoothing and decay parameters
        self.attack = 0.7      # How fast levels rise
        self.decay = 0.15      # How fast levels fall
        self.peak_decay = 0.02 # How fast peaks fall

        # Audio buffer for accumulating samples
        self.audio_buffer = np.zeros(fft_size)
        self.buffer_pos = 0

        # Window function for FFT
        self.window = np.hanning(fft_size)

    def update(self, audio_samples):
        """
        Update spectrum with new audio samples.

        Args:
            audio_samples: numpy array of audio samples (mono or stereo)
        """
        # Convert stereo to mono if needed
        if audio_samples.ndim == 2:
            audio_samples = audio_samples.mean(axis=1)

        # Add samples to buffer
        samples_to_add = len(audio_samples)
        if samples_to_add == 0:
            return

        # Fill buffer, wrapping if necessary
        if self.buffer_pos + samples_to_add <= self.fft_size:
            self.audio_buffer[self.buffer_pos:self.buffer_pos + samples_to_add] = audio_samples
            self.buffer_pos += samples_to_add
        else:
            # Buffer overflow - use most recent samples
            if samples_to_add >= self.fft_size:
                self.audio_buffer[:] = audio_samples[-self.fft_size:]
                self.buffer_pos = self.fft_size
            else:
                # Shift buffer and add new samples
                shift = samples_to_add
                self.audio_buffer[:-shift] = self.audio_buffer[shift:]
                self.audio_buffer[-shift:] = audio_samples
                self.buffer_pos = self.fft_size

        # Only compute FFT when we have enough samples
        if self.buffer_pos >= self.fft_size:
            self._compute_spectrum()
            self.buffer_pos = self.fft_size // 2  # 50% overlap

    def _compute_spectrum(self):
        """Compute FFT and update band levels."""
        # Apply window and compute FFT
        windowed = self.audio_buffer * self.window
        fft_result = np.fft.rfft(windowed)
        # Normalize magnitude by FFT size
        magnitude = np.abs(fft_result) / self.fft_size

        # Compute power in each band
        new_levels = np.zeros(self.NUM_BANDS)
        for i in range(self.NUM_BANDS):
            start_bin = self.band_bins[i]
            end_bin = self.band_bins[i + 1]
            if end_bin > start_bin:
                # Average power in band, converted to dB scale
                band_power = np.mean(magnitude[start_bin:end_bin] ** 2)
                if band_power > 1e-12:
                    # Convert to dB relative to full scale
                    # Use -70dB to -10dB range for typical broadcast audio
                    db = 10 * np.log10(band_power)
                    # Map -70dB to 0.0 and -10dB to 1.0
                    new_levels[i] = np.clip((db + 70) / 60, 0, 1)

        # Apply attack/decay smoothing
        for i in range(self.NUM_BANDS):
            if new_levels[i] > self.levels[i]:
                # Attack (rising)
                self.levels[i] += (new_levels[i] - self.levels[i]) * self.attack
            else:
                # Decay (falling)
                self.levels[i] += (new_levels[i] - self.levels[i]) * self.decay

            # Update peaks
            if self.levels[i] > self.peaks[i]:
                self.peaks[i] = self.levels[i]
            else:
                self.peaks[i] = max(0, self.peaks[i] - self.peak_decay)

    def render(self, height=8):
        """
        Render spectrum as rich Text with colored bars.

        Args:
            height: Height in rows

        Returns:
            List of rich Text objects, one per row (top to bottom)
        """
        # Fixed 2-char wide bars for consistent appearance
        bar_width = 2
        total_steps = height * 8  # 8 sub-steps per row using block characters

        rows = []
        for row in range(height):
            row_text = Text()
            row_from_bottom = height - 1 - row
            row_min = row_from_bottom * 8  # Minimum level for this row
            row_max = row_min + 8          # Maximum level for this row

            for band in range(self.NUM_BANDS):
                level_steps = int(self.levels[band] * total_steps)
                peak_step = int(self.peaks[band] * total_steps)

                # Determine what to draw in this cell
                if level_steps >= row_max:
                    # Full block
                    char = self.BLOCKS[8]
                    color = self._get_color(row_from_bottom, height)
                elif level_steps > row_min:
                    # Partial block
                    partial = level_steps - row_min
                    char = self.BLOCKS[partial]
                    color = self._get_color(row_from_bottom, height)
                elif peak_step >= row_min and peak_step < row_max:
                    # Peak indicator
                    char = '▔'
                    color = "bright_white"
                else:
                    # Empty
                    char = ' '
                    color = "default"

                # Add bar segment (repeated for bar width)
                row_text.append(char * bar_width, style=color)

                # Add spacing between bars (except after last)
                if band < self.NUM_BANDS - 1:
                    row_text.append(' ')

            rows.append(row_text)

        return rows

    def _get_color(self, row_from_bottom, total_height):
        """Get color for a row based on height (green->yellow->red)."""
        position = row_from_bottom / total_height
        if position < 0.33:
            return "green"
        elif position < 0.66:
            return "yellow"
        else:
            return "red"

    def reset(self):
        """Reset analyzer state."""
        self.levels = np.zeros(self.NUM_BANDS)
        self.peaks = np.zeros(self.NUM_BANDS)
        self.audio_buffer = np.zeros(self.fft_size)
        self.buffer_pos = 0


class AudioPlayer:
    """Manages audio playback via sounddevice with ring buffer. Supports mono and stereo."""

    def __init__(self, sample_rate=48000, channels=2, latency=0.1):
        """
        Initialize audio player.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            latency: Target latency in seconds
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.latency = latency

        # Ring buffer for audio samples (2D for stereo support)
        buffer_samples = int(sample_rate * latency * 4)  # 4x latency for safety
        self.buffer = np.zeros((buffer_samples, 2), dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.buffer_lock = threading.Lock()

        self.stream = None
        self.running = False
        self.volume = 1.0

        # Adaptive rate control: target buffer level for rate adjustment
        self._target_level_ms = 100

    def _audio_callback(self, outdata, frames, time_info, status):
        """Sounddevice callback for audio output."""
        with self.buffer_lock:
            buffer_len = len(self.buffer)
            available = (self.write_pos - self.read_pos) % buffer_len

            if available >= frames:
                # Enough data available
                end_pos = self.read_pos + frames
                if end_pos <= buffer_len:
                    outdata[:] = self.buffer[self.read_pos:end_pos] * self.volume
                else:
                    # Wrap around
                    first_part = buffer_len - self.read_pos
                    outdata[:first_part] = self.buffer[self.read_pos:] * self.volume
                    outdata[first_part:] = self.buffer[:frames - first_part] * self.volume
                self.read_pos = end_pos % buffer_len
            else:
                # Buffer underrun - output what we have plus silence
                if available > 0:
                    end_pos = self.read_pos + available
                    if end_pos <= buffer_len:
                        outdata[:available] = self.buffer[self.read_pos:end_pos] * self.volume
                    else:
                        first_part = buffer_len - self.read_pos
                        outdata[:first_part] = self.buffer[self.read_pos:] * self.volume
                        outdata[first_part:available] = self.buffer[:available - first_part] * self.volume
                    self.read_pos = end_pos % buffer_len
                outdata[available:] = 0  # Silence for missing samples

    def start(self):
        """Start audio playback stream."""
        self.running = True
        # Pre-fill buffer to target level (accounts for first block adding more)
        prefill = int(self.sample_rate * self._target_level_ms / 1000)
        self.write_pos = prefill
        self.read_pos = 0

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            latency=self.latency,
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        """Stop audio playback stream."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def reset(self):
        """Reset buffer to prefilled state (call after tuning)."""
        with self.buffer_lock:
            prefill = int(self.sample_rate * self._target_level_ms / 1000)
            self.buffer[:] = 0  # Clear to silence
            self.write_pos = prefill
            self.read_pos = 0

    @property
    def buffer_level_ms(self):
        """Return current buffer fill level in milliseconds."""
        with self.buffer_lock:
            available = (self.write_pos - self.read_pos) % len(self.buffer)
        return available / self.sample_rate * 1000

    @property
    def buffer_capacity_ms(self):
        """Return total buffer capacity in milliseconds."""
        return len(self.buffer) / self.sample_rate * 1000

    @property
    def rate_control_stats(self):
        """Return adaptive rate control statistics."""
        return {
            'target_ms': self._target_level_ms,
            'current_ms': self.buffer_level_ms,
        }

    def queue_audio(self, audio_data):
        """
        Add audio data to the ring buffer.

        Args:
            audio_data: numpy array of float32 audio samples
                       Shape (N,) for mono or (N, 2) for stereo
        """
        # Convert mono to stereo if needed
        if audio_data.ndim == 1:
            audio_data = np.column_stack((audio_data, audio_data))

        with self.buffer_lock:
            samples = len(audio_data)
            buffer_len = len(self.buffer)
            space = buffer_len - ((self.write_pos - self.read_pos) % buffer_len) - 1

            if samples > space:
                # Buffer overflow - drop oldest data
                samples = space

            if samples > 0:
                end_pos = self.write_pos + samples
                if end_pos <= buffer_len:
                    self.buffer[self.write_pos:end_pos] = audio_data[:samples]
                else:
                    # Wrap around
                    first_part = buffer_len - self.write_pos
                    self.buffer[self.write_pos:] = audio_data[:first_part]
                    self.buffer[:samples - first_part] = audio_data[first_part:samples]
                self.write_pos = end_pos % buffer_len

    def set_volume(self, volume):
        """Set playback volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))


class FMRadio:
    """
    FM Radio application controller.

    Uses IQ streaming with software FM demodulation.
    Supports BB60D and IC-R8600 as I/Q sources.
    """

    # IQ streaming parameters
    IQ_SAMPLE_RATE = 250000  # Results in 312.5kHz (40MHz/128) for BB60D, ~480kHz for R8600
    AUDIO_SAMPLE_RATE = 48000
    IQ_BLOCK_SIZE = 8192  # ~26.2ms budget at 312.5kHz

    # Signal level calibration offset (dB)
    # IQ samples need calibration to match true power in dBm.
    # These offsets were determined empirically by comparing to calibrated
    # readings. The R8600 offset accounts for its different I/Q output level.
    SIGNAL_CAL_OFFSET_DB_BB60D = -8.0
    SIGNAL_CAL_OFFSET_DB_R8600 = -23.0  # Base offset before IQ gain compensation

    # Config file path (in same directory as script)
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyfm.cfg')

    def __init__(self, initial_freq=89.9e6, use_icom=False):
        """
        Initialize FM Radio.

        Args:
            initial_freq: Initial frequency in Hz
            use_icom: If True, use IC-R8600 instead of BB60D
        """
        self.use_icom = use_icom

        if use_icom:
            if IcomR8600 is None:
                raise RuntimeError("IC-R8600 support not available. Check icom_r8600.py and pyusb installation.")
            self.device = IcomR8600()
        else:
            if BB60D is None:
                raise RuntimeError("BB60D support not available. Check bb60d.py and BB API installation.")
            self.device = BB60D()

        self.device.frequency = initial_freq

        # Audio player at 48 kHz with stereo support
        self.audio_player = AudioPlayer(
            sample_rate=self.AUDIO_SAMPLE_RATE,
            channels=2,
            latency=0.05  # 50ms driver latency; rate control handles drift
        )

        # Stereo FM decoder (handles both stereo and mono signals)
        self.stereo_decoder = None

        # RDS decoder (processed inline in audio thread for sample continuity)
        self.rds_decoder = None
        self.rds_enabled = False
        self.rds_data = {}

        # Auto stereo mode (enabled by default)
        self.auto_mode_enabled = True

        # Debug displays (hidden by default)
        self.show_buffer_stats = False

        # Spectrum analyzer
        self.spectrum_analyzer = SpectrumAnalyzer(
            sample_rate=self.AUDIO_SAMPLE_RATE,
            fft_size=2048
        )
        self.spectrum_enabled = False
        self.spectrum_box_enabled = True  # Show box around spectrum (not exposed in UI yet)

        # Squelch
        self.squelch_enabled = True
        self.squelch_threshold = -95.0  # dBm

        self.running = False
        self.audio_thread = None
        self.error_message = None
        self.signal_dbm = -140.0
        self.signal_lock = threading.Lock()
        self.tuning_lock = threading.Lock()
        self.is_tuning = False

        # Frequency presets (1-5), initialized to None
        self.presets = [None, None, None, None, None]

        # Tone control settings (applied when stereo decoder is created)
        self._initial_bass_boost = True
        self._initial_treble_boost = True

        # GPU demodulator, resampler, and FIR filters (off by default)
        self.gpu_demod = None
        self.gpu_resampler = None
        self.gpu_fir_bank = None
        self.gpu_fir_lr_diff = None
        self.gpu_enabled = True

        # Weather radio mode (NBFM for NWS)
        self.weather_mode = False
        self.nbfm_decoder = None

        # Load saved config (presets and last frequency)
        self._load_config()

    def _load_config(self):
        """Load presets and tone settings from config file (frequency is handled by main())."""
        if not os.path.exists(self.CONFIG_FILE):
            return

        config = configparser.ConfigParser()
        try:
            config.read(self.CONFIG_FILE)

            # Load presets
            for i in range(1, 6):
                key = str(i)
                if config.has_option('presets', key):
                    value = config.get('presets', key).strip()
                    if value:
                        freq_mhz = float(value)
                        if 88.0 <= freq_mhz <= 108.0:
                            self.presets[i - 1] = freq_mhz * 1e6

            # Load tone settings
            if config.has_option('tone', 'bass_boost'):
                self._initial_bass_boost = config.getboolean('tone', 'bass_boost')
            if config.has_option('tone', 'treble_boost'):
                self._initial_treble_boost = config.getboolean('tone', 'treble_boost')
        except (ValueError, configparser.Error):
            # Ignore invalid config
            pass

    def _save_config(self):
        """Save presets, tone settings, and current frequency to config file."""
        config = configparser.ConfigParser()

        # Radio section
        config['radio'] = {
            'last_frequency': f'{self.device.frequency / 1e6:.1f}',
            'device': 'icom' if self.use_icom else 'bb60d'
        }

        # Presets section
        config['presets'] = {}
        for i, freq in enumerate(self.presets, 1):
            if freq is not None:
                config['presets'][str(i)] = f'{freq / 1e6:.1f}'
            else:
                config['presets'][str(i)] = ''

        # Tone section
        config['tone'] = {
            'bass_boost': str(self.bass_boost_enabled).lower(),
            'treble_boost': str(self.treble_boost_enabled).lower()
        }

        try:
            with open(self.CONFIG_FILE, 'w') as f:
                config.write(f)
        except IOError:
            # Ignore write errors
            pass

    def start(self):
        """Start the radio."""
        try:
            self.device.open()

            # Configure IQ streaming first to learn the actual sample rate.
            # This is fast (~2 ms) and the BB60D starts buffering, but
            # we flush stale data before entering the audio loop.
            self.device.configure_iq_streaming(self.device.frequency, self.IQ_SAMPLE_RATE)
            actual_rate = self.device.iq_sample_rate

            # Create all decoders and GPU objects with the correct rate.
            # PyTorch/ROCm init and scipy filter design are heavyweight
            # (~2-3 s total). The BB60D buffers during this time; we flush
            # stale IQ data before the audio loop starts.
            self.gpu_demod = GPUFMDemodulator(
                sample_rate=actual_rate,
                deviation=75000
            )

            self.stereo_decoder = FMStereoDecoder(
                iq_sample_rate=actual_rate,
                audio_sample_rate=self.AUDIO_SAMPLE_RATE,
                deviation=75000,
                deemphasis=75e-6
            )
            self.stereo_decoder.bass_boost_enabled = self._initial_bass_boost
            self.stereo_decoder.treble_boost_enabled = self._initial_treble_boost

            self.rds_decoder = RDSDecoder(sample_rate=actual_rate)

            self.nbfm_decoder = NBFMDecoder(
                iq_sample_rate=actual_rate,
                audio_sample_rate=self.AUDIO_SAMPLE_RATE,
                deviation=5000
            )

            self.gpu_fir_bank = GPUFIRBank(
                [self.stereo_decoder.pilot_bpf,
                 self.stereo_decoder.lr_sum_lpf,
                 self.stereo_decoder.lr_diff_bpf],
                block_size=self.IQ_BLOCK_SIZE
            )

            self.gpu_fir_lr_diff = GPUFIRFilter(
                self.stereo_decoder.lr_diff_lpf,
                block_size=self.IQ_BLOCK_SIZE
            )

            # GPU resampler disabled - using adaptive CPU resampling for rate control
            # self.gpu_resampler = GPUResampler(
            #     up=self.stereo_decoder.resample_up,
            #     down=self.stereo_decoder.resample_down,
            #     n_input=self.IQ_BLOCK_SIZE
            # )
            self.gpu_resampler = None

            # Flush stale IQ data that accumulated during init.
            # The audio loop has startup stabilization that discards blocks
            # if the buffer fills too fast, so a simple flush here is sufficient.
            self.device.flush_iq()

            # Attach GPU components to stereo decoder (GPU enabled by default)
            if self.gpu_enabled:
                if self.gpu_demod:
                    self.stereo_decoder.gpu_demodulator = self.gpu_demod
                if self.gpu_resampler:
                    self.stereo_decoder.gpu_resampler = self.gpu_resampler
                if self.gpu_fir_bank:
                    self.stereo_decoder.gpu_fir_bank = self.gpu_fir_bank
                if self.gpu_fir_lr_diff:
                    self.stereo_decoder.gpu_fir_lr_diff = self.gpu_fir_lr_diff

            self.audio_player.start()
            self.running = True

            # Start audio processing thread (also handles RDS inline)
            self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self.audio_thread.start()

        except Exception as e:
            self.error_message = str(e)
            raise

    def stop(self):
        """Stop the radio."""
        self.running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        # Close rate control log file
        if hasattr(self, '_rate_log_file') and self._rate_log_file:
            self._rate_log_file.close()
            self._rate_log_file = None
        # Detach GPU components from stereo decoder before cleanup
        if self.stereo_decoder:
            self.stereo_decoder.gpu_demodulator = None
            self.stereo_decoder.gpu_resampler = None
            self.stereo_decoder.gpu_fir_bank = None
            self.stereo_decoder.gpu_fir_lr_diff = None
        # Release GPU/HIP resources while the runtime is still valid,
        # before Python shutdown triggers GC vs HIP teardown race
        if self.gpu_fir_bank:
            self.gpu_fir_bank.cleanup()
            self.gpu_fir_bank = None
        if self.gpu_fir_lr_diff:
            self.gpu_fir_lr_diff.cleanup()
            self.gpu_fir_lr_diff = None
        if self.gpu_resampler:
            self.gpu_resampler.cleanup()
            self.gpu_resampler = None
        if self.gpu_demod:
            self.gpu_demod.cleanup()
            self.gpu_demod = None
        self.audio_player.stop()
        self.device.close()

    def _audio_loop(self):
        """Background thread for IQ capture, demodulation, and signal measurement."""
        # Startup stabilization: discard blocks until buffer level is reasonable
        startup_blocks = 0
        max_startup_blocks = 50  # ~1 second worth

        # Rate control logging
        self._audio_loop_start = time.perf_counter()

        while self.running:
            try:
                # Skip if we're in the middle of tuning
                if self.is_tuning:
                    time.sleep(0.01)
                    continue

                with self.tuning_lock:
                    # Get IQ samples
                    iq = self.device.fetch_iq(self.IQ_BLOCK_SIZE)

                # Check again after fetch - if tuning started mid-fetch, discard samples
                if self.is_tuning:
                    continue

                # Measure signal power from IQ data (use subset for speed)
                # Only measure every 16th sample to reduce overhead
                iq_subset = iq[::16]
                mean_power = np.mean(iq_subset.real**2 + iq_subset.imag**2)
                if mean_power > 0:
                    # Apply calibration offset to convert raw IQ power to dBm
                    # Must compensate for any IQ gain applied by the device
                    if self.use_icom:
                        # R8600: compensate for IQ gain (100x = +40 dB in power)
                        iq_gain = getattr(self.device, '_iq_gain', 1.0)
                        gain_compensation = -20 * np.log10(iq_gain) if iq_gain > 0 else 0
                        dbm = 10 * np.log10(mean_power) + self.SIGNAL_CAL_OFFSET_DB_R8600 + gain_compensation
                    else:
                        # BB60D: no gain compensation needed
                        dbm = 10 * np.log10(mean_power) + self.SIGNAL_CAL_OFFSET_DB_BB60D
                else:
                    dbm = -140.0
                self.signal_dbm = dbm  # No lock needed for single float assignment

                # Check squelch
                squelched = self.squelch_enabled and dbm < self.squelch_threshold

                # Adaptive rate control: adjust resample ratio based on buffer level
                # If buffer > target, produce fewer samples; if buffer < target, produce more
                # Proportional control: 1ms error = 50ppm adjustment
                # (higher gain = less steady-state error, buffer settles closer to target)
                buf_level = self.audio_player.buffer_level_ms
                buf_target = self.audio_player._target_level_ms
                buf_error = buf_level - buf_target  # positive = buffer too full
                rate_adj = 1.0 - (buf_error * 0.00005)  # 50 ppm per ms error
                rate_adj = max(0.99, min(1.01, rate_adj))  # clamp to ±1%
                if self.weather_mode:
                    self.nbfm_decoder.rate_adjust = rate_adj
                else:
                    self.stereo_decoder.rate_adjust = rate_adj

                # Log rate control stats every ~1 second (60 blocks)
                if hasattr(self, '_rate_log_counter'):
                    self._rate_log_counter += 1
                else:
                    self._rate_log_counter = 0
                    self._rate_log_file = open('/tmp/pyfm_rate_control.log', 'w')
                    self._rate_log_file.write("time_s,buf_ms,target_ms,error_ms,adj_ppm\n")
                if self._rate_log_counter % 60 == 0:
                    elapsed = time.perf_counter() - self._audio_loop_start if hasattr(self, '_audio_loop_start') else 0
                    adj_ppm = (rate_adj - 1.0) * 1e6
                    self._rate_log_file.write(f"{elapsed:.1f},{buf_level:.1f},{buf_target:.0f},{buf_error:.1f},{adj_ppm:.1f}\n")
                    self._rate_log_file.flush()

                # Demodulate FM using appropriate decoder
                if self.weather_mode:
                    # NBFM for weather radio
                    audio = self.nbfm_decoder.demodulate(iq)
                else:
                    # Wideband FM stereo for broadcast
                    audio = self.stereo_decoder.demodulate(iq)

                # Auto mode: RDS is enabled when pilot is present (FM broadcast only)
                # (pilot = station has stereo/RDS capability)
                # Don't enable RDS on noise - require signal above squelch threshold
                if not self.weather_mode and self.auto_mode_enabled and self.stereo_decoder:
                    pilot_present = self.stereo_decoder.pilot_detected and dbm >= self.squelch_threshold
                    if pilot_present and not self.rds_enabled:
                        self.rds_enabled = True
                        # Reset decoder to clear any stale filter/timing state
                        if self.rds_decoder:
                            self.rds_decoder.reset()
                    elif not pilot_present and self.rds_enabled:
                        self.rds_enabled = False

                # Process RDS inline (no queue) for sample continuity (FM broadcast only)
                if not self.weather_mode and self.rds_enabled and self.rds_decoder and self.stereo_decoder.last_baseband is not None:
                    self.rds_data = self.rds_decoder.process(
                        self.stereo_decoder.last_baseband,
                        use_coherent=True  # Use pilot-derived carrier
                    )

                # Apply squelch (mute if signal below threshold)
                if squelched:
                    audio = np.zeros_like(audio)

                # Update spectrum analyzer
                if self.spectrum_enabled:
                    self.spectrum_analyzer.update(audio)

                # Startup stabilization: discard audio blocks if buffer is filling too fast
                # This handles the initial IQ backlog from USB/OS buffers
                if startup_blocks < max_startup_blocks:
                    startup_blocks += 1
                    if self.audio_player.buffer_level_ms > self.audio_player._target_level_ms:
                        # Buffer already has enough - don't queue this block
                        continue

                # Queue audio for playback
                self.audio_player.queue_audio(audio)

            except Exception as e:
                # Ignore errors during tuning
                if not self.is_tuning:
                    self.error_message = str(e)
                time.sleep(0.01)

    def get_signal_strength(self):
        """Get last measured signal strength in dBm."""
        return self.signal_dbm  # No lock needed for single float read

    def tune_up(self):
        """Tune up by 100 kHz (FM) or 25 kHz (Weather)."""
        self.is_tuning = True
        self.error_message = None
        with self.tuning_lock:
            if self.weather_mode:
                # Weather: 25 kHz steps within 162.400-162.550 MHz
                new_freq = self.device.frequency + 25000
                if new_freq > 162.550e6:
                    new_freq = 162.400e6  # Wrap around
                self.device.frequency = new_freq
                self.device.configure_iq_streaming(new_freq, self.IQ_SAMPLE_RATE)
                self.nbfm_decoder.reset()
            else:
                # FM broadcast: 100 kHz steps
                self.device.tune_up()
                self.stereo_decoder.reset()
                if self.rds_decoder:
                    self.rds_decoder.reset()
                    self.rds_data = {}
            self.audio_player.reset()  # Refill buffer after tuning
        self.is_tuning = False
        if not self.weather_mode:
            self._save_config()

    def tune_down(self):
        """Tune down by 100 kHz (FM) or 25 kHz (Weather)."""
        self.is_tuning = True
        self.error_message = None
        with self.tuning_lock:
            if self.weather_mode:
                # Weather: 25 kHz steps within 162.400-162.550 MHz
                new_freq = self.device.frequency - 25000
                if new_freq < 162.400e6:
                    new_freq = 162.550e6  # Wrap around
                self.device.frequency = new_freq
                self.device.configure_iq_streaming(new_freq, self.IQ_SAMPLE_RATE)
                self.nbfm_decoder.reset()
            else:
                # FM broadcast: 100 kHz steps
                self.device.tune_down()
                self.stereo_decoder.reset()
                if self.rds_decoder:
                    self.rds_decoder.reset()
                    self.rds_data = {}
            self.audio_player.reset()  # Refill buffer after tuning
        self.is_tuning = False
        if not self.weather_mode:
            self._save_config()

    def tune_to(self, freq_hz):
        """Tune to a specific frequency in Hz."""
        if self.weather_mode:
            # Weather mode: 162.400-162.550 MHz
            if not (162.400e6 <= freq_hz <= 162.550e6):
                return False
        else:
            # FM broadcast: 88-108 MHz
            if not (88.0e6 <= freq_hz <= 108.0e6):
                return False
        self.is_tuning = True
        self.error_message = None
        with self.tuning_lock:
            self.device.frequency = freq_hz
            self.device.configure_iq_streaming(freq_hz, self.IQ_SAMPLE_RATE)
            if self.weather_mode:
                self.nbfm_decoder.reset()
            else:
                self.stereo_decoder.reset()
                if self.rds_decoder:
                    self.rds_decoder.reset()
                    self.rds_data = {}
            self.audio_player.reset()  # Refill buffer after tuning
        self.is_tuning = False
        if not self.weather_mode:
            self._save_config()
        return True

    def set_preset(self, preset_num):
        """
        Set a preset (1-5) to the current frequency (FM mode only).

        Args:
            preset_num: Preset number 1-5
        """
        if self.weather_mode:
            return  # No user presets in weather mode
        if 1 <= preset_num <= 5:
            self.presets[preset_num - 1] = self.device.frequency
            self._save_config()

    def recall_preset(self, preset_num):
        """
        Recall a preset and tune to that frequency.

        In FM mode: presets 1-5 are user-defined.
        In Weather mode: presets 1-7 are fixed NWS channels.

        Args:
            preset_num: Preset number 1-5 (FM) or 1-7 (Weather)

        Returns:
            True if preset was recalled, False if preset is empty/invalid
        """
        if self.weather_mode:
            # Weather mode: fixed NWS channels 1-7
            if preset_num in WX_CHANNELS:
                return self.tune_to(WX_CHANNELS[preset_num])
            return False
        else:
            # FM mode: user presets 1-5
            if 1 <= preset_num <= 5:
                freq = self.presets[preset_num - 1]
                if freq is not None:
                    return self.tune_to(freq)
            return False

    def toggle_spectrum(self):
        """Toggle spectrum analyzer on/off."""
        self.spectrum_enabled = not self.spectrum_enabled
        if not self.spectrum_enabled:
            self.spectrum_analyzer.reset()

    def toggle_squelch(self):
        """Toggle squelch on/off."""
        self.squelch_enabled = not self.squelch_enabled

    def toggle_rds(self):
        """Toggle RDS decoding on/off."""
        self.rds_enabled = not self.rds_enabled
        if self.rds_decoder:
            self.rds_decoder.reset()
        self.rds_data = {}

    def toggle_profile(self):
        """Toggle demodulator profiling on/off."""
        if self.stereo_decoder:
            self.stereo_decoder.profile_enabled = not self.stereo_decoder.profile_enabled

    def toggle_bass_boost(self):
        """Toggle bass boost on/off."""
        if self.weather_mode and self.nbfm_decoder:
            self.nbfm_decoder.bass_boost_enabled = not self.nbfm_decoder.bass_boost_enabled
        elif self.stereo_decoder:
            self.stereo_decoder.bass_boost_enabled = not self.stereo_decoder.bass_boost_enabled
            self._save_config()

    def toggle_treble_boost(self):
        """Toggle treble boost on/off."""
        if self.weather_mode and self.nbfm_decoder:
            self.nbfm_decoder.treble_boost_enabled = not self.nbfm_decoder.treble_boost_enabled
        elif self.stereo_decoder:
            self.stereo_decoder.treble_boost_enabled = not self.stereo_decoder.treble_boost_enabled
            self._save_config()

    def toggle_buffer_stats(self):
        """Toggle buffer statistics display (hidden debug feature)."""
        self.show_buffer_stats = not self.show_buffer_stats

    def toggle_rds_diagnostics(self):
        """Toggle RDS timing diagnostics collection."""
        if self.rds_decoder:
            if self.rds_decoder._diag_enabled:
                # Dump and disable
                self.rds_decoder.dump_diagnostics()
                self.rds_decoder.enable_diagnostics(False)
                return "dumped"
            else:
                # Enable
                self.rds_decoder.enable_diagnostics(True)
                return "enabled"
        return None

    def toggle_gpu_demod(self):
        """Toggle GPU demodulator, resampler, and FIR filters on/off (FM mode only)."""
        self.gpu_enabled = not self.gpu_enabled
        if self.stereo_decoder:
            if self.gpu_enabled:
                if self.gpu_demod:
                    self.stereo_decoder.gpu_demodulator = self.gpu_demod
                    self.gpu_demod.reset()
                if self.gpu_resampler:
                    self.stereo_decoder.gpu_resampler = self.gpu_resampler
                if self.gpu_fir_bank:
                    self.stereo_decoder.gpu_fir_bank = self.gpu_fir_bank
                    self.gpu_fir_bank.reset()
                if self.gpu_fir_lr_diff:
                    self.stereo_decoder.gpu_fir_lr_diff = self.gpu_fir_lr_diff
                    self.gpu_fir_lr_diff.reset()
            else:
                self.stereo_decoder.gpu_demodulator = None
                self.stereo_decoder.gpu_resampler = None
                self.stereo_decoder.gpu_fir_bank = None
                self.stereo_decoder.gpu_fir_lr_diff = None
                # Reset CPU filter states to avoid stale state from GPU bypass
                from scipy import signal as sig
                self.stereo_decoder.pilot_bpf_state = sig.lfilter_zi(
                    self.stereo_decoder.pilot_bpf, 1.0)
                self.stereo_decoder.lr_sum_lpf_state = sig.lfilter_zi(
                    self.stereo_decoder.lr_sum_lpf, 1.0)
                self.stereo_decoder.lr_diff_bpf_state = sig.lfilter_zi(
                    self.stereo_decoder.lr_diff_bpf, 1.0)
                self.stereo_decoder.lr_diff_lpf_state = sig.lfilter_zi(
                    self.stereo_decoder.lr_diff_lpf, 1.0)

    @property
    def gpu_backend(self):
        """Return the GPU demodulator backend name, or None."""
        if self.gpu_demod:
            return self.gpu_demod.backend
        return None

    def toggle_weather_mode(self):
        """Toggle between FM broadcast and Weather radio modes."""
        self.is_tuning = True
        self.error_message = None
        self.weather_mode = not self.weather_mode

        with self.tuning_lock:
            if self.weather_mode:
                # Switch to weather mode: tune to WX1 (162.550 MHz)
                freq = WX_CHANNELS[1]
                self.device.frequency = freq
                self.device.configure_iq_streaming(freq, self.IQ_SAMPLE_RATE)
                self.nbfm_decoder.reset()
                # Disable RDS in weather mode
                self.rds_enabled = False
                self.rds_data = {}
            else:
                # Switch to FM mode: restore last FM frequency from config
                freq = 89.9e6  # Default
                config_path = self.CONFIG_FILE
                if os.path.exists(config_path):
                    config = configparser.ConfigParser()
                    try:
                        config.read(config_path)
                        if config.has_option('radio', 'last_frequency'):
                            freq_mhz = config.getfloat('radio', 'last_frequency')
                            if 88.0 <= freq_mhz <= 108.0:
                                freq = freq_mhz * 1e6
                    except (ValueError, configparser.Error):
                        pass
                self.device.frequency = freq
                self.device.configure_iq_streaming(freq, self.IQ_SAMPLE_RATE)
                self.stereo_decoder.reset()
                if self.rds_decoder:
                    self.rds_decoder.reset()

            self.audio_player.reset()

        self.is_tuning = False

    @property
    def is_squelched(self):
        """Returns True if audio is currently squelched."""
        return self.squelch_enabled and self.signal_dbm < self.squelch_threshold

    @property
    def pilot_detected(self):
        """Returns True if 19 kHz stereo pilot tone is detected (FM mode only)."""
        if self.weather_mode:
            return False  # No pilot in NBFM
        # Don't report pilot on noise - require signal above squelch threshold
        if self.signal_dbm < self.squelch_threshold:
            return False
        if self.stereo_decoder:
            return self.stereo_decoder.pilot_detected
        return False

    @property
    def snr_db(self):
        """Return the current audio SNR estimate in dB."""
        if self.weather_mode and self.nbfm_decoder:
            return self.nbfm_decoder.snr_db
        if self.stereo_decoder:
            return self.stereo_decoder.snr_db
        return 0.0

    @property
    def peak_amplitude(self):
        """Return peak audio amplitude (before limiting). >1.0 means limiter active."""
        if self.weather_mode and self.nbfm_decoder:
            return self.nbfm_decoder.peak_amplitude
        if self.stereo_decoder:
            return self.stereo_decoder.peak_amplitude
        return 0.0

    @property
    def stereo_blend_factor(self):
        """Return stereo blend factor (0=mono, 1=full stereo)."""
        if self.weather_mode:
            return 0.0  # Always mono in weather mode
        if self.stereo_decoder:
            return self.stereo_decoder.stereo_blend_factor
        return 0.0

    @property
    def frequency_mhz(self):
        """Current frequency in MHz."""
        return self.device.frequency_mhz

    @property
    def bass_boost_enabled(self):
        """Returns True if bass boost is enabled."""
        if self.weather_mode and self.nbfm_decoder:
            return self.nbfm_decoder.bass_boost_enabled
        if self.stereo_decoder:
            return self.stereo_decoder.bass_boost_enabled
        return False

    @property
    def treble_boost_enabled(self):
        """Returns True if treble boost is enabled."""
        if self.weather_mode and self.nbfm_decoder:
            return self.nbfm_decoder.treble_boost_enabled
        if self.stereo_decoder:
            return self.stereo_decoder.treble_boost_enabled
        return False

    @property
    def volume(self):
        """Current volume level (0.0 to 1.0)."""
        return self.audio_player.volume

    def volume_up(self, step=0.05):
        """Increase volume by step."""
        new_vol = min(1.0, self.audio_player.volume + step)
        self.audio_player.set_volume(new_vol)

    def volume_down(self, step=0.05):
        """Decrease volume by step."""
        new_vol = max(0.0, self.audio_player.volume - step)
        self.audio_player.set_volume(new_vol)


def render_s_meter_rich(dbm, width=30):
    """Render S-meter as rich Text with colors."""
    S9_DBM = -93.0
    S1_DBM = S9_DBM - (8 * 6)  # S1 = -141 dBm
    S9_PLUS_20 = S9_DBM + 20   # S9+20 = -73 dBm

    # Normalize to 0-1 range
    normalized = (dbm - S1_DBM) / (S9_PLUS_20 - S1_DBM)
    normalized = max(0, min(1, normalized))

    filled = int(normalized * width)
    s9_pos = int((S9_DBM - S1_DBM) / (S9_PLUS_20 - S1_DBM) * width)

    meter = Text()
    for i in range(width):
        if i < filled:
            if i < s9_pos:
                meter.append("█", style="green")
            else:
                meter.append("█", style="red bold")
        else:
            if i == s9_pos:
                meter.append("│", style="yellow")
            else:
                meter.append("░", style="dim")
    return meter


def render_band_indicator(freq_mhz, min_freq=88.0, max_freq=108.0, width=30):
    """Render band position indicator."""
    position = (freq_mhz - min_freq) / (max_freq - min_freq)
    pos = int(position * (width - 1))

    indicator = Text()
    for i in range(width):
        if i == pos:
            indicator.append("▼", style="cyan bold")
        else:
            indicator.append("─", style="dim")
    return indicator


def get_wx_channel_name(freq_hz):
    """Get the WX channel name for a frequency."""
    for ch, f in WX_CHANNELS.items():
        if abs(f - freq_hz) < 1000:  # Within 1 kHz tolerance
            return f"WX{ch}"
    return None


def build_display(radio, width=80):
    """Build the rich display panel."""
    freq = radio.frequency_mhz
    signal_dbm = radio.get_signal_strength()
    s_reading = format_s_meter(signal_dbm)

    # Create main table for aligned fields (not expanded, will be centered)
    table = Table(show_header=False, box=None, padding=(0, 1), expand=False)
    table.add_column("Label", style="cyan", width=12, justify="right")
    table.add_column("Value", style="green bold")

    # Frequency row
    freq_text = Text()
    freq_text.append(f"{freq:.3f} MHz", style="green bold")
    if radio.weather_mode:
        wx_ch = get_wx_channel_name(radio.device.frequency)
        if wx_ch:
            freq_text.append(f"  ({wx_ch})", style="cyan bold")
    table.add_row("Frequency:", freq_text)

    # Mode row
    if radio.weather_mode:
        table.add_row("Mode:", "NBFM Weather Radio (5 kHz dev)")
    else:
        table.add_row("Mode:", "Wideband FM Stereo (75 kHz dev)")

    # Volume row
    vol_pct = int(radio.volume * 100)
    vol_bars = int(radio.volume * 20)
    vol_text = Text()
    vol_text.append("█" * vol_bars, style="green")
    vol_text.append("░" * (20 - vol_bars), style="dim")
    vol_text.append(f" {vol_pct}%", style="yellow")
    table.add_row("Volume:", vol_text)

    # Signal row
    signal_text = Text()
    signal_text.append(f"{s_reading:<8}", style="green bold")
    signal_text.append(f"  ({signal_dbm:6.1f} dBm)", style="yellow")
    table.add_row("Signal:", signal_text)

    # SNR row - different thresholds for NBFM vs WBFM
    snr = radio.snr_db
    snr_text = Text()
    if radio.weather_mode:
        # NBFM voice thresholds (3 kHz audio bandwidth)
        # Voice is intelligible at much lower SNR than music
        if snr > 15:
            snr_text.append(f"{snr:.1f} dB", style="green bold")
            snr_text.append("  (Excellent)", style="green")
        elif snr > 10:
            snr_text.append(f"{snr:.1f} dB", style="green bold")
            snr_text.append("  (Good)", style="green")
        elif snr > 6:
            snr_text.append(f"{snr:.1f} dB", style="yellow bold")
            snr_text.append("  (Fair)", style="yellow")
        elif snr > 3:
            snr_text.append(f"{snr:.1f} dB", style="yellow bold")
            snr_text.append("  (Poor)", style="yellow")
        else:
            snr_text.append(f"{snr:.1f} dB", style="red bold")
            snr_text.append("  (Very Poor)", style="red")
        bw_label = "3kHz"
    else:
        # WBFM broadcast thresholds (15-53 kHz audio bandwidth)
        if snr > 40:
            snr_text.append(f"{snr:.1f} dB", style="green bold")
            snr_text.append("  (Excellent)", style="green")
        elif snr > 30:
            snr_text.append(f"{snr:.1f} dB", style="green bold")
            snr_text.append("  (Good)", style="green")
        elif snr > 20:
            snr_text.append(f"{snr:.1f} dB", style="yellow bold")
            snr_text.append("  (Fair)", style="yellow")
        elif snr > 10:
            snr_text.append(f"{snr:.1f} dB", style="yellow bold")
            snr_text.append("  (Poor)", style="yellow")
        else:
            snr_text.append(f"{snr:.1f} dB", style="red bold")
            snr_text.append("  (Very Poor)", style="red")
        bw_label = "53kHz" if radio.pilot_detected else "15kHz"
    snr_text.append(f"  [{bw_label}]", style="dim")
    table.add_row("SNR:", snr_text)

    # S-meter row
    s_meter = Text()
    s_meter.append("S: ", style="cyan")
    s_meter.append(render_s_meter_rich(signal_dbm, width=30))
    table.add_row("", s_meter)

    # S-meter scale (aligned to 30-char bar: S9 at pos 21, S9+20 at pos 30)
    # Bar: |----S1 to S9 (21 chars)----|--S9+ (9 chars)--|
    scale = Text()
    scale.append("   1    3    5    7    9      +20", style="dim")
    table.add_row("", scale)

    # Band position
    table.add_row("", "")  # Spacer
    band_text = Text()
    if radio.weather_mode:
        band_text.append("162.4", style="yellow")
        band_text.append(" ", style="")
        band_text.append(render_band_indicator(freq, min_freq=162.4, max_freq=162.55, width=30))
        band_text.append(" ", style="")
        band_text.append("162.55", style="yellow")
    else:
        band_text.append("88", style="yellow")
        band_text.append(" ", style="")
        band_text.append(render_band_indicator(freq, width=30))
        band_text.append(" ", style="")
        band_text.append("108", style="yellow")
    table.add_row("Band:", band_text)

    # Stereo status (stereo decoder always active, shows pilot detection status)
    table.add_row("", "")  # Spacer
    stereo_text = Text()
    if radio.weather_mode:
        stereo_text.append("Mono", style="cyan bold")
        stereo_text.append(" (NBFM)", style="dim")
    elif radio.pilot_detected:
        blend = radio.stereo_blend_factor
        if blend >= 0.99:
            stereo_text.append("Stereo", style="green bold")
            stereo_text.append(" (19 kHz pilot detected)", style="green")
        elif blend <= 0.01:
            stereo_text.append("Mono", style="yellow")
            stereo_text.append(" (blended - low SNR)", style="yellow")
        else:
            blend_pct = int(blend * 100)
            stereo_text.append(f"Blend {blend_pct}%", style="yellow bold")
            stereo_text.append(" (reduced stereo for noise)", style="yellow")
    else:
        stereo_text.append("Mono", style="yellow")
        stereo_text.append(" (no pilot)", style="dim")
    table.add_row("Audio:", stereo_text)

    # RDS data display (FM mode only - hidden in weather mode)
    if not radio.weather_mode:
        rds_snapshot = dict(radio.rds_data) if radio.rds_data else {}
        ps_name = rds_snapshot.get('station_name', '') if radio.rds_enabled else ''
        pty = rds_snapshot.get('program_type', '') if radio.rds_enabled else ''
        pi_hex = rds_snapshot.get('pi_hex') if radio.rds_enabled else None
        radio_text_val = rds_snapshot.get('radio_text', '') if radio.rds_enabled else ''
        clock_time = rds_snapshot.get('clock_time', '') if radio.rds_enabled else ''

        # Station line: Callsign [Genre]
        callsign = pi_to_callsign(pi_hex) if pi_hex else None
        station_info = Text()
        if callsign:
            station_info.append(callsign, style="cyan bold")
        elif pi_hex:
            station_info.append(f"PI:{pi_hex}", style="cyan")
        if pty and pty != "None":
            if callsign or pi_hex:
                station_info.append("  ", style="")
            station_info.append(f"({pty})", style="yellow")
        table.add_row("Station:", station_info)

        # Name line: PS (station branding)
        ps_text = Text(ps_name, style="green bold") if ps_name else Text()
        table.add_row("Name:", ps_text)

        # Radio text line (64 chars per RDS spec, left-justified to prevent layout shift)
        rt_display = radio_text_val[:64].ljust(64) if radio_text_val else " " * 64
        rt_text = Text(rt_display, style="green")
        table.add_row("Text:", rt_text)

        # Clock time line
        ct_text = Text(clock_time, style="magenta") if clock_time else Text()
        table.add_row("Time:", ct_text)
    else:
        rds_snapshot = {}  # For RDS status section below

    table.add_row("", "")  # Spacer

    # Spectrum analyzer status
    spectrum_text = Text()
    if radio.spectrum_enabled:
        spectrum_text.append("ON", style="green bold")
    else:
        spectrum_text.append("OFF", style="dim")
    table.add_row("Spectrum:", spectrum_text)

    # Squelch status
    squelch_text = Text()
    if radio.squelch_enabled:
        squelch_text.append(f"ON @ {radio.squelch_threshold:.0f} dBm", style="green bold")
        if radio.is_squelched:
            squelch_text.append("  [MUTED]", style="red bold")
    else:
        squelch_text.append("OFF", style="dim")
    table.add_row("Squelch:", squelch_text)

    # Tone controls status
    tone_text = Text()
    if radio.bass_boost_enabled:
        tone_text.append("Bass +3dB", style="green bold")
    else:
        tone_text.append("Bass OFF", style="dim")
    tone_text.append("  ", style="")
    if radio.treble_boost_enabled:
        tone_text.append("Treble +3dB", style="green bold")
    else:
        tone_text.append("Treble OFF", style="dim")
    table.add_row("Boost:", tone_text)

    # Demodulator status (FM mode only)
    if not radio.weather_mode:
        demod_text = Text()
        if radio.gpu_enabled:
            backend = radio.gpu_backend or "unknown"
            demod_text.append(f"GPU ({backend})", style="green bold")
            demod_text.append(" — demod + FIR", style="green")
        else:
            demod_text.append("CPU (quadrature)", style="dim")
        table.add_row("Demod:", demod_text)

    # RDS status (FM mode only)
    if not radio.weather_mode:
        rds_text = Text()
        if radio.rds_enabled:
            rds_text.append("ON", style="green bold")
            if rds_snapshot.get('synced'):
                rds_text.append("  [SYNC]", style="green")
            else:
                rds_text.append("  [SRCH]", style="yellow")
            # Show detailed stats only in debug mode
            if radio.show_buffer_stats:
                groups = rds_snapshot.get('groups_received', 0)
                block_rate = rds_snapshot.get('block_rate', 0)
                corrected = rds_snapshot.get('blocks_corrected', 0)
                sig_level = rds_snapshot.get('signal_level', 0)
                timing_range = rds_snapshot.get('timing_range', (0, 0))
                freq_offset = rds_snapshot.get('timing_freq_offset', 0)
                # freq_offset in samples/symbol - show as Hz offset from nominal 1187.5 Hz
                # effective_rate = sample_rate / (sps_nominal + freq_offset)
                # For small offsets: delta_Hz ≈ -freq_offset * symbol_rate^2 / sample_rate
                sample_rate = rds_snapshot.get('sample_rate', 250000)
                delta_hz = -freq_offset * (1187.5 ** 2) / sample_rate
                rds_text.append(f"  grp:{groups} blk:{block_rate:.0%} cor:{corrected} sig:{sig_level:.3f} tau:{timing_range[0]:.2f}/{timing_range[1]:.2f} df:{delta_hz:+.2f}Hz", style="dim")
        else:
            rds_text.append("OFF", style="dim")
        table.add_row("RDS:", rds_text)

    # Performance stats (disabled - only show sample loss if it occurs)
    sample_loss = getattr(radio.device, 'total_sample_loss', 0)
    if sample_loss > 0:
        perf_text = Text()
        perf_text.append(f"SAMPLE LOSS: {sample_loss}", style="red bold")
        table.add_row("Warning:", perf_text)

    # Error message if any
    if radio.error_message:
        table.add_row("", "")
        error_text = Text(f"⚠ {radio.error_message[:50]}", style="red bold")
        table.add_row("Error:", error_text)

    # Buffer stats (hidden debug display)
    if radio.show_buffer_stats:
        table.add_row("", "")
        buf_text = Text()
        level_ms = radio.audio_player.buffer_level_ms
        capacity_ms = radio.audio_player.buffer_capacity_ms
        fill_pct = (level_ms / capacity_ms) * 100

        # Color based on fill level (normal operation ~40%)
        if fill_pct > 30:
            style = "green"
        elif fill_pct > 15:
            style = "yellow"
        else:
            style = "red"

        buf_text.append(f"{level_ms:3.0f}ms", style=f"{style} bold")
        buf_text.append(f" / {capacity_ms:3.0f}ms", style="dim")
        buf_text.append(f"  ({fill_pct:2.0f}%)", style=style)

        # Visual bar
        bar_width = 20
        filled = int(bar_width * fill_pct / 100)
        buf_text.append("  [", style="dim")
        buf_text.append("█" * filled, style=style)
        buf_text.append("░" * (bar_width - filled), style="dim")
        buf_text.append("]", style="dim")

        table.add_row("Buffer:", buf_text)

        # Peak amplitude display
        peak_text = Text()
        peak = radio.peak_amplitude

        # Color based on amplitude (how close to limiting)
        if peak < 0.7:
            peak_style = "green"
            peak_status = "OK"
        elif peak < 1.0:
            peak_style = "yellow"
            peak_status = "Hot"
        else:
            peak_style = "red"
            peak_status = "Limiting"

        peak_text.append(f"{peak:.2f}", style=f"{peak_style} bold")
        peak_text.append(f"  ({peak_status:<12})", style=peak_style)

        # Visual meter (0 to 1.5 scale, with 1.0 marker)
        meter_width = 20
        # Map 0-1.5 to 0-20
        meter_pos = min(int(peak / 1.5 * meter_width), meter_width)
        limit_pos = int(1.0 / 1.5 * meter_width)  # Position of 1.0 threshold

        peak_text.append("  [", style="dim")
        for i in range(meter_width):
            if i < meter_pos:
                if i >= limit_pos:
                    peak_text.append("█", style="red")
                elif i >= int(0.7 / 1.5 * meter_width):
                    peak_text.append("█", style="yellow")
                else:
                    peak_text.append("█", style="green")
            elif i == limit_pos:
                peak_text.append("|", style="red")
            else:
                peak_text.append("░", style="dim")
        peak_text.append("]", style="dim")

        table.add_row("Peak:", peak_text)

        # Sample loss from BB60D
        total_loss = getattr(radio.device, 'total_sample_loss', 0)
        recent_loss = getattr(radio.device, 'recent_sample_loss', 0)
        loss_text = Text()
        if total_loss > 0:
            loss_text.append(f"{total_loss}", style="red bold")
            if recent_loss > 0:
                loss_text.append(f"  (recent: {recent_loss})", style="red")
        else:
            loss_text.append("0", style="green bold")
        table.add_row("IQ Loss:", loss_text)

        # Adaptive rate control stats
        rc_stats = radio.audio_player.rate_control_stats
        rc_text = Text()
        buf_level = rc_stats['current_ms']
        buf_target = rc_stats['target_ms']
        buf_error = buf_level - buf_target
        rate_ppm = -buf_error * 10  # 10 ppm per ms error
        rc_text.append(f"buf:{buf_level:.0f}ms", style="dim")
        rc_text.append(f"  target:{buf_target:.0f}ms", style="dim")
        # Show rate adjustment as ppm deviation from nominal
        if abs(buf_error) < 20:
            rc_text.append(f"  adj:{rate_ppm:+.0f}ppm", style="green")
        else:
            rc_text.append(f"  adj:{rate_ppm:+.0f}ppm", style="yellow")
        table.add_row("Rate Ctl:", rc_text)

        # RDS coherent demod diagnostics (when enabled)
        if not radio.weather_mode and radio.rds_decoder and radio.rds_decoder._diag_enabled:
            rds_diag = Text()
            pilot_rms = rds_snapshot.get('pilot_rms', 0)
            carrier_rms = rds_snapshot.get('carrier_rms', 0)
            bb_rms = rds_snapshot.get('baseband_rms', 0)
            symbol_snr = rds_snapshot.get('symbol_snr_db', 0)
            # Color symbol SNR based on quality
            if symbol_snr >= 15:
                snr_style = "green bold"
            elif symbol_snr >= 10:
                snr_style = "green"
            elif symbol_snr >= 6:
                snr_style = "yellow"
            else:
                snr_style = "red"
            rds_diag.append(f"pilot:{pilot_rms:.4f} ", style="dim")
            rds_diag.append(f"carrier:{carrier_rms:.3f} ", style="dim")
            rds_diag.append(f"bb:{bb_rms:.4f} ", style="dim")
            rds_diag.append(f"symSNR:", style="dim")
            rds_diag.append(f"{symbol_snr:.1f}dB", style=snr_style)
            table.add_row("RDS Coh:", rds_diag)

        # Demodulator stage profiling
        if not radio.weather_mode and radio.stereo_decoder and radio.stereo_decoder.profile_enabled:
            prof = radio.stereo_decoder.profile
            table.add_row("", "")
            prof_header = Text()
            prof_header.append("Demod Profile (µs/block, EMA)", style="yellow bold")
            table.add_row("Profile:", prof_header)

            # Sorted by cost descending for easy identification of hot spots
            stages = [
                ('fm_demod', 'FM Demod'),
                ('pilot_bpf', 'Pilot BPF'),
                ('lr_sum_lpf', 'L+R LPF'),
                ('lr_diff_bpf', 'L-R BPF'),
                ('lr_diff_lpf', 'L-R LPF'),
                ('noise_bpf', 'Noise BPF'),
                ('resample', 'Resample'),
                ('deemphasis', 'De-emph'),
                ('tone', 'Tone'),
                ('limiter', 'Limiter'),
            ]
            total_us = prof.get('total', 1.0)

            # Sort by time descending
            stage_times = [(key, label, prof.get(key, 0.0)) for key, label in stages]
            stage_times.sort(key=lambda x: x[2], reverse=True)

            for key, label, us in stage_times:
                pct = (us / total_us * 100) if total_us > 0 else 0
                stage_text = Text()
                # Bar proportional to percentage (max 20 chars)
                bar_len = int(pct / 5)  # 20 chars = 100%
                if pct > 30:
                    bar_style = "red bold"
                elif pct > 15:
                    bar_style = "yellow"
                else:
                    bar_style = "green"
                stage_text.append(f"{label:<12}", style="cyan")
                stage_text.append(f"{us:7.0f}", style=bar_style)
                stage_text.append(f" ({pct:4.1f}%) ", style="dim")
                stage_text.append("█" * bar_len, style=bar_style)
                table.add_row("", stage_text)

            # Total
            total_text = Text()
            total_text.append(f"{'Total':<12}", style="cyan bold")
            total_text.append(f"{total_us:7.0f}", style="bright_white bold")
            budget_us = 8192 / 312500 * 1e6  # ~26214 µs at 312.5 kHz
            budget_pct = total_us / budget_us * 100
            if budget_pct > 80:
                budget_style = "red bold"
            elif budget_pct > 50:
                budget_style = "yellow"
            else:
                budget_style = "green"
            total_text.append(f" ({budget_pct:.0f}% of {budget_us/1000:.1f}ms budget)", style=budget_style)
            table.add_row("", total_text)

    # Controls section
    controls = Text()
    controls.append("\n")
    controls.append("←/→ ", style="cyan bold")
    controls.append("Tune  ", style="dim")
    controls.append("↑/↓ ", style="cyan bold")
    controls.append("Vol  ", style="dim")
    controls.append("w ", style="cyan bold")
    controls.append("WX  ", style="dim")
    if not radio.weather_mode:
        controls.append("r ", style="cyan bold")
        controls.append("RDS  ", style="dim")
    controls.append("b ", style="cyan bold")
    controls.append("Bass  ", style="dim")
    controls.append("t ", style="cyan bold")
    controls.append("Treble  ", style="dim")
    controls.append("a ", style="cyan bold")
    controls.append("Spect  ", style="dim")
    controls.append("Q ", style="cyan bold")
    controls.append("Squelch  ", style="dim")
    if not radio.weather_mode:
        controls.append("G ", style="cyan bold")
        controls.append("GPU  ", style="dim")
    controls.append("q ", style="cyan bold")
    controls.append("Quit", style="dim")

    # Presets section
    presets = Text()
    if radio.weather_mode:
        # Weather mode: show WX1-WX7 presets
        presets.append("\nChannels: ", style="yellow bold")
        for ch in range(1, 8):
            wx_freq = WX_CHANNELS[ch]
            presets.append(f"{ch}", style="cyan bold")
            presets.append(":", style="dim")
            freq_mhz = wx_freq / 1e6
            # Highlight if this is the current frequency
            if abs(freq_mhz - freq) < 0.02:
                presets.append(f"WX{ch}", style="green bold")
            else:
                presets.append(f"WX{ch}", style="white")
            presets.append("  ", style="")
    else:
        # FM mode: show user presets 1-5
        presets.append("\nPresets: ", style="yellow bold")
        for i, preset_freq in enumerate(radio.presets, 1):
            presets.append(f"{i}", style="cyan bold")
            presets.append(":", style="dim")
            if preset_freq is not None:
                freq_mhz = preset_freq / 1e6
                # Highlight if this is the current frequency
                if abs(freq_mhz - freq) < 0.05:
                    presets.append(f"{freq_mhz:.1f}", style="green bold")
                else:
                    presets.append(f"{freq_mhz:.1f}", style="white")
            else:
                presets.append("---", style="dim")
            presets.append("  ", style="")
        presets.append("  (Shift+# to set)", style="dim")

    # Build final layout (centered)
    content = Table.grid(expand=True)
    content.add_row(Align.center(table))

    # Spectrum analyzer display (fixed 2-char wide bars, centered)
    if radio.spectrum_enabled:
        spectrum_rows = radio.spectrum_analyzer.render(height=6)
        spectrum_table = Table(show_header=False, box=None, padding=0, expand=False)
        spectrum_table.add_column("Spectrum")
        for row in spectrum_rows:
            spectrum_table.add_row(row)
        content.add_row(Text(""))  # Spacer
        if radio.spectrum_box_enabled:
            spectrum_panel = Panel(
                spectrum_table,
                subtitle="[dim]Spectrum[/]",
                box=box.ROUNDED,
                border_style="dim",
                padding=(0, 1),
            )
            content.add_row(Align.center(spectrum_panel))
        else:
            content.add_row(Align.center(spectrum_table))

    content.add_row(Align.center(controls))
    content.add_row(Align.center(presets))

    if radio.weather_mode:
        panel_title = "[bold yellow]🌦️ pyfm[/] [dim]- NOAA Weather Radio[/]"
        border_style = "yellow"
    else:
        device_name = "Icom IC-R8600" if radio.use_icom else "SignalHound BB60D"
        panel_title = f"[bold cyan]📻 pyfm[/] [dim]- {device_name} FM Receiver[/]"
        border_style = "cyan"

    panel = Panel(
        content,
        title=panel_title,
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 2),
    )

    return panel


def run_rich_ui(radio):
    """Run the rich-based user interface."""
    import sys
    import tty
    import termios
    import select
    import os

    console = Console()

    try:
        radio.start()
    except Exception as e:
        console.print(f"[red bold]Error starting radio:[/] {e}")
        return

    # Set up terminal for character-at-a-time input
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        with Live(build_display(radio, console.width), console=console, refresh_per_second=10, screen=True) as live:
            input_buffer = ''

            while True:
                # Update display (get current terminal width for responsive spectrum)
                live.update(build_display(radio, console.width))

                # Check if input is available using select (avoids non-blocking mode issues)
                readable, _, _ = select.select([sys.stdin], [], [], 0)
                if readable:
                    try:
                        # Use os.read() to get available bytes without blocking
                        chunk = os.read(sys.stdin.fileno(), 32)
                        if chunk:
                            input_buffer += chunk.decode('utf-8', errors='ignore')
                    except (BlockingIOError, IOError):
                        pass

                # Process the input buffer
                while input_buffer:
                    if input_buffer[0] == 'q':
                        # Quit (lowercase only)
                        input_buffer = ''
                        raise StopIteration
                    elif input_buffer[0] == 'Q':
                        # Toggle squelch (uppercase)
                        radio.toggle_squelch()
                        input_buffer = input_buffer[1:]
                        continue

                    # Check for escape sequences (arrow keys)
                    if input_buffer.startswith('\x1b[D') or input_buffer.startswith('\x1bOD'):
                        # Left arrow - tune down
                        radio.tune_down()
                        input_buffer = input_buffer[3:]
                    elif input_buffer.startswith('\x1b[C') or input_buffer.startswith('\x1bOC'):
                        # Right arrow - tune up
                        radio.tune_up()
                        input_buffer = input_buffer[3:]
                    elif input_buffer.startswith('\x1b[A') or input_buffer.startswith('\x1bOA'):
                        # Up arrow - volume up
                        radio.volume_up()
                        input_buffer = input_buffer[3:]
                    elif input_buffer.startswith('\x1b[B') or input_buffer.startswith('\x1bOB'):
                        # Down arrow - volume down
                        radio.volume_down()
                        input_buffer = input_buffer[3:]
                    elif input_buffer.startswith('\x1b[') or input_buffer.startswith('\x1bO'):
                        # Partial or other escape sequence - need more bytes or skip
                        if len(input_buffer) < 3:
                            break  # Wait for more input
                        # Skip unknown 3-byte sequence
                        input_buffer = input_buffer[3:]
                    elif input_buffer[0] == '\x1b':
                        # Lone escape or incomplete sequence
                        if len(input_buffer) == 1:
                            break  # Wait for more input
                        # Unknown escape - skip the ESC
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] in ('a', 'A'):
                        # Toggle spectrum analyzer
                        radio.toggle_spectrum()
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] in ('w', 'W'):
                        # Toggle weather mode
                        radio.toggle_weather_mode()
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] in ('r', 'R'):
                        # Toggle RDS decoder (FM mode only)
                        if not radio.weather_mode:
                            radio.toggle_rds()
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] in ('b', 'B'):
                        # Toggle bass boost
                        radio.toggle_bass_boost()
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] in ('t', 'T'):
                        # Toggle treble boost
                        radio.toggle_treble_boost()
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] == '/':
                        # Toggle buffer stats (hidden debug feature)
                        radio.toggle_buffer_stats()
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] == 'P':
                        # Toggle demod profiling (hidden debug feature)
                        radio.toggle_profile()
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] == 'G':
                        # Toggle GPU demodulator (FM mode only)
                        if not radio.weather_mode:
                            radio.toggle_gpu_demod()
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] in ('d', 'D'):
                        # Toggle RDS diagnostics (d=start, D=dump to /tmp/rds_timing_diag.txt)
                        result = radio.toggle_rds_diagnostics()
                        # Result shown via normal display update
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] in '1234567':
                        # Recall preset (1-5 for FM, 1-7 for Weather)
                        preset_num = int(input_buffer[0])
                        if radio.weather_mode:
                            # Weather mode: 1-7 are WX channels
                            radio.recall_preset(preset_num)
                        else:
                            # FM mode: 1-5 are user presets
                            if preset_num <= 5:
                                radio.recall_preset(preset_num)
                        input_buffer = input_buffer[1:]
                    elif input_buffer[0] in '!@#$%':
                        # Set preset (shift+1-5) - FM mode only
                        if not radio.weather_mode:
                            preset_map = {'!': 1, '@': 2, '#': 3, '$': 4, '%': 5}
                            preset_num = preset_map[input_buffer[0]]
                            radio.set_preset(preset_num)
                        input_buffer = input_buffer[1:]
                    else:
                        # Unknown character - skip it
                        input_buffer = input_buffer[1:]

                if not radio.running:
                    break

                time.sleep(0.05)

    except StopIteration:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        radio.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="pyfm - FM Radio Receiver for SignalHound BB60D or Icom IC-R8600"
    )
    parser.add_argument(
        "frequency",
        nargs="?",
        type=float,
        default=None,
        help="Initial frequency in MHz (default: last used or 89.9)"
    )
    parser.add_argument(
        "--icom",
        action="store_true",
        help="Use Icom IC-R8600 as I/Q source"
    )
    parser.add_argument(
        "--bb60d",
        action="store_true",
        help="Use SignalHound BB60D as I/Q source"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version info and exit"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Enable real-time scheduling (requires sudo or CAP_SYS_NICE)"
    )

    args = parser.parse_args()

    # Handle conflicting device flags
    if args.icom and args.bb60d:
        print("Error: Cannot specify both --icom and --bb60d")
        sys.exit(1)

    if args.version:
        print("pyfm - FM Radio Receiver")
        if bb60d_get_api_version:
            try:
                print(f"BB60D API Version: {bb60d_get_api_version()}")
            except Exception as e:
                print(f"BB60D: not available ({e})")
        else:
            print("BB60D: not available")
        if r8600_get_api_version:
            try:
                print(f"IC-R8600: {r8600_get_api_version()}")
            except Exception as e:
                print(f"IC-R8600: not available ({e})")
        else:
            print("IC-R8600: not available")
        return

    # Enable real-time scheduling if requested
    if args.realtime:
        rt_results = enable_realtime_mode()
        if rt_results['sched_fifo']:
            print(f"Real-time mode: SCHED_FIFO priority {rt_results['priority']}")
        if rt_results['mlockall']:
            print("Real-time mode: Memory locked")
        for err in rt_results['errors']:
            print(f"Warning: {err}")

    # Load config for defaults
    initial_freq = 89.9e6  # Default frequency
    use_icom = False  # Default device
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyfm.cfg')
    if os.path.exists(config_path):
        config = configparser.ConfigParser()
        try:
            config.read(config_path)
            if config.has_option('radio', 'last_frequency'):
                freq_mhz = config.getfloat('radio', 'last_frequency')
                if 88.0 <= freq_mhz <= 108.0:
                    initial_freq = freq_mhz * 1e6
            if config.has_option('radio', 'device'):
                device = config.get('radio', 'device').strip().lower()
                use_icom = (device == 'icom')
        except (ValueError, configparser.Error):
            pass

    # Command-line arguments override config
    if args.frequency is not None:
        if not (88.0 <= args.frequency <= 108.0):
            print("Error: Frequency must be between 88.0 and 108.0 MHz")
            sys.exit(1)
        initial_freq = args.frequency * 1e6

    if args.icom:
        use_icom = True
    elif args.bb60d:
        use_icom = False

    # Create radio instance
    radio = FMRadio(initial_freq=initial_freq, use_icom=use_icom)

    # Run rich UI
    try:
        run_rich_ui(radio)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    device_name = "IC-R8600" if use_icom else "BB60D"
    print(f"\n📻 Goodbye! Last frequency: {radio.frequency_mhz:.1f} MHz ({device_name})")

    # Hard exit to prevent ROCm/HIP double-free during Python shutdown.
    # Python's atexit handlers and GC race with the HIP runtime teardown,
    # causing "double free or corruption (!prev)" on process exit.
    # All resources are already cleaned up by radio.stop().
    os._exit(0)


if __name__ == "__main__":
    main()
