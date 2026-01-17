#!/usr/bin/env python3
"""
Phil's Weather Radio GUI

A PyQt5-based spectrum analyzer and waterfall display for the Signal Hound BB60D.
Default center frequency is 162.500 MHz (NOAA weather radio band).
Includes NBFM demodulator for NOAA Weather Radio reception.

Usage:
    python panadapter.py [--freq FREQ_MHZ]
"""

import sys
import argparse
import time
import numpy as np
from collections import deque
from scipy import signal

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSplitter, QFrame, QSlider,
    QCheckBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QDoubleValidator

import pyqtgraph as pg
from pyqtgraph import ColorMap
import sounddevice as sd

from bb60d import BB60D, BB_MIN_FREQ, BB_MAX_FREQ


# Default settings
DEFAULT_CENTER_FREQ = 162.525e6  # NOAA weather radio (WX7)
FREQ_STEP = 25e3  # 25 kHz step (weather channel spacing)
SAMPLE_RATE = 625000  # 625 kHz bandwidth (40 MHz / 64)
FFT_SIZE = 4096
WATERFALL_HISTORY = 300  # Number of rows in waterfall
UPDATE_RATE_MS = 50  # ~20 fps

# NBFM settings
NBFM_CHANNEL_BW = 12500  # 12.5 kHz channel bandwidth
NBFM_DEVIATION = 5000    # ±5 kHz deviation
AUDIO_SAMPLE_RATE = 48000  # Output audio sample rate


class NBFMDemodulator:
    """Narrowband FM demodulator for NOAA Weather Radio."""

    def __init__(self, input_sample_rate, audio_sample_rate=AUDIO_SAMPLE_RATE):
        self.input_sample_rate = input_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.tuned_offset = 0  # Offset from center freq in Hz
        self.squelch_level = -100  # dB threshold
        self.squelch_open = False

        # Calculate decimation to get to a reasonable IF rate (~32 kHz)
        # We'll do this in two stages for better filtering
        self.if_sample_rate = 32000
        self.decimation = int(input_sample_rate / self.if_sample_rate)
        self.actual_if_rate = input_sample_rate / self.decimation

        # Design channel filter (lowpass at IF)
        # NBFM channel is about ±7.5 kHz
        channel_cutoff = 7500 / (input_sample_rate / 2)
        self.channel_filter_b, self.channel_filter_a = signal.butter(
            5, channel_cutoff, btype='low'
        )
        self.channel_filter_state = None

        # Design audio lowpass filter (3 kHz for voice)
        audio_cutoff = 3000 / (self.actual_if_rate / 2)
        self.audio_filter_b, self.audio_filter_a = signal.butter(
            4, audio_cutoff, btype='low'
        )
        self.audio_filter_state = None

        # No de-emphasis for NBFM (NOAA Weather Radio uses flat audio)
        # WBFM broadcast uses 75µs, but NBFM typically has no pre-emphasis
        self.use_deemphasis = False
        self.deemph_alpha = 1.0  # Bypass if not used
        self.deemph_state = 0.0

        # Resampler for audio output
        self.resample_ratio = audio_sample_rate / self.actual_if_rate

        # State for FM demodulation
        self.prev_sample = 1 + 0j

        # Audio output buffer
        self.audio_buffer = deque(maxlen=int(audio_sample_rate * 0.5))  # 500ms buffer

    def set_tuned_offset(self, offset_hz):
        """Set the tuning offset from center frequency."""
        self.tuned_offset = offset_hz

    def set_squelch(self, level_db):
        """Set squelch threshold in dB."""
        self.squelch_level = level_db

    def process(self, iq_data):
        """
        Process IQ samples and return demodulated audio.

        Args:
            iq_data: Complex IQ samples at input_sample_rate

        Returns:
            Audio samples at audio_sample_rate, or None if squelched
        """
        if len(iq_data) == 0:
            return None

        # Frequency shift to center the desired channel
        if self.tuned_offset != 0:
            t = np.arange(len(iq_data)) / self.input_sample_rate
            shift = np.exp(-2j * np.pi * self.tuned_offset * t)
            iq_data = iq_data * shift

        # Apply channel filter
        if self.channel_filter_state is None:
            self.channel_filter_state = signal.lfilter_zi(
                self.channel_filter_b, self.channel_filter_a
            ) * iq_data[0]

        filtered, self.channel_filter_state = signal.lfilter(
            self.channel_filter_b, self.channel_filter_a,
            iq_data, zi=self.channel_filter_state
        )

        # Decimate to IF rate
        decimated = filtered[::self.decimation]

        # Check signal level for squelch
        signal_power = np.mean(np.abs(decimated) ** 2)
        signal_db = 10 * np.log10(signal_power + 1e-20)

        if signal_db < self.squelch_level:
            self.squelch_open = False
            return None

        self.squelch_open = True

        # FM demodulation using quadrature method
        # Instantaneous frequency = d(phase)/dt
        # Using: angle(x[n] * conj(x[n-1]))
        delayed = np.concatenate([[self.prev_sample], decimated[:-1]])
        self.prev_sample = decimated[-1]

        # Phase difference
        phase_diff = np.angle(decimated * np.conj(delayed))

        # Scale to audio (deviation determines gain)
        # phase_diff is in radians, max is pi for fs/2 deviation
        # For ±5kHz deviation at 32kHz sample rate: max_phase = 2*pi*5000/32000 = pi/3.2
        audio = phase_diff * (self.actual_if_rate / (2 * np.pi * NBFM_DEVIATION))

        # De-emphasis (bypassed for NBFM - no pre-emphasis used)
        if self.use_deemphasis:
            deemph_audio = np.zeros_like(audio)
            state = self.deemph_state
            for i in range(len(audio)):
                state = self.deemph_alpha * audio[i] + (1 - self.deemph_alpha) * state
                deemph_audio[i] = state
            self.deemph_state = state
        else:
            deemph_audio = audio

        # Apply audio lowpass filter
        if self.audio_filter_state is None:
            self.audio_filter_state = signal.lfilter_zi(
                self.audio_filter_b, self.audio_filter_a
            ) * deemph_audio[0]

        audio_filtered, self.audio_filter_state = signal.lfilter(
            self.audio_filter_b, self.audio_filter_a,
            deemph_audio, zi=self.audio_filter_state
        )

        # Resample to output audio rate
        num_output_samples = int(len(audio_filtered) * self.resample_ratio)
        if num_output_samples > 0:
            audio_resampled = signal.resample(audio_filtered, num_output_samples)
            # Normalize audio level
            audio_resampled = np.clip(audio_resampled * 0.5, -1.0, 1.0)
            return audio_resampled.astype(np.float32)

        return None

    def reset(self):
        """Reset demodulator state (call when changing frequency)."""
        self.channel_filter_state = None
        self.audio_filter_state = None
        self.prev_sample = 1 + 0j
        self.deemph_state = 0.0
        self.audio_buffer.clear()


class AudioOutput:
    """Audio output handler using sounddevice."""

    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=int(sample_rate * 1.0))  # 1 second buffer
        self.stream = None
        self.running = False
        self.gain = 1.0  # Volume gain (0.0 to 2.0)

    def set_gain(self, gain):
        """Set the audio gain (0.0 = mute, 1.0 = normal, 2.0 = +6dB)."""
        self.gain = max(0.0, min(2.0, gain))

    def start(self):
        """Start audio output stream."""
        self.running = True
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=1024,
            latency='low'
        )
        self.stream.start()

    def stop(self):
        """Stop audio output stream."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def write(self, audio_samples):
        """Add audio samples to the buffer."""
        if audio_samples is not None and len(audio_samples) > 0:
            self.buffer.extend(audio_samples)

    def _audio_callback(self, outdata, frames, time_info, status):
        """Sounddevice callback to fill audio output buffer."""
        gain = self.gain
        if len(self.buffer) >= frames:
            for i in range(frames):
                sample = self.buffer.popleft() * gain
                outdata[i, 0] = max(-1.0, min(1.0, sample))  # Clip to prevent distortion
        else:
            # Not enough data, output silence and use what we have
            available = len(self.buffer)
            for i in range(available):
                sample = self.buffer.popleft() * gain
                outdata[i, 0] = max(-1.0, min(1.0, sample))
            outdata[available:, 0] = 0


class DataThread(QThread):
    """Worker thread for continuous IQ acquisition from BB60D."""

    data_ready = pyqtSignal(np.ndarray)  # Emits IQ samples for display
    audio_ready = pyqtSignal(np.ndarray)  # Emits demodulated audio
    squelch_status = pyqtSignal(bool)  # Emits squelch open/closed
    error = pyqtSignal(str)

    # Target ~30 fps for display updates
    MIN_UPDATE_INTERVAL = 1.0 / 30.0

    def __init__(self, device, parent=None):
        super().__init__(parent)
        self.device = device
        self.running = False
        self.paused = False
        self.samples_per_block = FFT_SIZE * 2  # Get enough for FFT averaging
        self.demodulator = None
        self.demod_enabled = False
        self.last_squelch_state = False

    def set_demodulator(self, demodulator):
        """Set the NBFM demodulator instance."""
        self.demodulator = demodulator

    def enable_demod(self, enabled):
        """Enable or disable demodulation."""
        self.demod_enabled = enabled

    def run(self):
        """Continuously fetch IQ data and emit to GUI at limited rate."""
        self.running = True
        last_emit_time = 0

        while self.running:
            # Skip acquisition while paused (during reconfig)
            if self.paused:
                time.sleep(0.01)
                continue

            try:
                iq_data = self.device.fetch_iq(self.samples_per_block)

                # Process through demodulator if enabled
                if self.demod_enabled and self.demodulator:
                    audio = self.demodulator.process(iq_data)
                    if audio is not None:
                        self.audio_ready.emit(audio)

                    # Emit squelch status changes
                    if self.demodulator.squelch_open != self.last_squelch_state:
                        self.last_squelch_state = self.demodulator.squelch_open
                        self.squelch_status.emit(self.last_squelch_state)

                # Rate limit display updates
                now = time.time()
                if now - last_emit_time >= self.MIN_UPDATE_INTERVAL:
                    self.data_ready.emit(iq_data)
                    last_emit_time = now

            except Exception as e:
                # Ignore errors while paused (device being reconfigured)
                if not self.paused:
                    self.error.emit(str(e))
                    break

    def pause(self):
        """Pause data acquisition."""
        self.paused = True
        time.sleep(0.05)  # Give time for current fetch to complete

    def resume(self):
        """Resume data acquisition."""
        self.paused = False

    def stop(self):
        """Stop the acquisition thread."""
        self.running = False
        self.wait(1000)  # Wait up to 1 second for thread to finish


class SpectrumWidget(pg.PlotWidget):
    """Real-time spectrum display using pyqtgraph."""

    # Signal emitted when user clicks to tune (frequency in Hz)
    tuning_clicked = pyqtSignal(float)

    def __init__(self, center_freq, bandwidth, parent=None):
        super().__init__(parent)

        self.center_freq = center_freq
        self.bandwidth = bandwidth

        # Configure plot
        self.setLabel('left', 'Power', units='dBm')
        self.setLabel('bottom', 'Frequency', units='MHz')
        self.setTitle('Spectrum')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setYRange(-150, -70)
        # Fixed width for axes to align with waterfall
        self.getAxis('left').setWidth(80)
        self.getAxis('right').setWidth(0)

        # Create spectrum curve (light blue)
        self.spectrum_curve = self.plot(pen=pg.mkPen((100, 180, 255), width=1))

        # Create peak hold curve (optional, can be toggled)
        self.peak_curve = self.plot(pen=pg.mkPen('y', width=1, style=Qt.DotLine))
        self.peak_data = None
        self.peak_hold = False

        # Tuning indicator line
        self.tuning_line = pg.InfiniteLine(
            pos=center_freq / 1e6,
            angle=90,
            pen=pg.mkPen('r', width=2),
            movable=False
        )
        self.addItem(self.tuning_line)
        self.tuned_freq = center_freq

        # Enable mouse click for tuning
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # Frequency axis data
        self.freq_axis = None
        self._update_freq_axis()

    def _update_freq_axis(self, preserve_zoom=False):
        """Update frequency axis based on center freq and bandwidth."""
        freq_start = (self.center_freq - self.bandwidth / 2) / 1e6
        freq_end = (self.center_freq + self.bandwidth / 2) / 1e6
        self.freq_axis = np.linspace(freq_start, freq_end, FFT_SIZE)
        if not preserve_zoom:
            self.setXRange(freq_start, freq_end)

    def set_center_freq(self, freq):
        """Update center frequency, preserving current zoom span."""
        # Get current view span before changing
        view_range = self.viewRange()[0]
        span = view_range[1] - view_range[0]

        self.center_freq = freq
        self._update_freq_axis(preserve_zoom=True)

        # Re-center view on new frequency with same span
        new_center_mhz = freq / 1e6
        self.setXRange(new_center_mhz - span/2, new_center_mhz + span/2)

        self.peak_data = None  # Reset peak hold on freq change

    def set_bandwidth(self, bandwidth):
        """Update bandwidth."""
        self.bandwidth = bandwidth
        self._update_freq_axis()
        self.peak_data = None

    def update_spectrum(self, spectrum_db):
        """Update spectrum display with new data."""
        if self.freq_axis is not None and len(spectrum_db) == len(self.freq_axis):
            self.spectrum_curve.setData(self.freq_axis, spectrum_db)

            # Update peak hold
            if self.peak_hold:
                if self.peak_data is None:
                    self.peak_data = spectrum_db.copy()
                else:
                    self.peak_data = np.maximum(self.peak_data, spectrum_db)
                self.peak_curve.setData(self.freq_axis, self.peak_data)

    def toggle_peak_hold(self, enabled):
        """Toggle peak hold display."""
        self.peak_hold = enabled
        if not enabled:
            self.peak_curve.setData([], [])
            self.peak_data = None

    def set_tuned_freq(self, freq_hz):
        """Update the tuning indicator line position."""
        self.tuned_freq = freq_hz
        self.tuning_line.setValue(freq_hz / 1e6)

    def _on_mouse_clicked(self, event):
        """Handle mouse click for tuning."""
        if event.button() == Qt.LeftButton:
            # Get click position in plot coordinates
            pos = event.scenePos()
            if self.sceneBoundingRect().contains(pos):
                mouse_point = self.plotItem.vb.mapSceneToView(pos)
                freq_mhz = mouse_point.x()
                freq_hz = freq_mhz * 1e6
                # Emit the tuning signal
                self.tuning_clicked.emit(freq_hz)


class WaterfallWidget(pg.GraphicsLayoutWidget):
    """Scrolling waterfall display using pyqtgraph ImageItem."""

    # Signal emitted when user clicks to tune (frequency in Hz)
    tuning_clicked = pyqtSignal(float)

    def __init__(self, center_freq, bandwidth, history=WATERFALL_HISTORY, parent=None):
        super().__init__(parent)

        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.history = history

        # Initialize waterfall data buffer (rows=time, cols=freq)
        self.waterfall_data = np.zeros((history, FFT_SIZE), dtype=np.float32)
        self.waterfall_data.fill(-120)  # Initialize with floor value

        # Set intensity range (match spectrum display range)
        self.min_db = -150
        self.max_db = -70

        # Create plot and image item - set margins to align with spectrum
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.ci.layout.setSpacing(0)
        self.plot = self.addPlot()
        self.plot.setLabel('bottom', 'Frequency', units='MHz')
        self.plot.setTitle('Waterfall')
        # Fixed width for axes to align with spectrum
        self.plot.getAxis('left').setWidth(80)
        self.plot.getAxis('left').setStyle(showValues=False)
        self.plot.getAxis('left').setTicks([])
        self.plot.getAxis('right').setWidth(0)

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        # Disable auto-range to prevent view drift
        self.plot.vb.disableAutoRange()

        # Set up colormap (light blue gradient)
        colors = [
            (0, 0, 0),        # Black (noise floor)
            (0, 0, 40),       # Very dark blue
            (0, 40, 100),     # Dark blue
            (30, 100, 180),   # Medium blue
            (100, 180, 255),  # Light blue
            (200, 230, 255),  # Very light blue (strong signal)
        ]
        positions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.colormap = ColorMap(positions, colors)
        self.lut = self.colormap.getLookupTable(nPts=256)
        self.image_item.setLookupTable(self.lut)
        self.image_item.setLevels([self.min_db, self.max_db])

        # Tuning indicator line
        self.tuning_line = pg.InfiniteLine(
            pos=center_freq / 1e6,
            angle=90,
            pen=pg.mkPen('r', width=2),
            movable=False
        )
        self.plot.addItem(self.tuning_line)
        self.tuned_freq = center_freq

        # Enable mouse click for tuning
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # Set up axis range
        self._update_view()

    def _update_view(self):
        """Update view range for current frequency."""
        self.freq_start = (self.center_freq - self.bandwidth / 2) / 1e6
        self.freq_end = (self.center_freq + self.bandwidth / 2) / 1e6
        self.plot.setXRange(self.freq_start, self.freq_end, padding=0)
        self.plot.setYRange(0, self.history, padding=0)

    def set_center_freq(self, freq):
        """Update center frequency."""
        self.center_freq = freq
        self._update_view()
        # Clear waterfall on freq change
        self.waterfall_data.fill(-120)

    def set_bandwidth(self, bandwidth):
        """Update bandwidth."""
        self.bandwidth = bandwidth
        self._update_view()

    def update_waterfall(self, spectrum_db):
        """Add new spectrum line to waterfall."""
        # Scroll data (newest at row 0)
        self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
        self.waterfall_data[0, :] = spectrum_db[:FFT_SIZE]

        # Flip so newest is at top, transpose so freq is x-axis
        # Shape: (FFT_SIZE, history) - x=freq bins, y=time
        display_data = np.flipud(self.waterfall_data).T

        # Calculate transform to map image pixels to frequency axis
        # Offset by half a bin so pixel centers align with spectrum points
        freq_span = self.freq_end - self.freq_start
        bin_width = freq_span / FFT_SIZE
        tr = pg.QtGui.QTransform()
        tr.translate(self.freq_start - bin_width / 2, 0)
        tr.scale(bin_width, 1)

        self.image_item.setTransform(tr)
        self.image_item.setImage(display_data, autoLevels=False, levels=(self.min_db, self.max_db))

    def set_intensity_range(self, min_db, max_db):
        """Set the intensity range for the colormap."""
        self.min_db = min_db
        self.max_db = max_db

    def set_tuned_freq(self, freq_hz):
        """Update the tuning indicator line position."""
        self.tuned_freq = freq_hz
        self.tuning_line.setValue(freq_hz / 1e6)

    def _on_mouse_clicked(self, event):
        """Handle mouse click for tuning."""
        if event.button() == Qt.LeftButton:
            # Get click position in plot coordinates
            pos = event.scenePos()
            if self.plot.sceneBoundingRect().contains(pos):
                mouse_point = self.plot.vb.mapSceneToView(pos)
                freq_mhz = mouse_point.x()
                freq_hz = freq_mhz * 1e6
                # Emit the tuning signal
                self.tuning_clicked.emit(freq_hz)


class SMeterWidget(QFrame):
    """S-meter display widget showing signal strength.

    Standard S-meter calibration: S9 = -93 dBm, 6 dB per S-unit.
    """

    S9_DBM = -93  # S9 reference level in dBm
    DB_PER_S_UNIT = 6  # dB per S-unit below S9

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.signal_dbm = -120  # Current signal level

        # Fixed size - don't expand with window resize
        self.setFixedHeight(42)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Push content to the right
        layout.addStretch()

        # S-meter label
        label = QLabel('S:')
        label.setStyleSheet('font-weight: bold;')
        layout.addWidget(label)

        # S-meter bar using QProgressBar style
        self.meter_bar = pg.PlotWidget()
        self.meter_bar.setFixedHeight(30)
        self.meter_bar.setFixedWidth(200)
        self.meter_bar.setBackground('#1a1a1a')
        self.meter_bar.hideAxis('left')
        self.meter_bar.hideAxis('bottom')
        self.meter_bar.setMouseEnabled(False, False)
        self.meter_bar.setMenuEnabled(False)

        # Create bar item
        self.bar_item = pg.BarGraphItem(x=[0], height=[0], width=0.8, brush='lime')
        self.meter_bar.addItem(self.bar_item)

        # Set range: 0-9 for S-units, 9-12 for S9+10/20/30
        self.meter_bar.setXRange(-0.5, 12.5, padding=0)
        self.meter_bar.setYRange(0, 1, padding=0)

        # Add S-unit tick marks
        for i in range(10):
            line = pg.InfiniteLine(pos=i, angle=90, pen=pg.mkPen('#444', width=1))
            self.meter_bar.addItem(line)
        # Add +10, +20, +30 marks
        for i in [10, 11, 12]:
            line = pg.InfiniteLine(pos=i, angle=90, pen=pg.mkPen('#664400', width=1))
            self.meter_bar.addItem(line)

        layout.addWidget(self.meter_bar)

        # S-meter text readout
        self.reading_label = QLabel('S0')
        self.reading_label.setMinimumWidth(60)
        self.reading_label.setStyleSheet('font-family: monospace; font-weight: bold; font-size: 14px;')
        layout.addWidget(self.reading_label)

        # dBm readout
        self.dbm_label = QLabel('-120 dBm')
        self.dbm_label.setMinimumWidth(80)
        self.dbm_label.setStyleSheet('font-family: monospace; color: #888;')
        layout.addWidget(self.dbm_label)

    def set_level(self, dbm):
        """Set the signal level in dBm and update display."""
        self.signal_dbm = dbm

        # Convert dBm to S-units
        # S9 = -93 dBm, each S-unit below is 6 dB
        if dbm <= self.S9_DBM:
            # Below or at S9
            s_units = max(0, (dbm - (self.S9_DBM - 9 * self.DB_PER_S_UNIT)) / self.DB_PER_S_UNIT)
            s_text = f'S{int(s_units)}'
            bar_value = s_units
        else:
            # Above S9: S9+10, S9+20, etc.
            over_s9 = dbm - self.S9_DBM
            s_text = f'S9+{int(over_s9)}'
            # Bar extends past 9 for over-S9 signals
            bar_value = 9 + (over_s9 / 10)  # +10dB = 1 unit past S9

        # Update bar graph
        bar_value = min(bar_value, 12)  # Cap at S9+30

        # Color based on level (blue gradient)
        if bar_value < 3:
            color = '#1a3a5c'  # Dark blue for weak
        elif bar_value < 6:
            color = '#2266aa'  # Medium blue for moderate
        elif bar_value < 9:
            color = '#44aaff'  # Bright blue for good
        else:
            color = '#99ddff'  # Light blue for strong

        self.bar_item.setOpts(x=[bar_value/2], height=[1], width=bar_value, brush=color)

        # Update text
        self.reading_label.setText(s_text)
        self.dbm_label.setText(f'{dbm:.0f} dBm')


class MainWindow(QMainWindow):
    """Main application window with spectrum and waterfall displays."""

    def __init__(self, center_freq=DEFAULT_CENTER_FREQ):
        super().__init__()

        self.center_freq = center_freq
        self.bandwidth = SAMPLE_RATE  # Bandwidth equals sample rate for IQ
        self.device = None
        self.data_thread = None

        # FFT processing
        self.fft_window = np.hanning(FFT_SIZE)
        self.spectrum_avg = None
        self.avg_factor = 0.7  # Exponential averaging factor

        # NBFM demodulator and audio
        self.demodulator = None
        self.audio_output = None
        self.tuned_freq = center_freq  # Currently tuned frequency for demod

        self.setup_ui()
        self.setup_device()

    def setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(f"Phil's Weather Radio - {self.center_freq/1e6:.3f} MHz")
        self.setGeometry(100, 100, 2000, 1500)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control bar at top
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_layout = QHBoxLayout(control_frame)

        # Frequency display and entry
        freq_label = QLabel('Center Freq:')
        control_layout.addWidget(freq_label)

        self.freq_entry = QLineEdit(f'{self.center_freq/1e6:.3f}')
        self.freq_entry.setFixedWidth(100)
        self.freq_entry.setValidator(QDoubleValidator(0.009, 6400.0, 6))
        self.freq_entry.returnPressed.connect(self.on_freq_entry)
        control_layout.addWidget(self.freq_entry)

        mhz_label = QLabel('MHz')
        control_layout.addWidget(mhz_label)

        # Frequency buttons
        btn_down = QPushButton('<< -25 kHz')
        btn_down.clicked.connect(lambda: self.tune(-FREQ_STEP))
        control_layout.addWidget(btn_down)

        btn_up = QPushButton('+25 kHz >>')
        btn_up.clicked.connect(lambda: self.tune(FREQ_STEP))
        control_layout.addWidget(btn_up)

        control_layout.addStretch()

        # NOAA preset buttons
        noaa_label = QLabel('NOAA:')
        control_layout.addWidget(noaa_label)

        noaa_freqs = [
            ('WX1', 162.550),
            ('WX2', 162.400),
            ('WX3', 162.475),
            ('WX4', 162.425),
            ('WX5', 162.450),
            ('WX6', 162.500),
            ('WX7', 162.525),
        ]
        for name, freq in noaa_freqs:
            btn = QPushButton(name)
            btn.setFixedWidth(50)
            btn.clicked.connect(lambda checked, f=freq: self.set_frequency(f * 1e6))
            control_layout.addWidget(btn)

        control_layout.addStretch()

        # Status label
        self.status_label = QLabel('Initializing...')
        control_layout.addWidget(self.status_label)

        main_layout.addWidget(control_frame)

        # Demodulator control bar
        demod_frame = QFrame()
        demod_frame.setFrameStyle(QFrame.StyledPanel)
        demod_layout = QHBoxLayout(demod_frame)

        # Enable demod checkbox (on by default)
        self.demod_checkbox = QCheckBox('NBFM Demod')
        self.demod_checkbox.setChecked(True)
        self.demod_checkbox.stateChanged.connect(self.on_demod_toggle)
        demod_layout.addWidget(self.demod_checkbox)

        # Tuned frequency display
        tuned_label = QLabel('Tuned:')
        demod_layout.addWidget(tuned_label)
        self.tuned_freq_label = QLabel(f'{self.center_freq/1e6:.4f} MHz')
        self.tuned_freq_label.setMinimumWidth(120)
        demod_layout.addWidget(self.tuned_freq_label)

        demod_layout.addStretch()

        # Volume control
        volume_label = QLabel('Vol:')
        demod_layout.addWidget(volume_label)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)  # 50% = gain of 1.0
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        demod_layout.addWidget(self.volume_slider)

        self.volume_value_label = QLabel('50%')
        self.volume_value_label.setMinimumWidth(40)
        demod_layout.addWidget(self.volume_value_label)

        demod_layout.addStretch()

        # Squelch control
        squelch_label = QLabel('Squelch:')
        demod_layout.addWidget(squelch_label)

        self.squelch_slider = QSlider(Qt.Horizontal)
        self.squelch_slider.setMinimum(-140)
        self.squelch_slider.setMaximum(-60)
        self.squelch_slider.setValue(-100)
        self.squelch_slider.setFixedWidth(100)
        self.squelch_slider.valueChanged.connect(self.on_squelch_changed)
        demod_layout.addWidget(self.squelch_slider)

        self.squelch_value_label = QLabel('-100 dB')
        self.squelch_value_label.setMinimumWidth(60)
        demod_layout.addWidget(self.squelch_value_label)

        # Squelch indicator
        self.squelch_indicator = QLabel('◯')  # ◯ = closed, ● = open
        self.squelch_indicator.setStyleSheet('color: gray; font-size: 16px;')
        demod_layout.addWidget(self.squelch_indicator)

        demod_layout.addStretch()

        # Click-to-tune instruction
        click_label = QLabel('Click spectrum to tune')
        click_label.setStyleSheet('color: gray; font-style: italic;')
        demod_layout.addWidget(click_label)

        main_layout.addWidget(demod_frame)

        # S-meter bar
        self.smeter = SMeterWidget()
        main_layout.addWidget(self.smeter)

        # Splitter for spectrum and waterfall
        splitter = QSplitter(Qt.Vertical)

        # Spectrum display
        self.spectrum_widget = SpectrumWidget(self.center_freq, self.bandwidth)
        splitter.addWidget(self.spectrum_widget)

        # Waterfall display
        self.waterfall_widget = WaterfallWidget(self.center_freq, self.bandwidth)
        splitter.addWidget(self.waterfall_widget)

        # Link X-axes so zoom affects both displays
        self.waterfall_widget.plot.setXLink(self.spectrum_widget.getPlotItem())

        # Connect click-to-tune signals
        self.spectrum_widget.tuning_clicked.connect(self.on_tuning_clicked)
        self.waterfall_widget.tuning_clicked.connect(self.on_tuning_clicked)

        # Set initial zoom to 100 kHz centered view (±50 kHz) after UI is ready
        QTimer.singleShot(0, self._set_initial_zoom)

        # Set initial splitter sizes (spectrum 40%, waterfall 60%)
        splitter.setSizes([300, 450])

        main_layout.addWidget(splitter)

        # Keyboard shortcuts
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence

        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.tune(-FREQ_STEP))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.tune(FREQ_STEP))
        QShortcut(QKeySequence(Qt.Key_Up), self, lambda: self.tune(FREQ_STEP * 4))
        QShortcut(QKeySequence(Qt.Key_Down), self, lambda: self.tune(-FREQ_STEP * 4))
        QShortcut(QKeySequence('P'), self, self.toggle_peak_hold)
        QShortcut(QKeySequence('Q'), self, self.close)
        QShortcut(QKeySequence('Escape'), self, self.close)

    def _set_initial_zoom(self):
        """Set initial 100 kHz zoom after UI is ready."""
        center_mhz = self.center_freq / 1e6
        self.spectrum_widget.setXRange(center_mhz - 0.05, center_mhz + 0.05, padding=0)
        self.waterfall_widget.plot.setXRange(center_mhz - 0.05, center_mhz + 0.05, padding=0)

    def setup_device(self):
        """Initialize BB60D device and start data acquisition."""
        try:
            self.device = BB60D()
            # Override FM frequency limits for weather radio use
            self.device.FM_MIN_FREQ = BB_MIN_FREQ
            self.device.FM_MAX_FREQ = BB_MAX_FREQ

            self.device.open()
            self.device.configure_iq_streaming(self.center_freq, SAMPLE_RATE)

            # Update bandwidth to actual device sample rate (for FFT frequency axis)
            # Note: iq_sample_rate is the FFT span, iq_bandwidth is the filter bandwidth
            self.bandwidth = self.device.iq_sample_rate
            self.spectrum_widget.set_bandwidth(self.bandwidth)
            self.waterfall_widget.set_bandwidth(self.bandwidth)

            # Initialize NBFM demodulator at center frequency
            self.demodulator = NBFMDemodulator(self.device.iq_sample_rate)
            self.demodulator.set_squelch(self.squelch_slider.value())
            self.demodulator.set_tuned_offset(self.tuned_freq - self.center_freq)
            self.demodulator.reset()  # Ensure clean start

            # Initialize audio output with initial volume from slider
            self.audio_output = AudioOutput()
            self.audio_output.set_gain(self.volume_slider.value() / 50.0)

            # Start data acquisition thread
            self.data_thread = DataThread(self.device)
            self.data_thread.set_demodulator(self.demodulator)
            self.data_thread.data_ready.connect(self.process_iq_data)
            self.data_thread.audio_ready.connect(self.on_audio_ready)
            self.data_thread.squelch_status.connect(self.on_squelch_status)
            self.data_thread.error.connect(self.on_error)
            self.data_thread.start()

            # Enable demod by default (checkbox is pre-checked)
            self.data_thread.enable_demod(True)
            self.audio_output.start()

            self.status_label.setText(f'Running - {self.device.iq_sample_rate/1e3:.1f} kHz')

        except Exception as e:
            self.status_label.setText(f'Error: {e}')
            print(f'Device initialization error: {e}')

    def process_iq_data(self, iq_data):
        """Process IQ data: compute FFT and update displays."""
        # Take first FFT_SIZE samples
        if len(iq_data) < FFT_SIZE:
            return

        iq_block = iq_data[:FFT_SIZE]

        # Apply window and compute FFT
        windowed = iq_block * self.fft_window
        fft_result = np.fft.fftshift(np.fft.fft(windowed))

        # Convert to power in dB with proper normalization
        # Normalize by FFT size and window power correction
        # Hanning window coherent gain = 0.5, so power correction = 1/(0.5^2) = 4
        window_correction = 4.0
        power = (np.abs(fft_result) ** 2) / (FFT_SIZE ** 2) * window_correction
        # Avoid log of zero
        power = np.maximum(power, 1e-20)
        spectrum_db = 10 * np.log10(power)

        # Apply exponential averaging for smoother display
        if self.spectrum_avg is None:
            self.spectrum_avg = spectrum_db
        else:
            self.spectrum_avg = (self.avg_factor * self.spectrum_avg +
                                 (1 - self.avg_factor) * spectrum_db)

        # Update displays
        self.spectrum_widget.update_spectrum(self.spectrum_avg)
        self.waterfall_widget.update_waterfall(self.spectrum_avg)

        # Update S-meter with signal level at tuned frequency
        self._update_smeter(self.spectrum_avg)

    def _update_smeter(self, spectrum_db):
        """Update S-meter with signal level at the tuned frequency."""
        # Find the FFT bin corresponding to the tuned frequency
        freq_offset = self.tuned_freq - self.center_freq
        bin_hz = self.bandwidth / FFT_SIZE
        bin_index = int(FFT_SIZE / 2 + freq_offset / bin_hz)

        # Clamp to valid range
        bin_index = max(0, min(FFT_SIZE - 1, bin_index))

        # Average a few bins around the tuned frequency for stability
        half_width = 3  # Average ±3 bins (~1 kHz at 625 kHz / 4096)
        start_bin = max(0, bin_index - half_width)
        end_bin = min(FFT_SIZE, bin_index + half_width + 1)
        signal_db = np.mean(spectrum_db[start_bin:end_bin])

        # Update S-meter display
        self.smeter.set_level(signal_db)

    def tune(self, delta):
        """Change frequency by delta Hz."""
        new_freq = self.center_freq + delta
        self.set_frequency(new_freq)

    def set_frequency(self, freq):
        """Set new center frequency."""
        # Clamp to valid range
        freq = max(BB_MIN_FREQ, min(BB_MAX_FREQ, freq))

        if freq != self.center_freq:
            self.center_freq = freq

            # Pause data thread during device reconfiguration
            if self.data_thread:
                self.data_thread.pause()

            # Update device
            if self.device and self.device.streaming_mode:
                self.device.set_frequency(freq)

            # Resume data thread
            if self.data_thread:
                self.data_thread.resume()

            # Update displays
            self.spectrum_widget.set_center_freq(freq)
            self.waterfall_widget.set_center_freq(freq)

            # Update demodulator offset (tuned freq relative to new center)
            if self.demodulator:
                offset = self.tuned_freq - freq
                self.demodulator.set_tuned_offset(offset)
                self.demodulator.reset()

            # Update UI
            self.freq_entry.setText(f'{freq/1e6:.3f}')
            self.setWindowTitle(f"Phil's Weather Radio - {freq/1e6:.3f} MHz")

            # Reset averaging on frequency change
            self.spectrum_avg = None

    def on_freq_entry(self):
        """Handle frequency entry from text field."""
        try:
            freq_mhz = float(self.freq_entry.text())
            self.set_frequency(freq_mhz * 1e6)
        except ValueError:
            pass

    def toggle_peak_hold(self):
        """Toggle peak hold on spectrum display."""
        current = self.spectrum_widget.peak_hold
        self.spectrum_widget.toggle_peak_hold(not current)

    def on_error(self, error_msg):
        """Handle error from data thread."""
        self.status_label.setText(f'Error: {error_msg}')
        print(f'Data thread error: {error_msg}')

    def on_demod_toggle(self, state):
        """Handle demodulator enable/disable."""
        enabled = state == Qt.Checked
        if self.data_thread:
            self.data_thread.enable_demod(enabled)

        if enabled:
            # Start audio output
            if self.audio_output:
                self.audio_output.start()
        else:
            # Stop audio output
            if self.audio_output:
                self.audio_output.stop()
            # Reset squelch indicator
            self.squelch_indicator.setText('◯')
            self.squelch_indicator.setStyleSheet('color: gray; font-size: 16px;')

    def on_squelch_changed(self, value):
        """Handle squelch slider change."""
        self.squelch_value_label.setText(f'{value} dB')
        if self.demodulator:
            self.demodulator.set_squelch(value)

    def on_volume_changed(self, value):
        """Handle volume slider change."""
        self.volume_value_label.setText(f'{value}%')
        # Map 0-100% to gain 0.0-2.0 (50% = 1.0 = unity gain)
        gain = value / 50.0
        if self.audio_output:
            self.audio_output.set_gain(gain)

    def on_tuning_clicked(self, freq_hz):
        """Handle click-to-tune on spectrum or waterfall."""
        # Snap to nearest 25 kHz (NOAA weather channel spacing)
        freq_hz = round(freq_hz / 25e3) * 25e3
        self.tuned_freq = freq_hz

        # Update tuning indicator on both displays
        self.spectrum_widget.set_tuned_freq(freq_hz)
        self.waterfall_widget.set_tuned_freq(freq_hz)

        # Update tuned frequency label
        self.tuned_freq_label.setText(f'{freq_hz/1e6:.4f} MHz')

        # Calculate offset from center frequency and update demodulator
        if self.demodulator:
            offset = freq_hz - self.center_freq
            self.demodulator.set_tuned_offset(offset)
            self.demodulator.reset()  # Reset filter states for clean transition

    def on_audio_ready(self, audio_samples):
        """Handle demodulated audio from data thread."""
        if self.audio_output:
            self.audio_output.write(audio_samples)

    def on_squelch_status(self, is_open):
        """Handle squelch status change."""
        if is_open:
            self.squelch_indicator.setText('●')
            self.squelch_indicator.setStyleSheet('color: lime; font-size: 16px;')
        else:
            self.squelch_indicator.setText('◯')
            self.squelch_indicator.setStyleSheet('color: gray; font-size: 16px;')

    def closeEvent(self, event):
        """Clean up on window close."""
        # Stop audio output
        if self.audio_output:
            self.audio_output.stop()

        if self.data_thread:
            self.data_thread.stop()

        if self.device:
            try:
                self.device.close()
            except:
                pass

        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Phil's Weather Radio GUI")
    parser.add_argument('--freq', type=float, default=DEFAULT_CENTER_FREQ/1e6,
                        help='Center frequency in MHz (default: 162.500)')
    args = parser.parse_args()

    # Configure pyqtgraph for performance
    pg.setConfigOptions(antialias=False, useOpenGL=True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Consistent cross-platform look

    window = MainWindow(center_freq=args.freq * 1e6)
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
