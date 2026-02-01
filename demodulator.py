#!/usr/bin/env python3
"""
Software FM Stereo Demodulator

Provides FM stereo demodulation from IQ samples using quadrature demodulation
with 19 kHz pilot detection and L-R subcarrier decoding.
"""

import time
import numpy as np
from scipy import signal


class FMStereoDecoder:
    """
    FM Stereo decoder for broadcast FM.

    Detects 19 kHz pilot tone and decodes L-R difference signal
    from 38 kHz DSB-SC subcarrier.

    FM Stereo signal structure:
    - 0-15 kHz: L+R (mono compatible)
    - 19 kHz: Pilot tone
    - 23-53 kHz: L-R on 38 kHz DSB-SC carrier
    """

    def __init__(self, iq_sample_rate=250000, audio_sample_rate=48000,
                 deviation=75000, deemphasis=75e-6):
        """
        Initialize FM stereo decoder.

        Args:
            iq_sample_rate: Input IQ sample rate in Hz
            audio_sample_rate: Output audio sample rate in Hz
            deviation: FM deviation in Hz (75 kHz for broadcast FM)
            deemphasis: De-emphasis time constant in seconds
        """
        self.iq_sample_rate = iq_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.deviation = deviation

        # Pilot detection state
        self._pilot_detected = False
        self._pilot_level = 0.0
        self.pilot_threshold = 0.05  # Minimum pilot level for detection

        # State for continuous processing
        self.last_sample = complex(1, 0)

        # Adaptive rate control: adjusts resample ratio to match audio card clock
        # 1.0 = nominal, >1.0 = produce more samples (buffer low), <1.0 = fewer (buffer high)
        self._rate_adjust = 1.0
        self._nominal_ratio = audio_sample_rate / iq_sample_rate

        # Design filters (all at IQ sample rate)
        nyq = iq_sample_rate / 2

        # Pilot bandpass filter (18.5-19.5 kHz) — Kaiser beta=7.0 (~70 dB stopband)
        pilot_low = 18500 / nyq
        pilot_high = 19500 / nyq
        self.pilot_bpf = signal.firwin(201, [pilot_low, pilot_high],
                                        pass_zero=False, window=('kaiser', 7.0))
        self.pilot_bpf_state = signal.lfilter_zi(self.pilot_bpf, 1.0)

        # L+R lowpass filter (15 kHz) — Kaiser beta=6.0 (~60 dB stopband)
        lr_sum_cutoff = 15000 / nyq
        self.lr_sum_lpf = signal.firwin(127, lr_sum_cutoff, window=('kaiser', 6.0))
        self.lr_sum_lpf_state = signal.lfilter_zi(self.lr_sum_lpf, 1.0)

        # L-R bandpass filter (23-53 kHz) — Kaiser beta=7.0 (~70 dB stopband)
        lr_diff_low = 23000 / nyq
        lr_diff_high = min(53000 / nyq, 0.95)  # Stay below Nyquist
        self.lr_diff_bpf = signal.firwin(201, [lr_diff_low, lr_diff_high],
                                          pass_zero=False, window=('kaiser', 7.0))
        self.lr_diff_bpf_state = signal.lfilter_zi(self.lr_diff_bpf, 1.0)

        # L-R lowpass filter after demodulation (15 kHz) — Kaiser beta=6.0 (~60 dB stopband)
        self.lr_diff_lpf = signal.firwin(127, lr_sum_cutoff, window=('kaiser', 6.0))
        self.lr_diff_lpf_state = signal.lfilter_zi(self.lr_diff_lpf, 1.0)

        # De-emphasis filter (at output audio rate)
        # Use matched-pole first-order IIR for accurate 75µs time constant
        fs = audio_sample_rate
        a = np.exp(-1.0 / (deemphasis * fs))
        self.deem_b = np.array([1.0 - a])
        self.deem_a = np.array([1.0, -a])
        self.deem_state_l = signal.lfilter_zi(self.deem_b, self.deem_a)
        self.deem_state_r = signal.lfilter_zi(self.deem_b, self.deem_a)

        # Design noise measurement bandpass filter (80-120 kHz)
        # Must be:
        # - Above FM multiplex (stereo 38 kHz, RDS 57 kHz, HD sidebands ~75 kHz)
        # - Below 0.8 of Nyquist for good filter performance at all sample rates
        # At 480 kHz (R8600): 80-120 kHz is 0.33-0.50 of Nyquist (240 kHz) - good
        # At 312.5 kHz (BB60D): 80-120 kHz is 0.51-0.77 of Nyquist (156.25 kHz) - good
        noise_low_hz = 80000
        noise_high_hz = 120000
        noise_low = noise_low_hz / nyq
        noise_high = min(noise_high_hz / nyq, 0.90)
        if noise_high > noise_low and noise_low < 0.90:
            self.noise_bpf = signal.firwin(51, [noise_low, noise_high],
                                           pass_zero=False, window=('kaiser', 5.0))
            self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)
            self.noise_bandwidth = noise_high_hz - noise_low_hz  # 40000 Hz
        else:
            self.noise_bpf = None

        # SNR measurement state (pilot-referenced)
        # We use the 19 kHz pilot as the reference since it's broadcast at a fixed level
        self._snr_db = 0.0
        self._noise_power = 1e-10
        self.pilot_bandwidth = 1000  # Hz (18.5-19.5 kHz BPF)

        # Peak amplitude tracking (for clipping detection)
        self._peak_amplitude = 0.0

        # Group delay compensation buffer for L+R path.
        # The L-R path passes through an extra BPF (201 taps, delay=100)
        # before reaching the LPF. Both LPFs are 127-tap so their delays
        # cancel. Without compensation the L+R and L-R signals are
        # misaligned by 100 samples (~400 us at 250 kHz), which destroys
        # stereo separation at higher audio frequencies.
        self._lr_sum_delay = (len(self.lr_diff_bpf) - 1) // 2
        self._lr_sum_delay_buf = np.zeros(self._lr_sum_delay, dtype=np.float64)

        # Store last baseband for RDS processing
        self._last_baseband = None
        self._last_pilot = None

        # Tone controls (bass and treble boost)
        self.bass_boost_enabled = True    # Default on
        self.treble_boost_enabled = True  # Default on
        self._setup_tone_filters()

        # Stereo blend settings (blend to mono when SNR is low)
        # Thresholds based on pilot SNR measurement:
        # - 8 dB: very weak signal, use mono to reduce noise
        # - 20 dB: good signal, full stereo separation
        self.stereo_blend_enabled = True
        self.stereo_blend_low = 8.0     # Below this SNR: full mono
        self.stereo_blend_high = 20.0   # Above this SNR: full stereo
        self._blend_factor = 1.0        # Current blend (0=mono, 1=stereo)

        # Per-stage profiling (EMA-smoothed, microseconds)
        self.profile_enabled = False
        self._profile = {
            'fm_demod': 0.0,
            'pilot_bpf': 0.0,
            'lr_sum_lpf': 0.0,
            'lr_diff_bpf': 0.0,
            'lr_diff_lpf': 0.0,
            'noise_bpf': 0.0,
            'resample': 0.0,
            'deemphasis': 0.0,
            'tone': 0.0,
            'limiter': 0.0,
            'total': 0.0,
        }

    def _design_low_shelf(self, fc, gain_db, fs):
        """Design low shelf biquad filter coefficients (Audio EQ Cookbook)."""
        A = 10 ** (gain_db / 40)  # sqrt of linear gain
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / 2 * np.sqrt(2)  # Q = 0.707 for gentle slope

        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)
        two_sqrt_A_alpha = 2 * sqrt_A * alpha

        b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha

        # Normalize
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        return b, a

    def _design_high_shelf(self, fc, gain_db, fs):
        """Design high shelf biquad filter coefficients (Audio EQ Cookbook)."""
        A = 10 ** (gain_db / 40)  # sqrt of linear gain
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / 2 * np.sqrt(2)  # Q = 0.707 for gentle slope

        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)
        two_sqrt_A_alpha = 2 * sqrt_A * alpha

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha

        # Normalize
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        return b, a

    def _setup_tone_filters(self):
        """Set up bass and treble shelf filters (+3dB)."""
        fs = self.audio_sample_rate

        # Bass boost: +3dB low shelf at 250 Hz
        self.bass_b, self.bass_a = self._design_low_shelf(250, 3.0, fs)
        self.bass_state_l = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.bass_state_r = signal.lfilter_zi(self.bass_b, self.bass_a)

        # Treble boost: +3dB high shelf at 3500 Hz
        self.treble_b, self.treble_a = self._design_high_shelf(3500, 3.0, fs)
        self.treble_state_l = signal.lfilter_zi(self.treble_b, self.treble_a)
        self.treble_state_r = signal.lfilter_zi(self.treble_b, self.treble_a)

    @property
    def pilot_detected(self):
        """Returns True if 19 kHz pilot tone is detected."""
        return self._pilot_detected

    @property
    def snr_db(self):
        """Return the current SNR estimate in dB."""
        return self._snr_db

    @property
    def peak_amplitude(self):
        """Return peak amplitude (before limiting). >1.0 means limiter is active."""
        return self._peak_amplitude

    @property
    def stereo_blend_factor(self):
        """Return current stereo blend factor (0=mono, 1=full stereo)."""
        return self._blend_factor

    @property
    def rate_adjust(self):
        """Return current rate adjustment factor for adaptive resampling."""
        return self._rate_adjust

    @rate_adjust.setter
    def rate_adjust(self, value):
        """Set rate adjustment factor (0.99-1.01 typical range for clock drift)."""
        # Clamp to reasonable range to prevent audio artifacts
        self._rate_adjust = max(0.98, min(1.02, value))

    @property
    def profile(self):
        """Return per-stage profiling data (EMA-smoothed microseconds)."""
        return dict(self._profile)

    @property
    def last_baseband(self):
        """Returns the last FM baseband signal for RDS processing.

        This is the demodulated FM signal at the IQ sample rate,
        before audio filtering and decimation. Contains the full
        spectrum including 57 kHz RDS subcarrier.
        """
        return self._last_baseband

    @property
    def last_pilot(self):
        """Returns the last extracted 19 kHz pilot signal.

        Can be used by RDS decoder to derive 57 kHz carrier (3x pilot).
        """
        return self._last_pilot

    def _prof(self, key, t0):
        """Record EMA-smoothed stage timing in microseconds."""
        elapsed = (time.perf_counter() - t0) * 1e6
        self._profile[key] = 0.9 * self._profile[key] + 0.1 * elapsed
        return time.perf_counter()

    def demodulate(self, iq_samples):
        """
        Demodulate FM stereo signal from IQ samples.

        Args:
            iq_samples: numpy array of complex64 IQ samples

        Returns:
            numpy array of shape (N, 2) with L and R channels as float32
        """
        if len(iq_samples) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        profiling = self.profile_enabled
        if profiling:
            t_total = time.perf_counter()
            t0 = t_total

        # FM demodulation (quadrature discriminator)
        samples = np.concatenate([[self.last_sample], iq_samples])
        self.last_sample = iq_samples[-1]
        product = samples[1:] * np.conj(samples[:-1])
        baseband = np.angle(product) * (self.iq_sample_rate / (2 * np.pi * self.deviation))

        # Store baseband for RDS processing (at IQ sample rate, before decimation)
        self._last_baseband = baseband

        if profiling:
            t0 = self._prof('fm_demod', t0)

        # Extract pilot (19 kHz)
        pilot, self.pilot_bpf_state = signal.lfilter(
            self.pilot_bpf, 1.0, baseband, zi=self.pilot_bpf_state
        )

        # Store pilot for RDS decoder
        self._last_pilot = pilot  # No copy - pilot is fresh from filter

        # Measure pilot level for detection
        pilot_power = np.sqrt(np.mean(pilot ** 2))
        # Smooth the pilot level measurement
        self._pilot_level = 0.9 * self._pilot_level + 0.1 * pilot_power

        if profiling:
            t0 = self._prof('pilot_bpf', t0)

        # Pilot detection and 38 kHz carrier regeneration via pilot-squaring
        # Uses trig identity: cos(2x) = 2*cos^2(x) - 1
        carrier_38k = None
        if self._pilot_level > self.pilot_threshold:
            pilot_normalized = pilot / (np.abs(pilot).max() + 1e-10)
            carrier_38k = 2 * pilot_normalized ** 2 - 1
            self._pilot_detected = True
        else:
            self._pilot_detected = False

        # Extract L+R (mono, 0-15 kHz)
        lr_sum, self.lr_sum_lpf_state = signal.lfilter(
            self.lr_sum_lpf, 1.0, baseband, zi=self.lr_sum_lpf_state
        )

        # Apply group delay compensation: delay L+R to align with L-R path
        delay = self._lr_sum_delay
        delayed = np.empty_like(lr_sum)
        delayed[:delay] = self._lr_sum_delay_buf
        delayed[delay:] = lr_sum[:-delay] if delay > 0 else lr_sum
        self._lr_sum_delay_buf = lr_sum[-delay:].copy()
        lr_sum = delayed

        if profiling:
            t0 = self._prof('lr_sum_lpf', t0)

        if self._pilot_detected:
            # Extract L-R subcarrier region (23-53 kHz)
            lr_diff_mod, self.lr_diff_bpf_state = signal.lfilter(
                self.lr_diff_bpf, 1.0, baseband, zi=self.lr_diff_bpf_state
            )

            if profiling:
                t0 = self._prof('lr_diff_bpf', t0)

            # Demodulate L-R by multiplying with 38 kHz carrier
            lr_diff_demod = lr_diff_mod * carrier_38k * 2  # *2 for DSB-SC gain

            # Lowpass filter the demodulated L-R
            lr_diff, self.lr_diff_lpf_state = signal.lfilter(
                self.lr_diff_lpf, 1.0, lr_diff_demod, zi=self.lr_diff_lpf_state
            )

            if profiling:
                t0 = self._prof('lr_diff_lpf', t0)

            # Matrix decode
            left_stereo = lr_sum + lr_diff
            right_stereo = lr_sum - lr_diff

            # Apply stereo blend based on SNR (blend to mono when noisy)
            if self.stereo_blend_enabled:
                # Calculate blend factor from SNR (smoothed for stability)
                target_blend = (self._snr_db - self.stereo_blend_low) / (self.stereo_blend_high - self.stereo_blend_low)
                target_blend = max(0.0, min(1.0, target_blend))
                # Smooth blend factor to avoid sudden changes
                self._blend_factor = 0.95 * self._blend_factor + 0.05 * target_blend

                # Blend: 0 = mono (lr_sum), 1 = full stereo
                left = self._blend_factor * left_stereo + (1.0 - self._blend_factor) * lr_sum
                right = self._blend_factor * right_stereo + (1.0 - self._blend_factor) * lr_sum
            else:
                left = left_stereo
                right = right_stereo
        else:
            # No stereo - output mono to both channels
            left = lr_sum
            right = lr_sum
            self._blend_factor = 0.0  # Track that we're in mono

        # Pilot-referenced SNR measurement
        # The 19 kHz pilot is broadcast at a fixed level (~9% of deviation),
        # so comparing pilot power to noise floor gives a consistent metric.
        if self.noise_bpf is not None:
            noise_filtered, self.noise_bpf_state = signal.lfilter(
                self.noise_bpf, 1.0, baseband, zi=self.noise_bpf_state
            )
            noise_power = np.mean(noise_filtered ** 2)

            # Pilot power from the smoothed pilot level (already RMS, square for power)
            pilot_power = self._pilot_level ** 2

            # Normalize noise to pilot bandwidth for fair comparison
            # Pilot BPF: 1 kHz, Noise BPF: 10 kHz
            noise_in_pilot_bw = noise_power * (self.pilot_bandwidth / self.noise_bandwidth)

            # Smooth the noise measurement
            self._noise_power = 0.9 * self._noise_power + 0.1 * max(noise_in_pilot_bw, 1e-12)

            # Calculate SNR in dB (pilot power vs noise in same bandwidth)
            if self._noise_power > 0 and pilot_power > 0:
                self._snr_db = 10 * np.log10(pilot_power / self._noise_power)

        if profiling:
            t0 = self._prof('noise_bpf', t0)

        # Resample to audio rate with adaptive rate control
        # Adjusts output length to compensate for clock drift between IQ source and audio card
        nominal_output = int(round(len(left) * self._nominal_ratio))
        adjusted_output = int(round(nominal_output * self._rate_adjust))
        if adjusted_output > 0:
            left = signal.resample(left, adjusted_output)
            right = signal.resample(right, adjusted_output)
        else:
            left = np.array([], dtype=np.float64)
            right = np.array([], dtype=np.float64)

        if profiling:
            t0 = self._prof('resample', t0)

        # Apply de-emphasis to each channel
        left, self.deem_state_l = signal.lfilter(
            self.deem_b, self.deem_a, left, zi=self.deem_state_l
        )
        right, self.deem_state_r = signal.lfilter(
            self.deem_b, self.deem_a, right, zi=self.deem_state_r
        )

        if profiling:
            t0 = self._prof('deemphasis', t0)

        # Scale down to provide headroom for tone controls and over-modulated stations.
        # Applied BEFORE tone controls so boosts don't cause clipping.
        # 0.65 (-3.7dB) provides headroom for +3dB bass and +3dB treble.
        left = left * 0.65
        right = right * 0.65

        # Apply tone controls (bass and treble boost)
        if self.bass_boost_enabled:
            left, self.bass_state_l = signal.lfilter(
                self.bass_b, self.bass_a, left, zi=self.bass_state_l
            )
            right, self.bass_state_r = signal.lfilter(
                self.bass_b, self.bass_a, right, zi=self.bass_state_r
            )
        if self.treble_boost_enabled:
            left, self.treble_state_l = signal.lfilter(
                self.treble_b, self.treble_a, left, zi=self.treble_state_l
            )
            right, self.treble_state_r = signal.lfilter(
                self.treble_b, self.treble_a, right, zi=self.treble_state_r
            )

        if profiling:
            t0 = self._prof('tone', t0)

        # Track peak amplitude (before limiting, for clipping detection)
        peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
        self._peak_amplitude = max(0.95 * self._peak_amplitude, peak)  # Fast attack, slow decay

        # Apply soft limiting to catch any remaining peaks.
        # After 0.65 scaling + up to +6dB tone boost, signal is typically ~0.9 max.
        # Soft limiter using tanh handles any peaks that still exceed 1.0.
        tanh_scale = np.tanh(1.5)
        left = np.tanh(left * 1.5) / tanh_scale
        right = np.tanh(right * 1.5) / tanh_scale
        # Hard clip to ensure output never exceeds ±1.0
        left = np.clip(left, -1.0, 1.0)
        right = np.clip(right, -1.0, 1.0)
        left = left.astype(np.float32)
        right = right.astype(np.float32)

        if profiling:
            self._prof('limiter', t0)
            elapsed_total = (time.perf_counter() - t_total) * 1e6
            self._profile['total'] = 0.9 * self._profile['total'] + 0.1 * elapsed_total

        return np.column_stack((left, right))

    def reset(self):
        """Reset decoder state (call when changing frequency)."""
        self.last_sample = complex(1, 0)
        self._pilot_detected = False
        self._pilot_level = 0.0
        self._last_baseband = None

        # Reset filter states
        self.pilot_bpf_state = signal.lfilter_zi(self.pilot_bpf, 1.0)
        self.lr_sum_lpf_state = signal.lfilter_zi(self.lr_sum_lpf, 1.0)
        self.lr_diff_bpf_state = signal.lfilter_zi(self.lr_diff_bpf, 1.0)
        self.lr_diff_lpf_state = signal.lfilter_zi(self.lr_diff_lpf, 1.0)
        self.deem_state_l = signal.lfilter_zi(self.deem_b, self.deem_a)
        self.deem_state_r = signal.lfilter_zi(self.deem_b, self.deem_a)
        if self.noise_bpf is not None:
            self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)

        # Reset group delay compensation buffer
        self._lr_sum_delay_buf = np.zeros(self._lr_sum_delay, dtype=np.float64)

        # Reset SNR state
        self._snr_db = 0.0
        self._noise_power = 1e-10
        self._peak_amplitude = 0.0
        self._blend_factor = 1.0  # Start at full stereo

        # Reset tone control filter states
        self.bass_state_l = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.bass_state_r = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.treble_state_l = signal.lfilter_zi(self.treble_b, self.treble_a)
        self.treble_state_r = signal.lfilter_zi(self.treble_b, self.treble_a)


class NBFMDecoder:
    """
    Narrowband FM decoder for weather radio and similar services.

    NBFM characteristics:
    - Deviation: ~5 kHz (vs 75 kHz for broadcast FM)
    - Mono only (no stereo subcarrier)
    - Audio bandwidth: ~3 kHz
    - No de-emphasis (NWS transmits without pre-emphasis)
    """

    def __init__(self, iq_sample_rate=250000, audio_sample_rate=48000,
                 deviation=5000):
        """
        Initialize NBFM decoder.

        Args:
            iq_sample_rate: Input IQ sample rate in Hz
            audio_sample_rate: Output audio sample rate in Hz
            deviation: FM deviation in Hz (5 kHz typical for NBFM)
        """
        self.iq_sample_rate = iq_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.deviation = deviation

        # State for continuous processing
        self.last_sample = complex(1, 0)

        # Adaptive rate control (same as FMStereoDecoder)
        self._rate_adjust = 1.0
        self._nominal_ratio = audio_sample_rate / iq_sample_rate

        # Design filters (all at IQ sample rate)
        nyq = iq_sample_rate / 2

        # Audio lowpass filter (3 kHz for NBFM voice)
        audio_cutoff = 3000 / nyq
        self.audio_lpf = signal.firwin(101, audio_cutoff, window='hamming')
        self.audio_lpf_state = signal.lfilter_zi(self.audio_lpf, 1.0)

        # SNR measurement state
        self._snr_db = 0.0
        self._signal_power = 0.0
        self._noise_power = 1e-10

        # Design noise measurement bandpass filter (20-30 kHz)
        # Must be well outside NBFM signal bandwidth (Carson's rule: 2*(5kHz + 3kHz) = 16kHz)
        # Using 20-30kHz to be safely outside the signal spectrum
        noise_low = 20000 / nyq
        noise_high = min(30000 / nyq, 0.95)
        if noise_high > noise_low:
            self.noise_bpf = signal.firwin(51, [noise_low, noise_high],
                                           pass_zero=False, window='hamming')
            self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)
            self.noise_bandwidth = 10000  # Hz
        else:
            self.noise_bpf = None

        # Audio bandwidth for SNR scaling
        self.audio_bandwidth = 3000  # Hz

        # Peak amplitude tracking
        self._peak_amplitude = 0.0

        # Tone controls (bass and treble boost) - same as stereo decoder
        self.bass_boost_enabled = False   # Default off for weather
        self.treble_boost_enabled = False # Default off for weather
        self._setup_tone_filters()

    def _design_low_shelf(self, fc, gain_db, fs):
        """Design low shelf biquad filter coefficients."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / 2 * np.sqrt(2)
        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)
        two_sqrt_A_alpha = 2 * sqrt_A * alpha

        b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha

        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        return b, a

    def _design_high_shelf(self, fc, gain_db, fs):
        """Design high shelf biquad filter coefficients."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / 2 * np.sqrt(2)
        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)
        two_sqrt_A_alpha = 2 * sqrt_A * alpha

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha

        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        return b, a

    def _setup_tone_filters(self):
        """Set up bass and treble shelf filters (+3dB)."""
        fs = self.audio_sample_rate
        self.bass_b, self.bass_a = self._design_low_shelf(250, 3.0, fs)
        self.bass_state_l = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.treble_b, self.treble_a = self._design_high_shelf(3500, 3.0, fs)
        self.treble_state_l = signal.lfilter_zi(self.treble_b, self.treble_a)

    @property
    def snr_db(self):
        """Return the current SNR estimate in dB."""
        return self._snr_db

    @property
    def peak_amplitude(self):
        """Return peak amplitude (before limiting)."""
        return self._peak_amplitude

    @property
    def pilot_detected(self):
        """Always False for NBFM (no stereo pilot)."""
        return False

    @property
    def stereo_blend_factor(self):
        """Always 0 for NBFM (mono only)."""
        return 0.0

    @property
    def rate_adjust(self):
        """Return current rate adjustment factor for adaptive resampling."""
        return self._rate_adjust

    @rate_adjust.setter
    def rate_adjust(self, value):
        """Set rate adjustment factor (0.99-1.01 typical range for clock drift)."""
        self._rate_adjust = max(0.98, min(1.02, value))

    @property
    def last_baseband(self):
        """Returns None - NBFM doesn't provide baseband for RDS."""
        return None

    def demodulate(self, iq_samples):
        """
        Demodulate NBFM signal from IQ samples.

        Args:
            iq_samples: numpy array of complex64 IQ samples

        Returns:
            numpy array of shape (N, 2) with mono audio duplicated to L/R
        """
        if len(iq_samples) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # FM demodulation (quadrature discriminator)
        samples = np.concatenate([[self.last_sample], iq_samples])
        self.last_sample = iq_samples[-1]

        product = samples[1:] * np.conj(samples[:-1])
        baseband = np.angle(product) * (self.iq_sample_rate / (2 * np.pi * self.deviation))

        # Audio lowpass filter (3 kHz)
        audio, self.audio_lpf_state = signal.lfilter(
            self.audio_lpf, 1.0, baseband, zi=self.audio_lpf_state
        )

        # SNR measurement
        # Measure signal power from the filtered audio
        signal_power = np.mean(audio ** 2)

        # Measure noise power in a band outside the FM signal spectrum
        if self.noise_bpf is not None:
            noise_filtered, self.noise_bpf_state = signal.lfilter(
                self.noise_bpf, 1.0, baseband, zi=self.noise_bpf_state
            )
            noise_power = np.mean(noise_filtered ** 2)
            # Scale noise from measurement bandwidth to audio bandwidth
            noise_power_scaled = noise_power * (self.audio_bandwidth / self.noise_bandwidth)

            # Smooth the measurements
            self._signal_power = 0.9 * self._signal_power + 0.1 * signal_power
            self._noise_power = 0.9 * self._noise_power + 0.1 * max(noise_power_scaled, 1e-12)

            if self._noise_power > 0:
                self._snr_db = 10 * np.log10(self._signal_power / self._noise_power)

        # Resample to audio rate with adaptive rate control
        # Adjusts output length to compensate for clock drift
        nominal_output = int(round(len(audio) * self._nominal_ratio))
        adjusted_output = int(round(nominal_output * self._rate_adjust))
        if adjusted_output > 0:
            audio = signal.resample(audio, adjusted_output)
        else:
            audio = np.array([], dtype=np.float64)

        # Scale down to provide headroom for tone controls
        audio = audio * 0.65

        # Apply tone controls
        if self.bass_boost_enabled:
            audio, self.bass_state_l = signal.lfilter(
                self.bass_b, self.bass_a, audio, zi=self.bass_state_l
            )
        if self.treble_boost_enabled:
            audio, self.treble_state_l = signal.lfilter(
                self.treble_b, self.treble_a, audio, zi=self.treble_state_l
            )

        # Track peak amplitude
        peak = np.max(np.abs(audio))
        self._peak_amplitude = max(0.95 * self._peak_amplitude, peak)

        # Apply soft limiting to catch any remaining peaks
        tanh_scale = np.tanh(1.5)
        audio = np.tanh(audio * 1.5) / tanh_scale
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        # Duplicate mono to stereo
        return np.column_stack((audio, audio))

    def reset(self):
        """Reset decoder state (call when changing frequency)."""
        self.last_sample = complex(1, 0)
        self._snr_db = 0.0
        self._signal_power = 0.0
        self._noise_power = 1e-10
        self._peak_amplitude = 0.0

        # Reset filter states
        self.audio_lpf_state = signal.lfilter_zi(self.audio_lpf, 1.0)
        if self.noise_bpf is not None:
            self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)
        self.bass_state_l = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.treble_state_l = signal.lfilter_zi(self.treble_b, self.treble_a)
