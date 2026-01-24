#!/usr/bin/env python3
"""
Software FM Stereo Demodulator

Provides FM stereo demodulation from IQ samples using quadrature demodulation
with 19 kHz pilot detection and L-R subcarrier decoding.
"""

import numpy as np
from scipy import signal


class PilotPLL:
    """
    Phase-Locked Loop for FM stereo pilot tracking.

    Tracks the 19 kHz pilot tone and generates coherent
    19 kHz and 38 kHz carriers for stereo decoding.

    Uses a 2nd-order Type 2 PLL with proportional + integral
    loop filter for smooth tracking and phase coherency.
    """

    def __init__(self, sample_rate, center_freq=19000,
                 bandwidth=100, damping=0.707):
        """
        Initialize PLL.

        Args:
            sample_rate: Input sample rate in Hz
            center_freq: Pilot frequency in Hz (19000)
            bandwidth: Loop bandwidth in Hz (affects tracking speed vs noise)
            damping: Damping factor (0.707 = critical damping)
        """
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.damping = damping

        # Loop filter coefficients (2nd-order Type 2)
        omega_n = 2 * np.pi * bandwidth
        self.Kp = 2 * damping * omega_n
        self.Ki = omega_n ** 2

        # State variables
        self.phase = 0.0
        self.freq_offset = 0.0
        self.integrator = 0.0
        self.locked = False

        # Precompute constants
        self._dt = 1.0 / sample_rate
        self._omega_0 = 2 * np.pi * center_freq

    def process(self, pilot_signal):
        """
        Process pilot signal through PLL.

        Args:
            pilot_signal: Filtered 19 kHz pilot samples (numpy array)

        Returns:
            tuple: (carrier_19k, carrier_38k, locked)
                - carrier_19k: Coherent 19 kHz carrier
                - carrier_38k: Coherent 38 kHz carrier for L-R demod
                - locked: True if PLL is locked to pilot
        """
        n = len(pilot_signal)
        carrier_19k = np.zeros(n, dtype=np.float64)
        carrier_38k = np.zeros(n, dtype=np.float64)

        # Local copies for speed
        phase = self.phase
        integrator = self.integrator
        freq_offset = self.freq_offset
        dt = self._dt
        omega_0 = self._omega_0
        Kp = self.Kp
        Ki = self.Ki

        for i in range(n):
            # NCO outputs
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            carrier_19k[i] = cos_phase
            carrier_38k[i] = np.cos(2 * phase)

            # Phase detector: multiply input by quadrature
            # For cos input, sin output gives phase error
            phase_error = pilot_signal[i] * sin_phase

            # Loop filter (PI controller)
            integrator += phase_error * Ki * dt
            freq_offset = Kp * phase_error + integrator

            # Update NCO phase
            phase += (omega_0 + freq_offset) * dt

            # Wrap phase to [-pi, pi] to prevent overflow
            while phase > np.pi:
                phase -= 2 * np.pi
            while phase < -np.pi:
                phase += 2 * np.pi

        # Store state for next block
        self.phase = phase
        self.integrator = integrator
        self.freq_offset = freq_offset

        # Lock detection: frequency offset should be small when locked
        # Within 50 Hz of center = locked
        self.locked = abs(freq_offset) < 2 * np.pi * 50

        return carrier_19k, carrier_38k, self.locked

    def reset(self):
        """Reset PLL state (call when changing frequency)."""
        self.phase = 0.0
        self.freq_offset = 0.0
        self.integrator = 0.0
        self.locked = False


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
        self.pilot_phase = 0.0

        # Calculate resampling ratio
        from math import gcd
        iq_int = int(iq_sample_rate)
        audio_int = int(audio_sample_rate)
        g = gcd(iq_int, audio_int)
        self.resample_up = audio_int // g
        self.resample_down = iq_int // g

        # Design filters (all at IQ sample rate)
        nyq = iq_sample_rate / 2

        # Pilot bandpass filter (18.5-19.5 kHz) - reduced taps for speed
        pilot_low = 18500 / nyq
        pilot_high = 19500 / nyq
        self.pilot_bpf = signal.firwin(101, [pilot_low, pilot_high],
                                        pass_zero=False, window='hamming')
        self.pilot_bpf_state = signal.lfilter_zi(self.pilot_bpf, 1.0)

        # L+R lowpass filter (15 kHz)
        lr_sum_cutoff = 15000 / nyq
        self.lr_sum_lpf = signal.firwin(51, lr_sum_cutoff, window='hamming')
        self.lr_sum_lpf_state = signal.lfilter_zi(self.lr_sum_lpf, 1.0)

        # L-R bandpass filter (23-53 kHz) - reduced taps for speed
        lr_diff_low = 23000 / nyq
        lr_diff_high = min(53000 / nyq, 0.95)  # Stay below Nyquist
        self.lr_diff_bpf = signal.firwin(101, [lr_diff_low, lr_diff_high],
                                          pass_zero=False, window='hamming')
        self.lr_diff_bpf_state = signal.lfilter_zi(self.lr_diff_bpf, 1.0)

        # L-R lowpass filter after demodulation (15 kHz)
        self.lr_diff_lpf = signal.firwin(51, lr_sum_cutoff, window='hamming')
        self.lr_diff_lpf_state = signal.lfilter_zi(self.lr_diff_lpf, 1.0)

        # De-emphasis filter (at output audio rate)
        # Use matched-pole first-order IIR for accurate 75µs time constant
        fs = audio_sample_rate
        a = np.exp(-1.0 / (deemphasis * fs))
        self.deem_b = np.array([1.0 - a])
        self.deem_a = np.array([1.0, -a])
        self.deem_state_l = signal.lfilter_zi(self.deem_b, self.deem_a)
        self.deem_state_r = signal.lfilter_zi(self.deem_b, self.deem_a)

        # Design noise measurement bandpass filter (75-95 kHz)
        noise_low = 75000 / nyq
        noise_high = min(95000 / nyq, 0.95)
        if noise_high > noise_low:
            self.noise_bpf = signal.firwin(51, [noise_low, noise_high],
                                           pass_zero=False, window='hamming')
            self.noise_bpf_state = signal.lfilter_zi(self.noise_bpf, 1.0)
            self.noise_bandwidth = 20000  # Hz
        else:
            self.noise_bpf = None

        # SNR measurement state
        self._snr_db = 0.0
        self._signal_power = 0.0
        self._noise_power = 1e-10
        # Audio bandwidth: 15 kHz for mono-compatible (L+R), or 53 kHz for full stereo
        self.mono_bandwidth = 15000    # L+R only
        self.stereo_bandwidth = 53000  # Full stereo (L+R + L-R)

        # Peak amplitude tracking (for clipping detection)
        self._peak_amplitude = 0.0

        # Store last baseband for RDS processing
        self._last_baseband = None
        self._last_pilot = None

        # PLL for pilot tracking (optional, default enabled)
        # Provides cleaner 38 kHz carrier with better phase coherency
        self.use_pll = True
        self.pilot_pll = PilotPLL(
            sample_rate=iq_sample_rate,
            center_freq=19000,
            bandwidth=100,    # 100 Hz loop bandwidth
            damping=0.707     # Critical damping
        )

        # Tone controls (bass and treble boost)
        self.bass_boost_enabled = True    # Default on
        self.treble_boost_enabled = True  # Default on
        self._setup_tone_filters()

        # Stereo blend settings (blend to mono when SNR is low)
        self.stereo_blend_enabled = True
        self.stereo_blend_low = 15.0    # Below this SNR: full mono
        self.stereo_blend_high = 30.0   # Above this SNR: full stereo
        self._blend_factor = 1.0        # Current blend (0=mono, 1=stereo)

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

        # FM demodulation (quadrature discriminator)
        samples = np.concatenate([[self.last_sample], iq_samples])
        self.last_sample = iq_samples[-1]

        product = samples[1:] * np.conj(samples[:-1])
        baseband = np.angle(product) * (self.iq_sample_rate / (2 * np.pi * self.deviation))

        # Store baseband for RDS processing (at IQ sample rate, before decimation)
        # No copy needed - baseband is a fresh array from np.angle() * scalar
        self._last_baseband = baseband

        # Extract pilot tone (19 kHz)
        pilot, self.pilot_bpf_state = signal.lfilter(
            self.pilot_bpf, 1.0, baseband, zi=self.pilot_bpf_state
        )

        # Store pilot for RDS decoder
        self._last_pilot = pilot  # No copy - pilot is fresh from lfilter

        # Measure pilot level for detection
        pilot_power = np.sqrt(np.mean(pilot ** 2))
        # Smooth the pilot level measurement
        self._pilot_level = 0.9 * self._pilot_level + 0.1 * pilot_power

        # Pilot detection: require both RMS threshold AND PLL lock (if using PLL)
        # This prevents false positives from broadband noise on blank stations
        carrier_38k = None
        if self._pilot_level > self.pilot_threshold:
            if self.use_pll:
                # Run PLL to check for coherent tone lock
                _, carrier_38k, pll_locked = self.pilot_pll.process(pilot)
                self._pilot_detected = pll_locked
            else:
                # Legacy mode: use RMS threshold only
                pilot_normalized = pilot / (np.abs(pilot).max() + 1e-10)
                carrier_38k = 2 * pilot_normalized ** 2 - 1  # cos(2x) = 2cos^2(x) - 1
                self._pilot_detected = True
        else:
            self._pilot_detected = False

        # Extract L+R (mono, 0-15 kHz)
        lr_sum, self.lr_sum_lpf_state = signal.lfilter(
            self.lr_sum_lpf, 1.0, baseband, zi=self.lr_sum_lpf_state
        )

        if self._pilot_detected:
            # Extract L-R subcarrier region (23-53 kHz)
            lr_diff_mod, self.lr_diff_bpf_state = signal.lfilter(
                self.lr_diff_bpf, 1.0, baseband, zi=self.lr_diff_bpf_state
            )

            # Demodulate L-R by multiplying with 38 kHz carrier
            lr_diff_demod = lr_diff_mod * carrier_38k * 2  # *2 for DSB-SC gain

            # Lowpass filter the demodulated L-R
            lr_diff, self.lr_diff_lpf_state = signal.lfilter(
                self.lr_diff_lpf, 1.0, lr_diff_demod, zi=self.lr_diff_lpf_state
            )

            # Matrix decode
            left_stereo = lr_sum + lr_diff
            right_stereo = lr_sum - lr_diff

            # SNR measurement for stereo (use full stereo bandwidth)
            signal_power = np.mean(lr_sum ** 2) + np.mean(lr_diff ** 2)
            audio_bandwidth = self.stereo_bandwidth

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

            # SNR measurement for mono (L+R only)
            signal_power = np.mean(lr_sum ** 2)
            audio_bandwidth = self.mono_bandwidth

        # Measure noise power (in high frequency band above all subcarriers)
        if self.noise_bpf is not None:
            noise_filtered, self.noise_bpf_state = signal.lfilter(
                self.noise_bpf, 1.0, baseband, zi=self.noise_bpf_state
            )
            noise_power = np.mean(noise_filtered ** 2)
            # Scale noise to audio bandwidth (noise power density * bandwidth)
            noise_power_scaled = noise_power * (audio_bandwidth / self.noise_bandwidth)

            # Smooth the measurements
            self._signal_power = 0.9 * self._signal_power + 0.1 * signal_power
            self._noise_power = 0.9 * self._noise_power + 0.1 * max(noise_power_scaled, 1e-12)

            # Calculate SNR in dB
            if self._noise_power > 0:
                self._snr_db = 10 * np.log10(self._signal_power / self._noise_power)

        # Resample to audio rate
        left = signal.resample_poly(left, self.resample_up, self.resample_down)
        right = signal.resample_poly(right, self.resample_up, self.resample_down)

        # Apply de-emphasis to each channel
        left, self.deem_state_l = signal.lfilter(
            self.deem_b, self.deem_a, left, zi=self.deem_state_l
        )
        right, self.deem_state_r = signal.lfilter(
            self.deem_b, self.deem_a, right, zi=self.deem_state_r
        )

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

        # Track peak amplitude (before limiting, for clipping detection)
        peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
        self._peak_amplitude = max(0.95 * self._peak_amplitude, peak)  # Fast attack, slow decay

        # Apply soft limiting to prevent harsh clipping distortion
        # Scale down slightly to provide headroom for over-modulated stations
        left = left * 0.7
        right = right * 0.7
        # Soft limiter using tanh (smooth saturation instead of hard clipping)
        tanh_scale = np.tanh(1.5)
        left = np.tanh(left * 1.5) / tanh_scale
        right = np.tanh(right * 1.5) / tanh_scale
        # Hard clip to ensure output never exceeds ±1.0 (tanh asymptote is 1.105)
        left = np.clip(left, -1.0, 1.0)
        right = np.clip(right, -1.0, 1.0)
        left = left.astype(np.float32)
        right = right.astype(np.float32)

        return np.column_stack((left, right))

    def set_carrier_mode(self, use_pll=True):
        """
        Set carrier regeneration mode.

        Args:
            use_pll: True for PLL-based (default), False for pilot-squaring
        """
        self.use_pll = use_pll
        if use_pll:
            self.pilot_pll.reset()

    def set_bass_boost(self, enabled):
        """Enable or disable +3dB bass boost at 250 Hz."""
        self.bass_boost_enabled = enabled

    def set_treble_boost(self, enabled):
        """Enable or disable +3dB treble boost at 3.5 kHz."""
        self.treble_boost_enabled = enabled

    def set_tone_controls(self, bass=True, treble=True):
        """
        Set both tone controls at once.

        Args:
            bass: Enable bass boost (+3dB at 250 Hz)
            treble: Enable treble boost (+3dB at 3.5 kHz)
        """
        self.bass_boost_enabled = bass
        self.treble_boost_enabled = treble

    def reset(self):
        """Reset decoder state (call when changing frequency)."""
        self.last_sample = complex(1, 0)
        self._pilot_detected = False
        self._pilot_level = 0.0
        self.pilot_phase = 0.0
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

        # Reset PLL state
        self.pilot_pll.reset()

        # Reset SNR state
        self._snr_db = 0.0
        self._signal_power = 0.0
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

        # Calculate resampling ratio
        from math import gcd
        iq_int = int(iq_sample_rate)
        audio_int = int(audio_sample_rate)
        g = gcd(iq_int, audio_int)
        self.resample_up = audio_int // g
        self.resample_down = iq_int // g

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

        # Resample to audio rate
        audio = signal.resample_poly(audio, self.resample_up, self.resample_down)

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

        # Apply soft limiting
        audio = audio * 0.7
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


if __name__ == "__main__":
    # Test with synthetic FM signal
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Parameters
    iq_rate = 250000
    audio_rate = 48000
    duration = 0.1  # 100ms test
    audio_freq = 1000  # 1 kHz test tone
    deviation = 75000

    # Generate test audio (1 kHz sine wave)
    t_audio = np.arange(int(duration * iq_rate)) / iq_rate
    audio_in = np.sin(2 * np.pi * audio_freq * t_audio)

    # FM modulate
    phase = 2 * np.pi * deviation * np.cumsum(audio_in) / iq_rate
    iq_signal = np.exp(1j * phase).astype(np.complex64)

    # Demodulate using stereo decoder (handles mono signals gracefully)
    demod = FMStereoDecoder(iq_rate, audio_rate, deviation)
    audio_out = demod.demodulate(iq_signal)

    print(f"Input: {len(iq_signal)} IQ samples")
    print(f"Output: {len(audio_out)} audio samples (stereo)")
    print(f"Audio range L: [{audio_out[:, 0].min():.3f}, {audio_out[:, 0].max():.3f}]")
    print(f"Audio range R: [{audio_out[:, 1].min():.3f}, {audio_out[:, 1].max():.3f}]")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Input audio (first few ms)
    samples_to_show = int(0.01 * iq_rate)
    axes[0].plot(t_audio[:samples_to_show] * 1000, audio_in[:samples_to_show])
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Input Audio (1 kHz tone)')
    axes[0].grid(True)

    # Output audio (left channel)
    t_out = np.arange(len(audio_out)) / audio_rate
    samples_out = int(0.01 * audio_rate)
    axes[1].plot(t_out[:samples_out] * 1000, audio_out[:samples_out, 0], label='Left')
    axes[1].plot(t_out[:samples_out] * 1000, audio_out[:samples_out, 1], label='Right', alpha=0.7)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Demodulated Audio (Stereo)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('demod_test.png')
    print("Test plot saved to demod_test.png")
