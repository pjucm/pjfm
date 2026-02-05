#!/usr/bin/env python3
"""
RDS (Radio Data System) Decoder for FM Broadcasts

Uses coherent demodulation: extracts 19 kHz pilot and triples to 57 kHz carrier
via the identity cos(3θ) = 4cos³(θ) - 3cos(θ), then performs differential decoding.

Signal processing chain:
- Bandpass filtering at 57 kHz to extract RDS subcarrier
- Coherent BPSK demodulation using pilot-derived carrier
- Differential decoding (RDS uses differential encoding)
- Symbol timing recovery via Gardner TED
- Syndrome-based block synchronization
- Standard group decoding (PS, PTY, PI, RT)
"""

import numpy as np
from scipy import signal
from collections import deque


# RDS Constants
RDS_CARRIER_FREQ = 57000.0      # 57 kHz subcarrier (3 x 19 kHz pilot)
RDS_SYMBOL_RATE = 1187.5        # Symbols per second
RDS_BITS_PER_BLOCK = 26         # 16 data + 10 checkword
RDS_BLOCKS_PER_GROUP = 4        # A, B, C/C', D

# CRC generator polynomial: x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1 = 0x1B9
CRC_POLY = 0x1B9

# Block offset words (added to CRC to identify block position)
OFFSET_WORDS = {
    'A':  0x0FC,
    'B':  0x198,
    'C':  0x168,
    'Cp': 0x350,  # C' used in type B groups
    'D':  0x1B4,
}

# Error correction lookup table: maps error syndromes to (error_mask, burst_length)
# Built by computing syndromes for all burst errors up to 5 bits at all positions
# Per IEC 62106, the RDS code can correct burst errors up to 5 bits
ERROR_CORRECTION_TABLE = None  # Lazily initialized


def _build_error_correction_table():
    """
    Build syndrome lookup table for burst error correction.

    The RDS (26,16) shortened cyclic code can correct:
    - All single-bit errors (burst length 1)
    - All 2-bit burst errors (burst length 2)
    - All burst errors up to 5 bits in length

    For each burst error pattern at each position, we compute what syndrome
    it would produce. When decoding, if the error syndrome matches one of
    these, we can correct the error by XORing the error pattern.
    """
    table = {}

    # For burst errors of length 1 to 5 bits
    for burst_len in range(1, 6):
        # Error pattern is burst_len consecutive 1s
        error_pattern = (1 << burst_len) - 1  # e.g., 0b11111 for 5-bit burst

        # Try all positions where this burst can fit in a 26-bit block
        # Position 0 means MSB, position 25 means LSB
        for pos in range(26 - burst_len + 1):
            # Shift error pattern to position (from MSB)
            # pos=0 means error at bits 25-21 (MSB end)
            error_mask = error_pattern << (26 - burst_len - pos)

            # Compute syndrome for this error pattern
            error_syndrome = compute_syndrome(error_mask)

            # Only store if this syndrome isn't already in table
            # (prefer shorter bursts - they're more likely)
            if error_syndrome not in table:
                table[error_syndrome] = (error_mask, burst_len)

    return table


def get_error_correction_table():
    """Get or build the error correction lookup table."""
    global ERROR_CORRECTION_TABLE
    if ERROR_CORRECTION_TABLE is None:
        ERROR_CORRECTION_TABLE = _build_error_correction_table()
    return ERROR_CORRECTION_TABLE


def correct_block(block_26bit, expected_offset):
    """
    Attempt to correct burst errors in a received RDS block.

    Args:
        block_26bit: The 26-bit received block
        expected_offset: The expected offset word for this block position

    Returns:
        (corrected_block, burst_length) if correction successful, else (None, 0)
    """
    syndrome = compute_syndrome(block_26bit)

    # If already valid, no correction needed
    if syndrome == expected_offset:
        return (block_26bit, 0)

    # Compute error syndrome by XORing with expected offset
    error_syndrome = syndrome ^ expected_offset

    # Look up in correction table
    table = get_error_correction_table()
    if error_syndrome in table:
        error_mask, burst_len = table[error_syndrome]
        # Apply correction by XORing error pattern
        corrected = block_26bit ^ error_mask

        # Verify correction worked
        if compute_syndrome(corrected) == expected_offset:
            return (corrected, burst_len)

    # Could not correct
    return (None, 0)

# PTY (Program Type) codes for RBDS (North America)
def pi_to_callsign(pi_hex):
    """
    Decode RBDS PI code to North American call letters.

    PI codes encode US/Canadian station call signs:
    - 0x1000-0x54A7: K stations (KAAA-KZZZ)
    - 0x54A8-0x994F: W stations (WAAA-WZZZ)
    - 0x9950-0x9EFF: 3-letter calls (KEX, KOA, etc.)
    - 0xA100-0xAFFF: Canadian stations

    Returns call letters or None if not decodable.
    """
    if not pi_hex:
        return None

    try:
        pi = int(pi_hex, 16)
    except (ValueError, TypeError):
        return None

    # 4-letter K calls: KAAA (0x1000) to KZZZ (0x54A7)
    if 0x1000 <= pi <= 0x54A7:
        offset = pi - 0x1000
        l4 = offset % 26
        offset //= 26
        l3 = offset % 26
        offset //= 26
        l2 = offset
        if l2 <= 25:
            return 'K' + chr(ord('A') + l2) + chr(ord('A') + l3) + chr(ord('A') + l4)

    # 4-letter W calls: WAAA (0x54A8) to WZZZ (0x994F)
    elif 0x54A8 <= pi <= 0x994F:
        offset = pi - 0x54A8
        l4 = offset % 26
        offset //= 26
        l3 = offset % 26
        offset //= 26
        l2 = offset
        if l2 <= 25:
            return 'W' + chr(ord('A') + l2) + chr(ord('A') + l3) + chr(ord('A') + l4)

    # Could add 3-letter calls and Canadian stations here if needed

    return None


PTY_NAMES = {
    0: "None", 1: "News", 2: "Information", 3: "Sports",
    4: "Talk", 5: "Rock", 6: "Classic Rock", 7: "Adult Hits",
    8: "Soft Rock", 9: "Top 40", 10: "Country", 11: "Oldies",
    12: "Soft", 13: "Nostalgia", 14: "Jazz", 15: "Classical",
    16: "R&B", 17: "Soft R&B", 18: "Language", 19: "Religious Music",
    20: "Religious Talk", 21: "Personality", 22: "Public", 23: "College",
    24: "Spanish Talk", 25: "Spanish Music", 26: "Hip Hop",
    27: "Unassigned", 28: "Unassigned", 29: "Weather",
    30: "Emergency Test", 31: "Emergency",
}


def compute_syndrome(block_26bit):
    """
    Compute syndrome for a 26-bit RDS block.

    The syndrome is the remainder when dividing the received block
    by the CRC polynomial. For a valid block, the syndrome equals
    the offset word for that block position.
    """
    reg = 0
    for i in range(25, -1, -1):
        bit = (block_26bit >> i) & 1
        feedback = (reg >> 9) & 1
        reg = ((reg << 1) | bit) & 0x3FF
        if feedback:
            reg ^= CRC_POLY
    return reg


class RDSDecoder:
    """
    RDS decoder with coherent demodulation using pilot-derived carrier.

    Signal processing chain:
    1. Bandpass filter to extract 57 kHz RDS subcarrier
    2. Extract 19 kHz pilot, triple to 57 kHz carrier
    3. Coherent demodulation + differential decoding
    4. Low-pass filter to extract baseband
    5. AGC normalization
    6. Symbol timing recovery (Gardner TED)
    7. Block synchronization via syndrome matching
    8. Group assembly and decoding
    """

    def __init__(self, sample_rate=250000, matched_filter=False, matched_filter_len=None,
                 timing_gain_p=None, timing_gain_i=None, lpf_cutoff=None,
                 bpf_bandwidth=None, agc_alpha=None):
        """
        Initialize RDS decoder.

        Args:
            sample_rate: Sample rate of input baseband signal in Hz
            matched_filter: If True, use an FIR low-pass matched-style filter
                            instead of the IIR low-pass after demodulation.
            matched_filter_len: Override FIR tap count (default: 101)
            timing_gain_p: Proportional gain for timing recovery (default: 0.015)
            timing_gain_i: Integral gain for timing recovery (default: 0.0008)
            lpf_cutoff: Post-demod LPF cutoff in Hz (default: 800)
            bpf_bandwidth: RDS BPF half-bandwidth in Hz (default: 2000)
            agc_alpha: AGC tracking speed, 0-1 (default: 0.05, slower=more stable)
        """
        self.sample_rate = sample_rate
        self.samples_per_symbol = sample_rate / RDS_SYMBOL_RATE

        # Delay in samples for differential decoding (one symbol period)
        self._diff_delay = int(round(self.samples_per_symbol))

        nyq = sample_rate / 2

        # Design bandpass filter for 57 kHz RDS band
        # Default: ±2000 Hz (moderately tighter than old ±2400 Hz for better SNR)
        rds_bw = bpf_bandwidth if bpf_bandwidth is not None else 2000
        self._bpf_bandwidth = rds_bw
        low = (RDS_CARRIER_FREQ - rds_bw) / nyq
        high = (RDS_CARRIER_FREQ + rds_bw) / nyq
        low = max(0.01, low)
        high = min(0.99, high)
        self.bpf_taps = 201
        self.bpf_b = signal.firwin(self.bpf_taps, [low, high], pass_zero=False,
                                    window=('kaiser', 7.0))
        self.bpf_a = 1.0
        self.bpf_zi = signal.lfilter_zi(self.bpf_b, self.bpf_a) * 0

        # Design low-pass filter for baseband after demodulation
        # Default: 800 Hz (tighter than old 1200 Hz for better noise rejection)
        lpf_freq = lpf_cutoff if lpf_cutoff is not None else 800
        self._lpf_cutoff = lpf_freq
        self.lpf_b, self.lpf_a = signal.butter(2, lpf_freq / nyq, btype='low')
        self.lpf_zi = signal.lfilter_zi(self.lpf_b, self.lpf_a) * 0

        # Optional FIR low-pass (matched-style) filter
        self.matched_filter = matched_filter
        if matched_filter:
            mf_cutoff = 2400 / nyq
            self.mf_len = matched_filter_len or 101
            if self.mf_len % 2 == 0:
                self.mf_len += 1
            self.mf_b = signal.firwin(self.mf_len, mf_cutoff, window=('kaiser', 7.0))
            self.mf_a = 1.0
            self.mf_zi = signal.lfilter_zi(self.mf_b, self.mf_a) * 0
        else:
            self.mf_len = 0
            self.mf_b = None
            self.mf_a = None
            self.mf_zi = None

        # For coherent demod: use FIR filter for pilot with SAME tap count as RDS BPF
        # This ensures identical group delay for proper phase alignment
        pilot_low = 18500 / nyq
        pilot_high = 19500 / nyq
        self.pilot_bpf_b = signal.firwin(self.bpf_taps, [pilot_low, pilot_high],
                                          pass_zero=False, window=('kaiser', 7.0))
        self.pilot_bpf_a = 1.0
        self.pilot_bpf_zi = signal.lfilter_zi(self.pilot_bpf_b, self.pilot_bpf_a) * 0

        # Delay buffer for differential decoding
        self._diff_delay_buf = np.zeros(self._diff_delay)

        # Symbol timing recovery state (Gardner TED with PI loop)
        # Default gains reduced from 0.03/0.003 for better low-SNR stability
        self.timing_mu = 0.5              # Fractional sample offset within symbol (phase)
        self.timing_freq = 0.0            # Fractional accumulator for exact sample rate
        self.timing_freq_offset = 0.0     # Symbol rate offset (frequency error tracking)
        self.timing_gain_p = timing_gain_p if timing_gain_p is not None else 0.015
        self.timing_gain_i = timing_gain_i if timing_gain_i is not None else 0.0008
        self.sample_buffer = []

        # AGC parameters - slow tracking for noise immunity
        self._agc_alpha = agc_alpha if agc_alpha is not None else 0.05
        self._agc_rms = 0.1  # Initial AGC estimate

        # Bit buffer and block sync state
        self.bit_buffer = []
        self.synced = False
        self.inverted = False  # Whether signal is inverted
        self.expected_block = 0
        self.group_blocks = [0, 0, 0, 0]
        self.group_valid = [False, False, False, False]  # Track validity of each block in current group
        self.sync_confidence = 0
        self.consecutive_good = 0
        self.consecutive_bad = 0

        # Decoded RDS data
        self._pi_code = 0
        self._pi_hex = ''  # Cached hex string
        self._ps_chars = [' '] * 8
        self._ps_name = None  # Cached station name string
        self._ps_segment_seen = [False] * 4
        self._pty = 0
        self._pty_name = PTY_NAMES.get(0, "None")  # Cached PTY name string
        self._rt_chars = [' '] * 64
        self._rt_flag = 0
        self._ct_time = None  # (hour, minute, utc_offset_half_hours) or None

        # Statistics
        self.groups_received = 0
        self.blocks_received = 0    # Valid blocks (including corrected)
        self.blocks_expected = 0    # Total blocks attempted
        self.blocks_corrected = 0   # Blocks recovered via error correction
        self.crc_errors = 0         # Uncorrectable errors
        self.rds_signal_level = 0.0

        # Error correction statistics by burst length
        self.corrections_by_burst = [0, 0, 0, 0, 0, 0]  # Index 0 unused, 1-5 for burst lengths

        # Detailed diagnostics
        self.block_errors = [0, 0, 0, 0]  # Errors per block position (A, B, C, D)
        self.group_type_counts = {}        # Count of each group type received
        self.last_syndromes = []           # Last few syndromes for debugging
        self.bit_distribution = [0, 0]     # Count of 0s and 1s
        self.timing_mu_min = 0.5
        self.timing_mu_max = 0.5
        self.timing_mu_count = 0

        # Debug diagnostics buffer (ring buffer, no I/O during processing)
        self._diag_enabled = False
        self._diag_buffer = deque(maxlen=12000)  # ~10 seconds of symbols
        self._wrap_count_up = 0    # Times timing_mu wrapped 1.0 -> 0.0
        self._wrap_count_down = 0  # Times timing_mu wrapped 0.0 -> 1.0

        # Phase/amplitude diagnostics for coherent demod debugging
        self._pilot_rms = 0.0           # Smoothed pilot RMS level
        self._pilot_peak = 0.0          # Peak pilot amplitude
        self._carrier_rms = 0.0         # Smoothed 57 kHz carrier RMS
        self._baseband_rms_pre_agc = 0.0  # Baseband before AGC
        self._soft_symbols = deque(maxlen=1000)  # Soft decision values
        self._symbol_snr_estimate = 0.0  # Estimated symbol SNR in dB
        self._symbol_snr_counter = 0     # Counter for periodic symbol SNR updates
        self._phase_diag_buffer = deque(maxlen=500)  # Phase diagnostic samples

        # Timing log file for real-time analysis
        self._timing_log_file = None
        self._timing_log_counter = 0
        self._timing_log_interval = 50  # Log every N symbols (~24 Hz at 1187.5 sym/s)

    def enable_diagnostics(self, enabled=True):
        """Enable/disable diagnostic data collection and timing log."""
        self._diag_enabled = enabled
        if enabled:
            self._diag_buffer.clear()
            self._soft_symbols.clear()
            # Start timing log file
            if self._timing_log_file is None:
                self._timing_log_file = open('/tmp/rds_timing.log', 'w')
                self._timing_log_file.write("sym_num,timing_mu,timing_err,timing_err_raw,freq_off,soft_sym,advance,blk_rate\n")
                self._timing_log_counter = 0
        else:
            # Close timing log file
            if self._timing_log_file:
                self._timing_log_file.close()
                self._timing_log_file = None

    def get_diagnostics(self):
        """
        Get diagnostic data and statistics.

        Returns dict with:
        - 'samples': list of (timing_mu, timing_error, timing_freq_offset) tuples
        - 'histogram': timing_mu distribution in 10 bins
        - 'stats': min, max, mean, std of timing_mu
        - 'coherent': pilot/carrier/baseband amplitude metrics
        - 'constellation': soft symbol statistics (SNR estimate)
        """
        result = {
            'count': 0,
            'coherent': {
                'pilot_rms': self._pilot_rms,
                'pilot_peak': self._pilot_peak,
                'carrier_rms': self._carrier_rms,
                'baseband_rms_pre_agc': self._baseband_rms_pre_agc,
            },
        }

        # Timing diagnostics
        if self._diag_buffer:
            samples = list(self._diag_buffer)
            mu_values = [s[0] for s in samples]

            # Histogram of timing_mu (10 bins from 0 to 1)
            hist, bin_edges = np.histogram(mu_values, bins=10, range=(0, 1))

            result.update({
                'samples': samples,
                'count': len(samples),
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'stats': {
                    'min': min(mu_values),
                    'max': max(mu_values),
                    'mean': np.mean(mu_values),
                    'std': np.std(mu_values),
                },
                'wraps_up': self._wrap_count_up,
                'wraps_down': self._wrap_count_down,
            })

        # Constellation/soft symbol analysis
        if self._soft_symbols:
            soft = np.array(self._soft_symbols)
            # For BPSK, symbols should cluster around +1 and -1 (after AGC)
            # Symbol SNR estimate: mean |symbol| / std
            abs_soft = np.abs(soft)
            mean_amplitude = np.mean(abs_soft)
            # Deviation from ideal ±1 decision points
            positive = soft[soft > 0]
            negative = soft[soft < 0]

            if len(positive) > 10 and len(negative) > 10:
                pos_mean = np.mean(positive)
                pos_std = np.std(positive)
                neg_mean = np.mean(negative)
                neg_std = np.std(negative)

                # Eye opening: distance between clusters relative to noise
                cluster_separation = pos_mean - neg_mean
                avg_noise = (pos_std + neg_std) / 2
                if avg_noise > 1e-10:
                    symbol_snr_linear = (cluster_separation / 2) / avg_noise
                    symbol_snr_db = 20 * np.log10(max(symbol_snr_linear, 0.01))
                    self._symbol_snr_estimate = symbol_snr_db
                else:
                    symbol_snr_db = 40.0  # Very clean signal

                result['constellation'] = {
                    'positive_mean': float(pos_mean),
                    'positive_std': float(pos_std),
                    'negative_mean': float(neg_mean),
                    'negative_std': float(neg_std),
                    'cluster_separation': float(cluster_separation),
                    'symbol_snr_db': float(symbol_snr_db),
                    'mean_amplitude': float(mean_amplitude),
                    'sample_count': len(soft),
                }

                # Eye diagram: histogram of soft values (should be bimodal)
                eye_hist, eye_edges = np.histogram(soft, bins=50, range=(-2, 2))
                result['eye_histogram'] = eye_hist.tolist()
                result['eye_edges'] = eye_edges.tolist()

        return result

    def dump_diagnostics(self, filepath='/tmp/rds_diag.txt'):
        """Dump diagnostic data to file (call when NOT processing)."""
        diag = self.get_diagnostics()
        if not diag:
            return False

        with open(filepath, 'w') as f:
            f.write("RDS Phase/Amplitude Diagnostics\n")
            f.write("=" * 60 + "\n\n")

            # Coherent demodulation signal levels
            coh = diag.get('coherent', {})
            f.write("COHERENT DEMODULATION SIGNAL LEVELS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Pilot RMS:           {coh.get('pilot_rms', 0):.6f}\n")
            f.write(f"  Pilot Peak:          {coh.get('pilot_peak', 0):.6f}\n")
            f.write(f"  57kHz Carrier RMS:   {coh.get('carrier_rms', 0):.6f}\n")
            f.write(f"  Baseband RMS (pre-AGC): {coh.get('baseband_rms_pre_agc', 0):.6f}\n")
            # Pilot crest factor (peak/RMS) should be ~1.414 for a clean sinusoid
            if coh.get('pilot_rms', 0) > 1e-10:
                crest = coh.get('pilot_peak', 0) / coh.get('pilot_rms', 0)
                f.write(f"  Pilot Crest Factor:  {crest:.3f} (ideal: 1.414)\n")
            f.write("\n")

            # Constellation/symbol quality
            const = diag.get('constellation', {})
            if const:
                f.write("SYMBOL CONSTELLATION ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Positive cluster:  mean={const['positive_mean']:+.4f}  std={const['positive_std']:.4f}\n")
                f.write(f"  Negative cluster:  mean={const['negative_mean']:+.4f}  std={const['negative_std']:.4f}\n")
                f.write(f"  Cluster separation: {const['cluster_separation']:.4f}\n")
                f.write(f"  Symbol SNR:        {const['symbol_snr_db']:.1f} dB\n")
                f.write(f"  Mean |amplitude|:  {const['mean_amplitude']:.4f}\n")
                f.write(f"  Samples analyzed:  {const['sample_count']}\n")
                f.write("\n")

                # Quality assessment
                f.write("QUALITY ASSESSMENT\n")
                f.write("-" * 40 + "\n")
                snr = const['symbol_snr_db']
                if snr >= 15:
                    f.write(f"  Symbol SNR: EXCELLENT ({snr:.1f} dB) - expect >90% block rate\n")
                elif snr >= 10:
                    f.write(f"  Symbol SNR: GOOD ({snr:.1f} dB) - expect 70-90% block rate\n")
                elif snr >= 6:
                    f.write(f"  Symbol SNR: MARGINAL ({snr:.1f} dB) - expect 40-70% block rate\n")
                else:
                    f.write(f"  Symbol SNR: POOR ({snr:.1f} dB) - expect <40% block rate\n")

                # Check for phase issues (asymmetry in clusters)
                pos_abs = abs(const['positive_mean'])
                neg_abs = abs(const['negative_mean'])
                if pos_abs > 1e-6 and neg_abs > 1e-6:
                    asymmetry = abs(pos_abs - neg_abs) / max(pos_abs, neg_abs)
                    if asymmetry > 0.2:
                        f.write(f"  WARNING: Cluster asymmetry {asymmetry:.1%} - possible DC offset or phase error\n")

                # Check for amplitude imbalance
                if const['positive_std'] > 1e-6 and const['negative_std'] > 1e-6:
                    std_ratio = const['positive_std'] / const['negative_std']
                    if std_ratio > 1.5 or std_ratio < 0.67:
                        f.write(f"  WARNING: Noise asymmetry (ratio={std_ratio:.2f}) - possible phase error\n")

                f.write("\n")

            # Eye diagram histogram
            if 'eye_histogram' in diag:
                f.write("EYE DIAGRAM (symbol value histogram)\n")
                f.write("-" * 40 + "\n")
                eye = diag['eye_histogram']
                edges = diag['eye_edges']
                max_count = max(eye) if eye else 1
                for i, count in enumerate(eye):
                    lo = edges[i]
                    hi = edges[i + 1]
                    bar_len = int(40 * count / max_count) if max_count > 0 else 0
                    bar = '#' * bar_len
                    f.write(f"  {lo:+.2f} to {hi:+.2f}: {count:5d} {bar}\n")
                f.write("\n")

            # Timing diagnostics
            if diag.get('count', 0) > 0:
                f.write("TIMING RECOVERY DIAGNOSTICS\n")
                f.write("-" * 40 + "\n")
                stats = diag.get('stats', {})
                f.write(f"  Samples: {diag['count']}\n")
                f.write(f"  timing_mu: min={stats.get('min', 0):.4f} max={stats.get('max', 0):.4f} "
                        f"mean={stats.get('mean', 0):.4f} std={stats.get('std', 0):.4f}\n")
                f.write(f"  Wraps: up={diag.get('wraps_up', 0)} (late) down={diag.get('wraps_down', 0)} (early)\n\n")

                f.write("  Histogram (timing_mu distribution):\n")
                for i, count in enumerate(diag.get('histogram', [])):
                    lo = i * 0.1
                    hi = (i + 1) * 0.1
                    bar = '#' * (count // 20)
                    f.write(f"    {lo:.1f}-{hi:.1f}: {count:5d} {bar}\n")

        return True

    def process(self, baseband, use_coherent=None):
        """
        Process baseband samples and extract RDS data.

        Args:
            baseband: FM-demodulated baseband signal at sample_rate
            use_coherent: Deprecated, ignored. Coherent demod always used.

        Returns:
            Dictionary containing decoded RDS data and status
        """
        if len(baseband) == 0:
            return self._get_result()

        baseband = np.asarray(baseband, dtype=np.float64)

        # Bandpass filter to extract 57 kHz RDS signal
        rds_signal, self.bpf_zi = signal.lfilter(
            self.bpf_b, self.bpf_a, baseband, zi=self.bpf_zi
        )

        # Measure RDS signal level (RMS of filtered signal)
        rds_power = np.sqrt(np.mean(rds_signal ** 2))
        self.rds_signal_level = 0.9 * self.rds_signal_level + 0.1 * rds_power

        # Coherent BPSK demodulation using pilot-derived carrier
        bpsk_baseband = self._coherent_demod(rds_signal, baseband)

        # AGC - normalize signal amplitude for consistent symbol decisions
        # Use slow-tracking AGC to avoid noise modulation at low SNR
        rms = np.sqrt(np.mean(bpsk_baseband ** 2))
        if rms > 1e-10:
            # Slow tracking: blend current RMS with history
            self._agc_rms = (1.0 - self._agc_alpha) * self._agc_rms + self._agc_alpha * rms
            # Normalize and clip to reject impulse noise
            bpsk_baseband = np.clip(bpsk_baseband / max(self._agc_rms, 1e-10), -3.0, 3.0)

        # Symbol timing recovery and bit extraction
        self._extract_symbols(bpsk_baseband)

        # Update symbol SNR estimate periodically using soft symbol history
        self._symbol_snr_counter += 1
        if self._symbol_snr_counter >= 10:
            self._symbol_snr_counter = 0
            if len(self._soft_symbols) > 200:
                soft = np.array(self._soft_symbols)
                positive = soft[soft > 0]
                negative = soft[soft < 0]
                if len(positive) > 10 and len(negative) > 10:
                    pos_mean = np.mean(positive)
                    pos_std = np.std(positive)
                    neg_mean = np.mean(negative)
                    neg_std = np.std(negative)
                    cluster_separation = pos_mean - neg_mean
                    avg_noise = (pos_std + neg_std) / 2
                    if avg_noise > 1e-10:
                        symbol_snr_linear = (cluster_separation / 2) / avg_noise
                        self._symbol_snr_estimate = 20 * np.log10(max(symbol_snr_linear, 0.01))

        return self._get_result()

    def _coherent_demod(self, rds_signal, baseband):
        """
        Coherent BPSK demodulation using pilot-derived 57 kHz carrier.

        Uses the 19 kHz pilot to generate a phase-locked 57 kHz carrier via
        the tripling identity: cos(3θ) = 4cos³(θ) - 3cos(θ)

        RDS uses differential encoding, so after carrier mixing we multiply
        by a one-symbol-delayed version to perform differential decoding.

        IMPORTANT: Both the RDS BPF and pilot BPF are FIR filters with the
        same tap count (201), ensuring identical group delay (100 samples).
        This is critical for phase alignment when tripling the pilot.
        """
        baseband = np.asarray(baseband, dtype=np.float64)

        # Extract 19 kHz pilot using FIR BPF (same taps as RDS BPF for matched delay)
        pilot, self.pilot_bpf_zi = signal.lfilter(
            self.pilot_bpf_b, self.pilot_bpf_a, baseband, zi=self.pilot_bpf_zi
        )

        # Normalize pilot to unit amplitude for tripling formula
        pilot_rms = np.sqrt(np.mean(pilot ** 2))
        if pilot_rms < 1e-10:
            # No pilot detected - return zeros (RDS requires pilot for coherent demod)
            return np.zeros(len(rds_signal))

        # Diagnostics: track pilot quality
        if self._diag_enabled:
            self._pilot_rms = 0.9 * self._pilot_rms + 0.1 * pilot_rms
            pilot_peak = np.max(np.abs(pilot))
            self._pilot_peak = 0.9 * self._pilot_peak + 0.1 * pilot_peak

        # Normalize to unit amplitude (RMS of sinusoid = peak/sqrt(2))
        pilot_norm = pilot / (pilot_rms * np.sqrt(2))

        # Generate 57 kHz carrier using tripling identity: cos(3θ) = 4cos³(θ) - 3cos(θ)
        # This preserves phase relationship without FFT edge effects
        carrier_57k = 4 * pilot_norm**3 - 3 * pilot_norm

        # Diagnostics: track carrier quality
        if self._diag_enabled:
            carrier_rms = np.sqrt(np.mean(carrier_57k ** 2))
            self._carrier_rms = 0.9 * self._carrier_rms + 0.1 * carrier_rms

        # Mix RDS with carrier (carrier removal)
        coherent_product = rds_signal * carrier_57k

        # Differential decoding: multiply by one-symbol-delayed version
        n = len(coherent_product)
        delay = self._diff_delay

        delayed = np.zeros(n)
        if n >= delay:
            delayed[:delay] = self._diff_delay_buf
            delayed[delay:] = coherent_product[:n - delay]
        else:
            delayed[:n] = self._diff_delay_buf[:n]

        if n >= delay:
            self._diff_delay_buf = coherent_product[-delay:].copy()
        else:
            self._diff_delay_buf[:-n] = self._diff_delay_buf[n:]
            self._diff_delay_buf[-n:] = coherent_product

        diff_decoded = coherent_product * delayed

        # Low-pass filter to extract baseband
        if self.matched_filter:
            bpsk_baseband, self.mf_zi = signal.lfilter(
                self.mf_b, self.mf_a, diff_decoded, zi=self.mf_zi
            )
        else:
            bpsk_baseband, self.lpf_zi = signal.lfilter(
                self.lpf_b, self.lpf_a, diff_decoded, zi=self.lpf_zi
            )

        # Diagnostics: track pre-AGC baseband level
        if self._diag_enabled:
            bb_rms = np.sqrt(np.mean(bpsk_baseband ** 2))
            self._baseband_rms_pre_agc = 0.9 * self._baseband_rms_pre_agc + 0.1 * bb_rms

        return bpsk_baseband

    def _extract_symbols(self, samples):
        """
        Extract symbols with precise fractional sample tracking.

        Uses two-part timing:
        1. Fractional accumulator to track exact sample rate (263.158 samples/symbol)
        2. Gardner TED for fine phase adjustment
        """
        self.sample_buffer.extend(samples.tolist())

        sps = self.samples_per_symbol  # 263.157894...
        sps_int = int(sps)
        sps_frac = sps - sps_int       # 0.157894...
        half_sps = sps / 2.0

        # Need enough samples for interpolated early/center/late
        while True:
            # Use timing_mu for interpolation (phase within symbol)
            mu = self.timing_mu
            if mu < 0:
                mu = 0
            elif mu > 1:
                mu = 1

            pos_prev = mu
            pos_mid = mu + half_sps
            pos_next = mu + sps
            if len(self.sample_buffer) < int(pos_next) + 2:
                break

            def interp(pos):
                idx = int(pos)
                frac = pos - idx
                return (1 - frac) * self.sample_buffer[idx] + frac * self.sample_buffer[idx + 1]

            prev_sample = interp(pos_prev)
            mid_sample = interp(pos_mid)
            next_sample = interp(pos_next)

            # Hard decision symbol detection
            data_bit = 0 if next_sample > 0 else 1
            self.bit_distribution[data_bit] += 1

            # Diagnostics: capture soft symbol value for constellation analysis
            if self._diag_enabled:
                self._soft_symbols.append(next_sample)

            # Gardner TED for phase and frequency adjustment (PI loop)
            timing_error_raw = -(next_sample - prev_sample) * mid_sample
            norm = abs(next_sample) + abs(prev_sample) + abs(mid_sample) + 1e-10
            timing_error = timing_error_raw / norm
            # Clamp error to prevent runaway
            timing_error = max(-0.5, min(0.5, timing_error))

            # Proportional term: immediate phase correction
            self.timing_mu += self.timing_gain_p * timing_error

            # Integral term: frequency offset adaptation
            # This allows tracking stations that are slightly off the nominal 1187.5 Hz
            self.timing_freq_offset += self.timing_gain_i * timing_error
            # Clamp frequency offset to reasonable range (+/- 1.0 samples/symbol max)
            self.timing_freq_offset = max(-1.0, min(1.0, self.timing_freq_offset))

            # Advance buffer - use fractional accumulator to track exact rate
            # Apply frequency offset to adjust effective samples per symbol
            effective_sps_frac = sps_frac + self.timing_freq_offset
            self.timing_freq += effective_sps_frac
            advance = sps_int
            while self.timing_freq >= 1.0:
                advance += 1
                self.timing_freq -= 1.0
            while self.timing_freq < 0.0:
                advance -= 1
                self.timing_freq += 1.0

            # Allow timing_mu to wrap and adjust advance accordingly
            # This is critical - without wrapping, timing gets stuck at boundaries
            while self.timing_mu >= 1.0:
                self.timing_mu -= 1.0
                advance += 1  # We're late, consume extra sample
                self._wrap_count_up += 1
            while self.timing_mu < 0.0:
                self.timing_mu += 1.0
                advance -= 1  # We're early, consume fewer samples
                self._wrap_count_down += 1

            # Safety: ensure we always advance at least 1 sample
            if advance < 1:
                advance = 1

            # Diagnostic logging (cheap append to deque, no I/O)
            if self._diag_enabled:
                self._diag_buffer.append((self.timing_mu, timing_error, self.timing_freq_offset))

                # Periodic file logging for timing analysis
                self._timing_log_counter += 1
                if self._timing_log_file and self._timing_log_counter % self._timing_log_interval == 0:
                    blk_rate = self.blocks_received / max(1, self.blocks_expected)
                    self._timing_log_file.write(
                        f"{self._timing_log_counter},{self.timing_mu:.4f},{timing_error:.6f},"
                        f"{timing_error_raw:.6f},{self.timing_freq_offset:.6f},"
                        f"{next_sample:.4f},{advance},{blk_rate:.3f}\n"
                    )
                    self._timing_log_file.flush()

            # Track statistics (after wrapping, so we see actual operating range)
            self.timing_mu_min = min(self.timing_mu_min, self.timing_mu)
            self.timing_mu_max = max(self.timing_mu_max, self.timing_mu)
            self.timing_mu_count += 1
            if self.timing_mu_count >= 1187:
                self.timing_mu_min = self.timing_mu
                self.timing_mu_max = self.timing_mu
                self.timing_mu_count = 1

            self.sample_buffer = self.sample_buffer[advance:]
            self.bit_buffer.append(data_bit)

            if len(self.bit_buffer) >= RDS_BITS_PER_BLOCK:
                self._process_bits()

    def _process_bits(self):
        """Process accumulated bits to find and decode blocks."""
        if len(self.bit_buffer) < RDS_BITS_PER_BLOCK:
            return

        if not self.synced:
            self._search_sync()
        else:
            self._decode_block()

    def _search_sync(self):
        """Search for block sync by checking syndromes."""
        if len(self.bit_buffer) < RDS_BITS_PER_BLOCK:
            return

        block = self._bits_to_int(self.bit_buffer[:RDS_BITS_PER_BLOCK])

        # Try normal and inverted
        for invert in [False, True]:
            check_block = block if not invert else (block ^ 0x3FFFFFF)
            check_syndrome = compute_syndrome(check_block)

            for block_name, offset in OFFSET_WORDS.items():
                if check_syndrome == offset:
                    block_idx = {'A': 0, 'B': 1, 'C': 2, 'Cp': 2, 'D': 3}[block_name]
                    self.expected_block = (block_idx + 1) % 4
                    self.group_blocks[block_idx] = (check_block >> 10) & 0xFFFF
                    self.synced = True
                    self.inverted = invert  # Remember if we need to invert
                    self.sync_confidence = 1
                    self.consecutive_good = 1
                    self.consecutive_bad = 0
                    self.bit_buffer = self.bit_buffer[RDS_BITS_PER_BLOCK:]
                    return

        # No sync, slide by one bit
        self.bit_buffer = self.bit_buffer[1:]

    def _decode_block(self):
        """Decode a block when synchronized."""
        if len(self.bit_buffer) < RDS_BITS_PER_BLOCK:
            return

        block = self._bits_to_int(self.bit_buffer[:RDS_BITS_PER_BLOCK])
        self.bit_buffer = self.bit_buffer[RDS_BITS_PER_BLOCK:]

        # Apply inversion if we synced with inverted polarity
        if self.inverted:
            block = block ^ 0x3FFFFFF

        syndrome = compute_syndrome(block)

        block_names = ['A', 'B', 'C', 'D']
        expected_name = block_names[self.expected_block]
        expected_offset = OFFSET_WORDS[expected_name]

        valid = (syndrome == expected_offset)
        corrected = False
        burst_len = 0

        # For block C, also try C' offset
        if not valid and self.expected_block == 2:
            valid = (syndrome == OFFSET_WORDS['Cp'])

        # If still not valid, attempt error correction
        if not valid:
            # Try primary offset first
            corrected_block, burst_len = correct_block(block, expected_offset)

            # For block C, also try C' offset if primary failed
            if corrected_block is None and self.expected_block == 2:
                corrected_block, burst_len = correct_block(block, OFFSET_WORDS['Cp'])

            if corrected_block is not None:
                block = corrected_block
                valid = True
                corrected = True

        # Reset validity tracking at start of new group
        if self.expected_block == 0:
            self.group_valid = [False, False, False, False]

        self.blocks_expected += 1

        if valid:
            data = (block >> 10) & 0xFFFF
            self.group_blocks[self.expected_block] = data
            self.group_valid[self.expected_block] = True
            self.blocks_received += 1
            self.consecutive_good += 1
            self.consecutive_bad = 0
            self.sync_confidence = min(10, self.sync_confidence + 1)

            # Track correction statistics
            if corrected:
                self.blocks_corrected += 1
                if 1 <= burst_len <= 5:
                    self.corrections_by_burst[burst_len] += 1

            # Only decode if we have all 4 valid blocks
            if self.expected_block == 3 and all(self.group_valid):
                self._decode_group()
        else:
            self.crc_errors += 1
            self.block_errors[self.expected_block] += 1
            self.group_valid[self.expected_block] = False
            self.consecutive_bad += 1
            self.consecutive_good = 0
            self.sync_confidence = max(0, self.sync_confidence - 1)
            # Track recent syndromes for debugging
            self.last_syndromes.append((self.expected_block, syndrome))
            if len(self.last_syndromes) > 20:
                self.last_syndromes.pop(0)

            if self.consecutive_bad >= 25:
                # Try to re-sync at current position before giving up
                self.synced = False
                self.sync_confidence = 0
                return

        self.expected_block = (self.expected_block + 1) % 4

    def _bits_to_int(self, bits):
        """Convert list of bits to integer, MSB first."""
        val = 0
        for b in bits:
            val = (val << 1) | (b & 1)
        return val

    def _decode_group(self):
        """Decode a complete RDS group."""
        self.groups_received += 1

        block_a = self.group_blocks[0]
        block_b = self.group_blocks[1]
        block_c = self.group_blocks[2]
        block_d = self.group_blocks[3]

        if block_a != self._pi_code:
            self._pi_code = block_a
            self._pi_hex = f'{block_a:04X}'

        group_type = (block_b >> 12) & 0x0F
        version = (block_b >> 11) & 0x01
        pty = (block_b >> 5) & 0x1F
        if pty != self._pty:
            self._pty = pty
            self._pty_name = PTY_NAMES.get(pty, f"Unknown ({pty})")

        # Track group type distribution
        gt_key = f"{group_type}{'A' if version == 0 else 'B'}"
        self.group_type_counts[gt_key] = self.group_type_counts.get(gt_key, 0) + 1

        if group_type == 0:
            self._decode_group_0(block_b, block_c, block_d, version)
        elif group_type == 2:
            self._decode_group_2(block_b, block_c, block_d, version)
        elif group_type == 4 and version == 0:
            self._decode_group_4a(block_b, block_c, block_d)

    def _decode_group_0(self, block_b, block_c, block_d, version):
        """Decode Group 0A/0B: Program Service name."""
        segment = block_b & 0x03
        char1 = (block_d >> 8) & 0xFF
        char2 = block_d & 0xFF

        pos = segment * 2
        changed = False
        if 0x20 <= char1 <= 0x7E:
            new_char = chr(char1)
            if self._ps_chars[pos] != new_char:
                self._ps_chars[pos] = new_char
                changed = True
        if 0x20 <= char2 <= 0x7E:
            new_char = chr(char2)
            if self._ps_chars[pos + 1] != new_char:
                self._ps_chars[pos + 1] = new_char
                changed = True

        was_complete = all(self._ps_segment_seen)
        self._ps_segment_seen[segment] = True
        now_complete = all(self._ps_segment_seen)

        # Rebuild cached name if chars changed, or if we just completed all segments
        if (changed or (now_complete and not was_complete)) and now_complete:
            name = ''.join(self._ps_chars).strip()
            self._ps_name = name if name else None

    def _decode_group_2(self, block_b, block_c, block_d, version):
        """Decode Group 2A/2B: RadioText."""
        text_flag = (block_b >> 4) & 0x01
        segment = block_b & 0x0F

        if text_flag != self._rt_flag:
            self._rt_flag = text_flag
            self._rt_chars = [' '] * 64

        if version == 0:
            pos = segment * 4
            chars = [
                (block_c >> 8) & 0xFF,
                block_c & 0xFF,
                (block_d >> 8) & 0xFF,
                block_d & 0xFF,
            ]
            for i, c in enumerate(chars):
                if pos + i < 64:
                    if 0x20 <= c <= 0x7E:
                        self._rt_chars[pos + i] = chr(c)
                    elif c == 0x0D:
                        self._rt_chars[pos + i] = ' '
        else:
            pos = segment * 2
            char1 = (block_d >> 8) & 0xFF
            char2 = block_d & 0xFF
            if pos < 64 and 0x20 <= char1 <= 0x7E:
                self._rt_chars[pos] = chr(char1)
            if pos + 1 < 64 and 0x20 <= char2 <= 0x7E:
                self._rt_chars[pos + 1] = chr(char2)

    def _decode_group_4a(self, block_b, block_c, block_d):
        """
        Decode Group 4A: Clock Time and Date (CT).

        Format:
        - MJD (Modified Julian Day): 17 bits spanning block_b[1:0] and block_c[15:1]
        - Hour: 5 bits spanning block_c[0] and block_d[15:12]
        - Minute: 6 bits in block_d[11:6]
        - UTC offset: 6 bits in block_d[5:0], signed (half-hours from UTC)
        """
        # Extract MJD (17 bits)
        mjd = ((block_b & 0x03) << 15) | ((block_c >> 1) & 0x7FFF)

        # Extract time
        hour = ((block_c & 0x01) << 4) | ((block_d >> 12) & 0x0F)
        minute = (block_d >> 6) & 0x3F
        offset = block_d & 0x3F

        # Sign-extend offset (6-bit signed value, half-hours from UTC)
        if offset & 0x20:
            offset = offset - 64  # Convert to signed

        # Validate ranges
        if hour < 24 and minute < 60 and -24 <= offset <= 24:
            self._ct_time = (hour, minute, offset)

    def _get_result(self):
        """Build result dictionary."""
        # Calculate block success rate
        if self.blocks_expected > 0:
            block_rate = self.blocks_received / self.blocks_expected
        else:
            block_rate = 0.0

        # Calculate bit balance (should be ~50/50 for random data)
        total_bits = self.bit_distribution[0] + self.bit_distribution[1]
        if total_bits > 0:
            bit_balance = self.bit_distribution[0] / total_bits
        else:
            bit_balance = 0.5

        return {
            'pi_code': self._pi_code,
            'pi_hex': self._pi_hex,
            'station_name': self.station_name,
            'program_type': self._pty_name,
            'pty_code': self._pty,
            'radio_text': self.radio_text,
            'clock_time': self.clock_time,
            'synced': self.synced and self.sync_confidence >= 3,
            'groups_received': self.groups_received,
            'blocks_received': self.blocks_received,
            'blocks_expected': self.blocks_expected,
            'blocks_corrected': self.blocks_corrected,
            'block_rate': block_rate,
            'crc_errors': self.crc_errors,
            'signal_level': self.rds_signal_level,
            'lock_level': self.sync_confidence / 10.0,
            # Diagnostics
            'block_errors': self.block_errors.copy(),
            'group_types': dict(self.group_type_counts),
            'bit_balance': bit_balance,
            'samples_per_symbol': self.samples_per_symbol,
            'sample_rate': self.sample_rate,
            'timing_range': (self.timing_mu_min, self.timing_mu_max),
            'timing_freq': self.timing_freq,
            'timing_freq_offset': self.timing_freq_offset,
            # Error correction statistics
            'corrections_by_burst': self.corrections_by_burst[1:].copy(),  # Burst lengths 1-5
            # Coherent demod diagnostics (always available when diagnostics enabled)
            'pilot_rms': self._pilot_rms,
            'carrier_rms': self._carrier_rms,
            'baseband_rms': self._baseband_rms_pre_agc,
            'symbol_snr_db': self._symbol_snr_estimate,
        }

    @property
    def station_name(self):
        """Get decoded station name (PS) only when complete (cached to prevent UI flicker)."""
        # Return cached name - it's updated in _decode_group_0 when chars change
        if all(self._ps_segment_seen):
            return self._ps_name
        return None

    @property
    def program_type(self):
        """Get program type as string (cached to prevent UI flicker)."""
        return self._pty_name

    @property
    def radio_text(self):
        """Get decoded RadioText if available."""
        text = ''.join(self._rt_chars).strip()
        return text if text else None

    @property
    def clock_time(self):
        """Get decoded clock time as formatted string, or None if not received."""
        if self._ct_time is None:
            return None
        hour, minute, offset = self._ct_time
        # Format offset as +/-HH:MM
        offset_hours = offset // 2
        offset_half = (offset % 2) * 30
        if offset >= 0:
            offset_str = f"+{offset_hours}:{offset_half:02d}"
        else:
            offset_str = f"{offset_hours}:{abs(offset_half):02d}"
        return f"{hour:02d}:{minute:02d} UTC{offset_str}"

    def reset(self):
        """Reset decoder state."""
        # Filter states
        self.bpf_zi = signal.lfilter_zi(self.bpf_b, self.bpf_a) * 0
        self.lpf_zi = signal.lfilter_zi(self.lpf_b, self.lpf_a) * 0
        self.pilot_bpf_zi = signal.lfilter_zi(self.pilot_bpf_b, self.pilot_bpf_a) * 0
        if self.matched_filter:
            self.mf_zi = signal.lfilter_zi(self.mf_b, self.mf_a) * 0

        # Delay buffer for differential decoding
        self._diff_delay_buf = np.zeros(self._diff_delay)

        # Timing recovery
        self.timing_mu = 0.5
        self.timing_freq = 0.0
        self.timing_freq_offset = 0.0
        self.sample_buffer = []
        self.timing_mu_min = 0.5
        self.timing_mu_max = 0.5
        self.timing_mu_count = 0

        # Block sync
        self.bit_buffer = []
        self.synced = False
        self.inverted = False
        self.expected_block = 0
        self.group_blocks = [0, 0, 0, 0]
        self.group_valid = [False, False, False, False]
        self.sync_confidence = 0
        self.consecutive_good = 0
        self.consecutive_bad = 0

        # Decoded data
        self._pi_code = 0
        self._pi_hex = ''
        self._ps_chars = [' '] * 8
        self._ps_name = None
        self._ps_segment_seen = [False] * 4
        self._pty = 0
        self._pty_name = PTY_NAMES.get(0, "None")
        self._rt_chars = [' '] * 64
        self._rt_flag = 0
        self._ct_time = None

        # Statistics
        self.groups_received = 0
        self.blocks_received = 0
        self.blocks_expected = 0
        self.blocks_corrected = 0
        self.crc_errors = 0
        self.rds_signal_level = 0.0
        self.block_errors = [0, 0, 0, 0]
        self.corrections_by_burst = [0, 0, 0, 0, 0, 0]
        self.group_type_counts = {}
        self.last_syndromes = []
        self.bit_distribution = [0, 0]

        # AGC state
        self._agc_rms = 0.1

        # Diagnostics
        self._diag_buffer.clear()
        self._wrap_count_up = 0
        self._wrap_count_down = 0
        self._pilot_rms = 0.0
        self._pilot_peak = 0.0
        self._carrier_rms = 0.0
        self._baseband_rms_pre_agc = 0.0
        self._soft_symbols.clear()
        self._symbol_snr_estimate = 0.0
        self._symbol_snr_counter = 0
        self._phase_diag_buffer.clear()
        # Close timing log file if open
        if self._timing_log_file:
            self._timing_log_file.close()
            self._timing_log_file = None
        self._timing_log_counter = 0
