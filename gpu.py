#!/usr/bin/env python3
"""
GPU-accelerated DSP operations for pyfm.

Provides GPU-accelerated FM demodulation and polyphase resampling
using ROCm (AMD GPUs) via PyTorch, with CPU fallback.
"""

import numpy as np


class GPUFMDemodulator:
    """
    GPU-accelerated FM demodulator using the arctangent-differentiate method.

    Algorithm:
        1. phase(n) = atan2(Q(n), I(n))
        2. delta_phase(n) = phase(n) - phase(n-1)
        3. Unwrap: if |delta_phase| > pi, adjust by ±2*pi
        4. baseband = delta_phase * (sample_rate / (2*pi * deviation))

    Usage:
        demod = GPUFMDemodulator(sample_rate=250000, deviation=75000)
        baseband = demod.demodulate(iq_samples)
    """

    def __init__(self, sample_rate=250000, deviation=75000):
        self.sample_rate = sample_rate
        self.deviation = deviation
        self._last_phase = 0.0
        self._scale = sample_rate / (2 * np.pi * deviation)

        self._torch = None
        self._torch_device = None
        self._backend = 'cpu'

        self._init_rocm()

    def _init_rocm(self):
        """Initialize ROCm backend via PyTorch."""
        try:
            import torch
            self._torch = torch
            if torch.cuda.is_available():
                self._torch_device = torch.device('cuda')
                device_name = torch.cuda.get_device_name(0)
                print(f"GPUFMDemodulator: Using ROCm/HIP on {device_name}")
                self._backend = 'rocm'
            else:
                print("GPUFMDemodulator: ROCm not available, using CPU")
        except (ImportError, RuntimeError) as e:
            print(f"GPUFMDemodulator: ROCm init failed ({e}), using CPU")

    @property
    def backend(self):
        """Return the active backend name."""
        return self._backend

    def demodulate(self, iq_samples):
        if len(iq_samples) == 0:
            return np.array([], dtype=np.float32)

        if self._backend == 'rocm':
            return self._demodulate_rocm(iq_samples)
        return self._demodulate_cpu(iq_samples)

    def _demodulate_cpu(self, iq_samples):
        """CPU implementation using NumPy."""
        phase = np.arctan2(iq_samples.imag, iq_samples.real)
        phase_with_prev = np.concatenate([[self._last_phase], phase])
        delta_phase = np.diff(phase_with_prev)
        delta_phase = np.where(delta_phase > np.pi, delta_phase - 2*np.pi, delta_phase)
        delta_phase = np.where(delta_phase < -np.pi, delta_phase + 2*np.pi, delta_phase)
        self._last_phase = phase[-1]
        return (delta_phase * self._scale).astype(np.float32)

    def _demodulate_rocm(self, iq_samples):
        """ROCm/PyTorch GPU implementation."""
        torch = self._torch
        device = self._torch_device

        try:
            iq_real = torch.from_numpy(iq_samples.real.astype(np.float32)).to(device)
            iq_imag = torch.from_numpy(iq_samples.imag.astype(np.float32)).to(device)
            phase = torch.atan2(iq_imag, iq_real)
            last_phase_tensor = torch.tensor([self._last_phase], dtype=torch.float32, device=device)
            phase_with_prev = torch.cat([last_phase_tensor, phase])
            delta_phase = phase_with_prev[1:] - phase_with_prev[:-1]
            delta_phase = torch.where(delta_phase > np.pi, delta_phase - 2*np.pi, delta_phase)
            delta_phase = torch.where(delta_phase < -np.pi, delta_phase + 2*np.pi, delta_phase)
            self._last_phase = phase[-1].item()
            return (delta_phase * self._scale).cpu().numpy().astype(np.float32)

        except RuntimeError as e:
            if "HIP error" in str(e) or "invalid device function" in str(e):
                print(f"GPUFMDemodulator: ROCm kernel failed ({e}), falling back to CPU")
                self._backend = 'cpu'
                return self._demodulate_cpu(iq_samples)
            raise

    def cleanup(self):
        """Release GPU resources before exit."""
        if self._torch is not None and self._torch_device is not None:
            try:
                self._torch.cuda.synchronize()
                self._torch.cuda.empty_cache()
            except Exception:
                pass
        self._torch = None
        self._torch_device = None
        self._backend = 'cpu'

    def reset(self):
        """Reset demodulator state (call when changing frequency)."""
        self._last_phase = 0.0


class GPUResampler:
    """
    GPU-accelerated polyphase resampler.

    Implements rational resampling (up/down) using a polyphase FIR
    decomposition. The anti-aliasing filter matches scipy's
    resample_poly design exactly (Kaiser-windowed sinc, half_len=10).

    The polyphase structure avoids computing at the full upsampled
    rate. For each output sample, only one polyphase branch (phase)
    is evaluated — a dot product of ~131 filter taps with input
    samples. This maps to a precomputed gather + batched dot product,
    which is well-suited for GPU execution.

    For 312.5 kHz -> 48 kHz (up=96, down=625):
    - 8192 input samples -> 1258 output samples
    - 131 taps per phase, 96 phases
    - ~165K multiply-accumulates per channel per block

    L and R channels are batched into a single GPU transfer.
    """

    def __init__(self, up, down, n_input):
        from scipy.signal import firwin

        self.up = up
        self.down = down
        self._n_input = n_input

        # Design anti-aliasing filter (improved over scipy default)
        # half_len=16, beta=8.0 gives ~80 dB stopband (vs ~40 dB with 10/5.0)
        # Increases taps per phase from 131 to 209 — negligible GPU cost
        max_rate = max(up, down)
        half_len = 16
        n_taps = 2 * half_len * max_rate + 1
        h = firwin(n_taps, 1.0 / max_rate, window=('kaiser', 8.0))
        h = h * up  # Gain correction for upsampling

        # Polyphase decomposition:
        # Pad h to a multiple of `up`, then reshape into (up, taps_per_phase)
        # Phase p gets taps h[p], h[p+up], h[p+2*up], ...
        n_padded = int(np.ceil(len(h) / up)) * up
        h_padded = np.zeros(n_padded)
        h_padded[:len(h)] = h
        self._taps_per_phase = n_padded // up
        # bank[p, m] = h[p + m*up]  (phase p, tap m)
        self._filter_bank = h_padded.reshape(self._taps_per_phase, up).T.copy()

        # Precompute output mapping for the expected input length
        self._precompute(n_input)

        # History buffers for block-boundary continuity
        self._history_left = np.zeros(self._pad_left, dtype=np.float32)
        self._history_right = np.zeros(self._pad_left, dtype=np.float32)

        # GPU state
        self._torch = None
        self._device = None
        self._gpu_weights = None
        self._gpu_indices = None
        self._backend = 'cpu'

        self._init_rocm()

    def _precompute(self, n_in):
        """Precompute gather indices and filter weights for a given input length.

        Polyphase resampling: output[n] = sum_m h_p[m] * x[q - m]
        where p = (n * down) % up, q = (n * down) // up.

        We precompute (gather_indices, filter_weights) so that:
            output[n] = dot(filter_weights[n], input_padded[gather_indices[n]])
        """
        # Number of output samples (matches scipy convention)
        n_out = (n_in * self.up + self.down - 1) // self.down
        taps = self._taps_per_phase

        # For each output sample, compute polyphase phase and input position
        out_idx = np.arange(n_out, dtype=np.int64)
        up_pos = out_idx * self.down        # position in upsampled stream
        phases = up_pos % self.up            # which polyphase branch
        positions = up_pos // self.up        # corresponding input index

        # Gather: for output n, we need input[q], input[q-1], ..., input[q-(taps-1)]
        # Pad input on the left by (taps-1) so indices are always non-negative
        self._pad_left = taps - 1
        tap_offsets = np.arange(taps, dtype=np.int64)
        # gather[n, m] = (positions[n] - m) + pad_left
        gather = positions[:, None] - tap_offsets[None, :] + self._pad_left
        # Clamp to valid padded range
        n_padded = n_in + self._pad_left
        self._gather_indices = np.clip(gather, 0, n_padded - 1)

        # Weights: filter_bank[phase[n], m] for each output sample
        self._filter_weights = self._filter_bank[phases].astype(np.float32)

        self._n_out = n_out
        self._n_padded = n_padded

    def _init_rocm(self):
        """Upload precomputed weights and indices to GPU."""
        try:
            import torch
            self._torch = torch
            self._device = torch.device('cuda')
            self._gpu_weights = torch.from_numpy(self._filter_weights).to(self._device)
            self._gpu_indices = torch.from_numpy(self._gather_indices).to(self._device)
            self._backend = 'rocm'
        except Exception as e:
            print(f"GPUResampler: ROCm init failed ({e}), using CPU")
            self._backend = 'cpu'

    @property
    def backend(self):
        """Return active backend name."""
        return self._backend

    def resample(self, left, right):
        """
        Resample L and R channels.

        Args:
            left: numpy float array, length n_input
            right: numpy float array, length n_input

        Returns:
            tuple: (left_out, right_out) resampled to output rate
        """
        if len(left) != self._n_input:
            # Unexpected size — fall back to scipy
            from scipy import signal
            return (
                signal.resample_poly(left, self.up, self.down).astype(np.float32),
                signal.resample_poly(right, self.up, self.down).astype(np.float32),
            )

        if self._backend == 'rocm':
            return self._resample_rocm(left, right)
        else:
            return self._resample_cpu(left, right)

    def _resample_cpu(self, left, right):
        """CPU polyphase resampling via precomputed gather + dot product."""
        pad = self._pad_left
        left_padded = np.empty(self._n_padded, dtype=np.float32)
        right_padded = np.empty(self._n_padded, dtype=np.float32)
        left_padded[:pad] = self._history_left
        left_padded[pad:] = left
        right_padded[:pad] = self._history_right
        right_padded[pad:] = right

        # Save trailing samples for next block
        self._history_left = left[-pad:].copy().astype(np.float32)
        self._history_right = right[-pad:].copy().astype(np.float32)

        # Gather + weighted sum
        left_out = np.sum(
            left_padded[self._gather_indices] * self._filter_weights, axis=1
        ).astype(np.float32)
        right_out = np.sum(
            right_padded[self._gather_indices] * self._filter_weights, axis=1
        ).astype(np.float32)

        return left_out, right_out

    def _resample_rocm(self, left, right):
        """ROCm GPU polyphase resampling — batched L+R in single transfer."""
        torch = self._torch
        device = self._device

        try:
            # Pad both channels
            pad = self._pad_left
            left_padded = np.empty(self._n_padded, dtype=np.float32)
            right_padded = np.empty(self._n_padded, dtype=np.float32)
            left_padded[:pad] = self._history_left
            left_padded[pad:] = left.astype(np.float32)
            right_padded[:pad] = self._history_right
            right_padded[pad:] = right.astype(np.float32)

            # Save trailing samples for next block
            self._history_left = left[-pad:].copy().astype(np.float32)
            self._history_right = right[-pad:].copy().astype(np.float32)

            # Batch L+R into single transfer: shape (2, n_padded)
            batch = np.stack([left_padded, right_padded])
            t_batch = torch.from_numpy(batch).to(device)

            # Gather: t_batch[:, gather_indices] -> (2, n_out, taps_per_phase)
            t_gathered = t_batch[:, self._gpu_indices]

            # Weighted sum: (2, n_out, taps) * (1, n_out, taps) -> sum -> (2, n_out)
            t_out = (t_gathered * self._gpu_weights.unsqueeze(0)).sum(dim=2)

            # Transfer back to CPU
            result = t_out.cpu().numpy()
            return result[0], result[1]

        except RuntimeError as e:
            if "HIP error" in str(e) or "invalid device function" in str(e):
                print(f"GPUResampler: ROCm failed ({e}), falling back to CPU")
                self._backend = 'cpu'
                return self._resample_cpu(left, right)
            raise

    def reset(self):
        """Reset history state (call on frequency change)."""
        self._history_left = np.zeros(self._pad_left, dtype=np.float32)
        self._history_right = np.zeros(self._pad_left, dtype=np.float32)

    def cleanup(self):
        """Release GPU resources."""
        self._gpu_weights = None
        self._gpu_indices = None
        if self._torch is not None:
            try:
                self._torch.cuda.synchronize()
            except Exception:
                pass
        self._torch = None
        self._device = None
        self._backend = 'cpu'


def fm_demodulate_arctan(iq_samples, sample_rate=250000, deviation=75000):
    """
    Convenience function for one-shot FM demodulation using arctangent-differentiate.

    Stateless wrapper - for continuous streaming, use GPUFMDemodulator
    class to maintain phase continuity between blocks.
    """
    demod = GPUFMDemodulator(sample_rate, deviation)
    return demod.demodulate(iq_samples)
