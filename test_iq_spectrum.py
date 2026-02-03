#!/usr/bin/env python3
"""
Analyze I/Q spectrum to verify signal presence

Check if the I/Q data contains actual FM signal or just noise.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/philj/dev/pjfm')
import icom_r8600 as r8600

def main():
    print("=" * 60)
    print("I/Q Spectrum Analysis")
    print("=" * 60)

    radio = r8600.IcomR8600()
    radio._iq_gain = 1.0  # Unity gain for analysis

    try:
        radio.open()

        # Setup streaming
        radio._send_command(r8600._build_civ_command([0x1A, 0x13, 0x00, 0x01]))  # REMOTE ON
        time.sleep(0.1)

        freq = 89.9e6
        bcd_freq = r8600._freq_to_bcd(freq)
        radio._send_command(r8600._build_civ_command([0x05] + list(bcd_freq)))
        print(f"\nCenter frequency: {freq/1e6:.1f} MHz")

        # Query settings
        settings = radio.query_rf_settings()
        print(f"RF Settings: {settings}")

        # Start streaming
        radio.streaming_mode = "iq"
        radio.iq_sample_rate = 480000
        radio._bit_depth = 16
        radio._bytes_per_sample = 4
        radio._running = True
        radio._iq_buffer = []

        import threading
        radio._iq_thread = threading.Thread(target=radio._iq_reader_loop, daemon=True)
        radio._iq_thread.start()

        # Enable I/Q output
        radio._send_command(r8600._build_civ_command([0x1A, 0x13, 0x01, 0x01, 0x00, 0x05]))
        time.sleep(0.5)

        # Collect samples
        radio.flush_iq()
        time.sleep(0.2)

        print("\nCollecting I/Q samples...")
        samples_list = []
        for _ in range(20):
            samples = radio.fetch_iq(16384)
            samples_list.append(samples)
            time.sleep(0.05)

        iq = np.concatenate(samples_list)
        print(f"Collected {len(iq)} samples at {radio.iq_sample_rate/1000:.0f} kHz")

        # Time domain analysis
        print("\n--- Time Domain ---")
        power = np.mean(np.abs(iq)**2)
        peak = np.abs(iq).max()
        print(f"  Power: {10*np.log10(power + 1e-10):.1f} dB")
        print(f"  Peak: {peak:.4f}")
        print(f"  Std I: {np.real(iq).std():.4f}")
        print(f"  Std Q: {np.imag(iq).std():.4f}")

        # Frequency domain analysis
        print("\n--- Frequency Domain ---")
        fft = np.fft.fftshift(np.fft.fft(iq))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1/radio.iq_sample_rate))
        power_spectrum = np.abs(fft)**2

        # Find peak frequency
        peak_idx = np.argmax(power_spectrum)
        peak_freq_offset = freqs[peak_idx]
        print(f"  Peak frequency offset: {peak_freq_offset/1000:.1f} kHz")
        print(f"  Peak at: {(freq + peak_freq_offset)/1e6:.4f} MHz")

        # Measure FM bandwidth (where power is > -20dB of peak)
        peak_power = power_spectrum[peak_idx]
        threshold = peak_power / 100  # -20 dB
        above_threshold = power_spectrum > threshold
        if np.any(above_threshold):
            first_idx = np.argmax(above_threshold)
            last_idx = len(above_threshold) - 1 - np.argmax(above_threshold[::-1])
            bw = freqs[last_idx] - freqs[first_idx]
            print(f"  Approximate signal BW: {bw/1000:.1f} kHz")

        # Check for FM modulation by looking at instantaneous frequency
        print("\n--- FM Analysis ---")
        # Compute instantaneous phase and frequency
        phase = np.angle(iq)
        # Unwrap phase
        unwrapped = np.unwrap(phase)
        # Instantaneous frequency is derivative of phase
        inst_freq = np.diff(unwrapped) * radio.iq_sample_rate / (2 * np.pi)

        print(f"  Inst. freq mean: {np.mean(inst_freq)/1000:.1f} kHz offset")
        print(f"  Inst. freq std:  {np.std(inst_freq)/1000:.1f} kHz")
        print(f"  Inst. freq range: {np.min(inst_freq)/1000:.1f} to {np.max(inst_freq)/1000:.1f} kHz")

        # FM deviation is the std of instantaneous frequency
        # For FM broadcast, expect 75 kHz deviation
        fm_deviation = np.std(inst_freq)
        print(f"\n  Estimated FM deviation: {fm_deviation/1000:.1f} kHz")

        if 20000 < fm_deviation < 100000:
            print("  --> Looks like valid FM broadcast signal!")
        elif fm_deviation < 5000:
            print("  --> Very low deviation - might be just noise")
        else:
            print("  --> Unusual deviation")

        # Check DC offset
        print("\n--- DC Offset ---")
        dc_i = np.mean(np.real(iq))
        dc_q = np.mean(np.imag(iq))
        print(f"  DC offset I: {dc_i:.6f}")
        print(f"  DC offset Q: {dc_q:.6f}")

        # Check I/Q balance
        print("\n--- I/Q Balance ---")
        print(f"  I power: {np.var(np.real(iq)):.6f}")
        print(f"  Q power: {np.var(np.imag(iq)):.6f}")
        print(f"  Ratio: {np.var(np.real(iq)) / (np.var(np.imag(iq)) + 1e-10):.3f}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            radio._running = False
            if radio._iq_thread:
                radio._iq_thread.join(timeout=1.0)
            radio.close()
        except:
            pass

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
