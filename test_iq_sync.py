#!/usr/bin/env python3
"""
IC-R8600 I/Q Sync Pattern Analysis

Tests whether sync patterns appear at exactly the expected intervals in the
I/Q stream, with no variance. If sync positions are deterministic, we can
simplify the fetch_iq() parsing logic.

Usage:
    ./test_iq_sync.py [--duration SECONDS] [--freq MHZ] [--16bit|--24bit] [--rate RATE]

Examples:
    ./test_iq_sync.py --16bit --rate 960000
    ./test_iq_sync.py --24bit --rate 480000
    ./test_iq_sync.py --all  # Test all supported rate/bit-depth combinations
"""

import argparse
import time
import usb.core
import usb.util
import os
import sys
import statistics

# Import from icom_r8600
from icom_r8600 import (
    ICOM_VID, R8600_PID_READY, R8600_PID_BOOTLOADER,
    EP_IQ_IN, SAMPLE_RATES, SAMPLE_RATES_24BIT, SYNC_INTERVAL,
    _switch_to_iq_mode, _build_civ_command, _freq_to_bcd,
    CIV_NG, EP_CMD_OUT, EP_RESP_IN
)

# Sync patterns per Icom I/Q Reference Guide
# 16-bit: 0x8000, 0x8000 (little-endian)
# 24-bit: 0x8000, 0x8001, 0x8002 (little-endian)
SYNC_16BIT = b"\x00\x80\x00\x80"
SYNC_24BIT = b"\x00\x80\x01\x80\x02\x80"

SAMPLE_SIZE_16BIT = 4  # 2 bytes I + 2 bytes Q
SAMPLE_SIZE_24BIT = 6  # 3 bytes I + 3 bytes Q


def find_all_sync_positions(data: bytes, sync_pattern: bytes,
                            expected_gap: int = 0) -> list[int]:
    """Find all positions of the sync pattern in data.

    If expected_gap > 0, each candidate is validated by checking that another
    sync pattern exists exactly expected_gap bytes later (matching the driver's
    double-sync validation).  This rejects false positives where real I/Q sample
    data happens to contain the sync byte sequence.
    """
    sync_len = len(sync_pattern)
    positions = []
    pos = 0
    while True:
        idx = data.find(sync_pattern, pos)
        if idx == -1:
            break
        if expected_gap > 0:
            next_pos = idx + sync_len + expected_gap
            if next_pos + sync_len <= len(data):
                if data[next_pos:next_pos + sync_len] != sync_pattern:
                    pos = idx + 1
                    continue
            else:
                # Not enough data to verify — accept if prior validated sync
                # is at the expected distance behind us
                if positions and idx - positions[-1] != sync_len + expected_gap:
                    pos = idx + 1
                    continue
        positions.append(idx)
        pos = idx + 1
    return positions


def analyze_sync_intervals(positions: list[int], sample_size: int,
                           sync_pattern: bytes, sample_rate: int) -> dict:
    """
    Analyze intervals between sync patterns.

    Returns dict with:
        - intervals_bytes: list of byte intervals between consecutive syncs
        - intervals_samples: list of sample intervals
        - expected_samples: expected samples between syncs
        - all_aligned: True if all syncs are sample-aligned
        - all_expected: True if all intervals match expected
    """
    if len(positions) < 2:
        return {"error": "Not enough sync patterns found"}

    intervals_bytes = []
    for i in range(1, len(positions)):
        intervals_bytes.append(positions[i] - positions[i-1])

    # Check if all positions are sample-aligned (divisible by sample_size after sync)
    # First sync establishes alignment; subsequent should be at sync + N*sample_size
    first_sync = positions[0]
    all_aligned = all(
        (pos - first_sync) % sample_size == 0
        for pos in positions
    )

    # Convert to sample intervals (subtract sync pattern length)
    intervals_samples = [
        (b - len(sync_pattern)) // sample_size
        for b in intervals_bytes
    ]

    # Expected interval for this sample rate
    expected = SYNC_INTERVAL.get(sample_rate, 1024)
    all_expected = all(s == expected for s in intervals_samples)

    return {
        "intervals_bytes": intervals_bytes,
        "intervals_samples": intervals_samples,
        "expected_samples": expected,
        "all_aligned": all_aligned,
        "all_expected": all_expected,
        "sync_count": len(positions),
        "first_sync_pos": positions[0] if positions else None,
    }


class RawIQStreamer:
    """Minimal streamer for raw I/Q data analysis."""

    def __init__(self, sample_rate=480000, use_24bit=True):
        self.device = None
        self.sample_rate = sample_rate
        self.use_24bit = use_24bit
        self._validate_rate()

    def _validate_rate(self):
        """Validate that sample_rate is supported for the chosen bit depth."""
        if self.use_24bit:
            if self.sample_rate not in SAMPLE_RATES_24BIT:
                available = sorted(SAMPLE_RATES_24BIT.keys())
                raise ValueError(
                    f"Sample rate {self.sample_rate} not supported for 24-bit mode. "
                    f"Available: {available}"
                )
        else:
            if self.sample_rate not in SAMPLE_RATES:
                available = sorted(SAMPLE_RATES.keys())
                raise ValueError(
                    f"Sample rate {self.sample_rate} not supported for 16-bit mode. "
                    f"Available: {available}"
                )

    def open(self):
        """Open connection to IC-R8600."""
        self.device = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_READY)

        if self.device is None:
            bootloader = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_BOOTLOADER)
            if bootloader:
                self.device = _switch_to_iq_mode(bootloader)
                time.sleep(0.5)
                self.device = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_READY)
            else:
                raise RuntimeError("IC-R8600 not found")

        try:
            if self.device.is_kernel_driver_active(0):
                self.device.detach_kernel_driver(0)
        except (usb.core.USBError, NotImplementedError):
            pass

        self.device.set_configuration()
        usb.util.claim_interface(self.device, 0)
        print(f"Connected to IC-R8600")

    def _send_command(self, cmd_bytes, timeout=2000):
        """Send CI-V command and return response."""
        self.device.write(EP_CMD_OUT, cmd_bytes, timeout=timeout)
        try:
            response = self.device.read(EP_RESP_IN, 64, timeout=timeout)
            return bytes(response)
        except usb.core.USBTimeoutError:
            return None

    def configure(self, freq_hz):
        """Configure for I/Q streaming."""
        # Get rate code based on bit depth
        if self.use_24bit:
            bit_depth, rate_code = SAMPLE_RATES_24BIT[self.sample_rate]
        else:
            bit_depth, rate_code = SAMPLE_RATES[self.sample_rate]

        # Disable any existing I/Q output
        try:
            self._send_command(_build_civ_command([0x1A, 0x13, 0x01, 0x00]), timeout=500)
        except:
            pass
        time.sleep(0.1)

        # Enable I/Q mode
        resp = self._send_command(_build_civ_command([0x1A, 0x13, 0x00, 0x01]))
        if resp and CIV_NG in resp:
            raise RuntimeError("Failed to enable I/Q mode")

        # Set frequency
        bcd_freq = _freq_to_bcd(freq_hz)
        resp = self._send_command(_build_civ_command([0x05] + list(bcd_freq)))
        if resp and CIV_NG in resp:
            raise RuntimeError("Failed to set frequency")

        # Enable I/Q output
        resp = self._send_command(_build_civ_command([0x1A, 0x13, 0x01, 0x01, bit_depth, rate_code]))
        if resp and CIV_NG in resp:
            raise RuntimeError("Failed to enable I/Q output")

        bit_str = "24-bit" if self.use_24bit else "16-bit"
        print(f"Streaming: {self.sample_rate/1e3:.0f} kHz, {bit_str}, freq={freq_hz/1e6:.3f} MHz")

    def read_raw(self, duration_sec: float) -> bytes:
        """Read raw I/Q data for specified duration."""
        data = bytearray()
        start = time.time()
        read_count = 0

        print(f"Collecting data for {duration_sec} seconds...")

        while time.time() - start < duration_sec:
            try:
                chunk = self.device.read(EP_IQ_IN, 65536, timeout=1000)
                if chunk:
                    data.extend(chunk)
                    read_count += 1
            except usb.core.USBTimeoutError:
                continue

        elapsed = time.time() - start
        print(f"Collected {len(data):,} bytes in {elapsed:.2f}s ({read_count} reads)")
        print(f"Data rate: {len(data)/elapsed/1e6:.2f} MB/s")

        return bytes(data)

    def close(self):
        """Close connection."""
        if self.device:
            try:
                self._send_command(_build_civ_command([0x1A, 0x13, 0x01, 0x00]), timeout=500)
                time.sleep(0.1)
                self._send_command(_build_civ_command([0x1A, 0x13, 0x00, 0x00]), timeout=500)
            except:
                pass
            try:
                usb.util.release_interface(self.device, 0)
                usb.util.dispose_resources(self.device)
            except:
                pass
            self.device = None


def run_test(sample_rate: int, use_24bit: bool, freq_hz: int, duration: float) -> bool:
    """
    Run sync pattern test for a specific configuration.

    Returns True if test passed, False otherwise.
    """
    bit_str = "24-bit" if use_24bit else "16-bit"
    sync_pattern = SYNC_24BIT if use_24bit else SYNC_16BIT
    sample_size = SAMPLE_SIZE_24BIT if use_24bit else SAMPLE_SIZE_16BIT

    print(f"\n{'='*60}")
    print(f"Testing: {sample_rate/1e3:.0f} kHz, {bit_str}")
    print(f"{'='*60}")

    streamer = RawIQStreamer(sample_rate=sample_rate, use_24bit=use_24bit)

    try:
        streamer.open()
        streamer.configure(freq_hz)

        # Let stream stabilize
        time.sleep(0.5)

        # Collect raw data
        raw_data = streamer.read_raw(duration)

        # Find all sync patterns
        print("\nSYNC PATTERN ANALYSIS")
        print("-" * 40)

        sync_interval = SYNC_INTERVAL.get(sample_rate, 1024)
        expected_gap = sync_interval * sample_size  # bytes of I/Q data between syncs
        positions = find_all_sync_positions(raw_data, sync_pattern, expected_gap)
        print(f"Found {len(positions)} sync patterns")

        if len(positions) < 2:
            print("ERROR: Not enough sync patterns to analyze")
            return False

        # Analyze intervals
        analysis = analyze_sync_intervals(positions, sample_size, sync_pattern, sample_rate)

        print(f"\nExpected interval: {analysis['expected_samples']} samples")
        print(f"                   ({analysis['expected_samples'] * sample_size + len(sync_pattern)} bytes)")

        # Show first sync position
        print(f"\nFirst sync at byte position: {analysis['first_sync_pos']}")
        print(f"  Offset from sample alignment: {analysis['first_sync_pos'] % sample_size}")

        # Interval statistics
        intervals = analysis['intervals_samples']
        if intervals:
            unique_intervals = set(intervals)
            print(f"\nUnique sample intervals found: {sorted(unique_intervals)}")

            if len(unique_intervals) == 1:
                print(f"\n*** ALL {len(intervals)} INTERVALS ARE IDENTICAL: {intervals[0]} samples ***")
            else:
                print(f"\nInterval statistics:")
                print(f"  Min: {min(intervals)} samples")
                print(f"  Max: {max(intervals)} samples")
                print(f"  Mean: {statistics.mean(intervals):.2f}")
                print(f"  Stdev: {statistics.stdev(intervals):.4f}")

                # Show distribution
                print(f"\nInterval distribution:")
                for val in sorted(unique_intervals):
                    count = intervals.count(val)
                    pct = count / len(intervals) * 100
                    print(f"  {val} samples: {count} occurrences ({pct:.1f}%)")

        # Byte-level interval analysis
        byte_intervals = analysis['intervals_bytes']
        expected_bytes = analysis['expected_samples'] * sample_size + len(sync_pattern)
        unique_bytes = set(byte_intervals)

        print(f"\nByte interval analysis:")
        print(f"  Expected: {expected_bytes} bytes")
        print(f"  Unique values: {sorted(unique_bytes)}")

        # Final verdict
        print("\n" + "-"*40)
        print("VERDICT")
        print("-"*40)

        if analysis['all_expected'] and analysis['all_aligned']:
            print("SUCCESS: All sync patterns are at exactly expected positions!")
            return True
        else:
            if not analysis['all_aligned']:
                print("FAILED: Sync patterns are not all sample-aligned")
            if not analysis['all_expected']:
                print("FAILED: Sync intervals vary from expected")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        streamer.close()


def main():
    # Build list of available rates for help text
    rates_16bit = sorted(SAMPLE_RATES.keys())
    rates_24bit = sorted(SAMPLE_RATES_24BIT.keys())

    parser = argparse.ArgumentParser(
        description="IC-R8600 I/Q Sync Pattern Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported sample rates:
  16-bit: {', '.join(f'{r/1e3:.0f}k' for r in rates_16bit)}
  24-bit: {', '.join(f'{r/1e3:.0f}k' for r in rates_24bit)}

Examples:
  ./test_iq_sync.py 1920/16 3840/24  # Test specific rate/bit-depth combos
  ./test_iq_sync.py 960/16            # Single test
  ./test_iq_sync.py --all             # Test all combinations
  ./test_iq_sync.py --16bit --rate 960000  # Explicit form
""")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration to stream in seconds (default: 5)")
    parser.add_argument("--freq", type=float, default=89.9,
                        help="Frequency in MHz (default: 89.9)")

    # Bit depth selection (mutually exclusive)
    bit_group = parser.add_mutually_exclusive_group()
    bit_group.add_argument("--16bit", dest="use_16bit", action="store_true",
                           help="Use 16-bit samples")
    bit_group.add_argument("--24bit", dest="use_24bit", action="store_true",
                           help="Use 24-bit samples (default)")
    bit_group.add_argument("--all", dest="test_all", action="store_true",
                           help="Test all supported rate/bit-depth combinations")

    parser.add_argument("--rate", type=int, default=480000,
                        help="Sample rate in Hz (default: 480000)")
    parser.add_argument("tests", nargs="*", metavar="RATE/BITS",
                        help="Rate/bit-depth combos, e.g. 1920/16 3840/24 (rate in kHz)")
    args = parser.parse_args()

    freq_hz = int(args.freq * 1e6)

    # Positional rate/bits arguments
    if args.tests:
        results = []
        for spec in args.tests:
            try:
                rate_str, bits_str = spec.split("/")
                rate = int(rate_str.rstrip("k")) * 1000
                use_24bit = bits_str in ("24", "24bit")
                if not use_24bit and bits_str not in ("16", "16bit"):
                    print(f"Error: invalid bit depth '{bits_str}' in '{spec}' (use 16 or 24)")
                    return 1
                valid_rates = SAMPLE_RATES_24BIT if use_24bit else SAMPLE_RATES
                if rate not in valid_rates:
                    bit_str = "24-bit" if use_24bit else "16-bit"
                    print(f"Error: rate {rate} not available for {bit_str} mode")
                    print(f"Available: {sorted(valid_rates.keys())}")
                    return 1
            except ValueError:
                print(f"Error: invalid test spec '{spec}' (expected RATE/BITS, e.g. 1920/16)")
                return 1
            passed = run_test(rate, use_24bit, freq_hz, args.duration)
            results.append((rate, f"{'24' if use_24bit else '16'}-bit", passed))
            if spec != args.tests[-1]:
                time.sleep(1.0)

        if len(results) > 1:
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"\n{'Rate':>12}  {'Bits':>8}  {'Result':>10}")
            print("-" * 35)
            for rate, bits, passed in results:
                status = "PASS" if passed else "FAIL"
                print(f"{rate/1e3:>10.0f}k  {bits:>8}  {status:>10}")
            passed_count = sum(1 for _, _, p in results if p)
            print("-" * 35)
            print(f"Total: {passed_count}/{len(results)} passed")

        return 0 if all(p for _, _, p in results) else 1

    # Test all combinations
    if args.test_all:
        results = []

        # Test all 16-bit rates
        for rate in sorted(SAMPLE_RATES.keys()):
            passed = run_test(rate, use_24bit=False, freq_hz=freq_hz, duration=args.duration)
            results.append((rate, "16-bit", passed))
            time.sleep(1.0)  # Give device time to settle between tests

        # Test all 24-bit rates
        for rate in sorted(SAMPLE_RATES_24BIT.keys()):
            passed = run_test(rate, use_24bit=True, freq_hz=freq_hz, duration=args.duration)
            results.append((rate, "24-bit", passed))
            time.sleep(1.0)

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\n{'Rate':>12}  {'Bits':>8}  {'Result':>10}")
        print("-" * 35)
        for rate, bits, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"{rate/1e3:>10.0f}k  {bits:>8}  {status:>10}")

        passed_count = sum(1 for _, _, p in results if p)
        print("-" * 35)
        print(f"Total: {passed_count}/{len(results)} passed")

        return 0 if passed_count == len(results) else 1

    # Single test mode
    use_24bit = not args.use_16bit  # Default to 24-bit unless --16bit specified

    # Validate rate for chosen bit depth
    if use_24bit:
        if args.rate not in SAMPLE_RATES_24BIT:
            print(f"Error: Rate {args.rate} not available for 24-bit mode")
            print(f"Available 24-bit rates: {sorted(SAMPLE_RATES_24BIT.keys())}")
            return 1
    else:
        if args.rate not in SAMPLE_RATES:
            print(f"Error: Rate {args.rate} not available for 16-bit mode")
            print(f"Available 16-bit rates: {sorted(SAMPLE_RATES.keys())}")
            return 1

    passed = run_test(args.rate, use_24bit, freq_hz, args.duration)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
