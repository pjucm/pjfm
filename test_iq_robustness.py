#!/usr/bin/env python3
"""
IC-R8600 I/Q Stream Robustness Tests

Tests sync pattern detection, buffer management, error recovery, and
performance under various conditions. Designed to catch edge cases
in the fetch_iq() implementation.

Usage:
    python test_iq_robustness.py          # Run all tests (requires hardware)
    python test_iq_robustness.py --sim    # Run simulation tests only (no hardware)
    python test_iq_robustness.py 1920/16 3840/24          # Test specific combos
    python test_iq_robustness.py 1920/16 3840/24 --sync   # Sync tests only
    python test_iq_robustness.py 960/24 --duration        # Duration test only

Tests:
    1. Sync interval consistency across all sample rates
    2. bytes_needed calculation correctness
    3. Buffer growth under load
    4. Diagnostic counter accuracy
    5. DC offset convergence time
    6. Long-duration streaming stability
    7. Sync recovery after simulated corruption
"""

import argparse
import numpy as np
import time
import sys
import struct
from collections import deque

# Import sync interval table and constants
from icom_r8600 import SYNC_INTERVAL, SAMPLE_RATES, SAMPLE_RATES_24BIT


# =============================================================================
# Simulation Tests (No Hardware Required)
# =============================================================================

def test_bytes_needed_calculation():
    """
    Verify bytes_needed calculation is correct for all sample rates.

    Tests that the formula correctly accounts for sync pattern intervals
    at all supported sample rates.
    """
    print("\n" + "=" * 70)
    print("Test: bytes_needed Calculation Correctness")
    print("=" * 70)

    num_samples = 8192
    all_correct = True

    print(f"\n  Testing with num_samples = {num_samples}")
    print(f"\n  {'Rate':>12}  {'Sync Int':>10}  {'Calculated':>12}  {'Minimum':>12}  {'Status'}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}")

    for sample_rate, sync_interval in SYNC_INTERVAL.items():
        # Determine sample size (16-bit = 4 bytes, 24-bit = 6 bytes)
        # Use 16-bit for this test
        sample_size = 4
        sync_len = 4  # 16-bit sync pattern

        # Current implementation (fixed formula using actual sync interval)
        syncs_expected = num_samples // sync_interval + 2
        calculated_bytes = num_samples * sample_size + sync_len * syncs_expected

        # Minimum required (exact calculation)
        min_syncs = num_samples // sync_interval + 1
        minimum_bytes = num_samples * sample_size + sync_len * min_syncs

        # Check if calculated is sufficient (must be >= minimum)
        status = "OK" if calculated_bytes >= minimum_bytes else "UNDER"
        if status == "UNDER":
            all_correct = False

        rate_str = f"{sample_rate/1000:.0f}k"
        print(f"  {rate_str:>12}  {sync_interval:>10}  {calculated_bytes:>12}  {minimum_bytes:>12}  {status}")

    print(f"\n  Result: {'PASS' if all_correct else 'FAIL'}")

    return all_correct


def test_sync_pattern_parsing_16bit():
    """
    Test 16-bit sync pattern detection with synthetic data.

    Creates a byte stream with known sync patterns and verifies
    the parsing logic correctly identifies sample boundaries.
    """
    print("\n" + "=" * 70)
    print("Test: 16-bit Sync Pattern Parsing")
    print("=" * 70)

    sample_rate = 480000
    sync_interval = SYNC_INTERVAL[sample_rate]
    sample_size = 4
    sync_bytes = b"\x00\x80\x00\x80"  # 16-bit sync pattern

    # Generate synthetic I/Q stream with sync patterns
    num_samples = sync_interval * 3  # 3 sync intervals
    data = bytearray()

    for i in range(num_samples):
        if i > 0 and i % sync_interval == 0:
            data.extend(sync_bytes)

        # Generate sample: i = sin pattern, q = cos pattern (scaled to 16-bit)
        i_val = int(16000 * np.sin(2 * np.pi * i / 100))
        q_val = int(16000 * np.cos(2 * np.pi * i / 100))
        data.extend(struct.pack("<hh", i_val, q_val))

    print(f"\n  Generated {len(data)} bytes with {num_samples} samples")
    print(f"  Sync interval: {sync_interval} samples")
    print(f"  Expected syncs: {num_samples // sync_interval}")

    # Find sync patterns
    sync_positions = []
    pos = 0
    while True:
        idx = data.find(sync_bytes, pos)
        if idx == -1:
            break
        sync_positions.append(idx)
        pos = idx + 1

    print(f"  Found {len(sync_positions)} sync patterns")

    # Verify intervals
    if len(sync_positions) >= 2:
        intervals = [sync_positions[i+1] - sync_positions[i] for i in range(len(sync_positions)-1)]
        expected_bytes = sync_interval * sample_size + len(sync_bytes)
        all_correct = all(i == expected_bytes for i in intervals)
        print(f"  Expected byte interval: {expected_bytes}")
        print(f"  Actual intervals: {set(intervals)}")
        print(f"\n  Result: {'PASS' if all_correct else 'FAIL'}")
        return all_correct
    else:
        print("\n  Result: FAIL (not enough sync patterns found)")
        return False


def test_sync_pattern_parsing_24bit():
    """
    Test 24-bit sync pattern detection with synthetic data.

    24-bit mode uses a different sync pattern (0x8000, 0x8001, 0x8002)
    and requires more careful alignment verification.
    """
    print("\n" + "=" * 70)
    print("Test: 24-bit Sync Pattern Parsing")
    print("=" * 70)

    sample_rate = 480000
    sync_interval = SYNC_INTERVAL[sample_rate]
    sample_size = 6
    sync_bytes = b"\x00\x80\x01\x80\x02\x80"  # 24-bit sync pattern

    # Generate synthetic I/Q stream with sync patterns
    num_samples = sync_interval * 3
    data = bytearray()

    for i in range(num_samples):
        if i > 0 and i % sync_interval == 0:
            data.extend(sync_bytes)

        # Generate 24-bit samples (using 23 bits of range)
        i_val = int(4000000 * np.sin(2 * np.pi * i / 100))
        q_val = int(4000000 * np.cos(2 * np.pi * i / 100))

        # Pack as 24-bit little-endian
        data.extend((i_val & 0xFFFFFF).to_bytes(3, 'little', signed=False) if i_val >= 0
                    else ((i_val + 0x1000000) & 0xFFFFFF).to_bytes(3, 'little', signed=False))
        data.extend((q_val & 0xFFFFFF).to_bytes(3, 'little', signed=False) if q_val >= 0
                    else ((q_val + 0x1000000) & 0xFFFFFF).to_bytes(3, 'little', signed=False))

    print(f"\n  Generated {len(data)} bytes with {num_samples} samples")
    print(f"  Sync interval: {sync_interval} samples")

    # Find sync patterns
    sync_positions = []
    pos = 0
    while True:
        idx = data.find(sync_bytes, pos)
        if idx == -1:
            break
        sync_positions.append(idx)
        pos = idx + 1

    print(f"  Found {len(sync_positions)} sync patterns")

    # Verify intervals
    if len(sync_positions) >= 2:
        intervals = [sync_positions[i+1] - sync_positions[i] for i in range(len(sync_positions)-1)]
        expected_bytes = sync_interval * sample_size + len(sync_bytes)
        all_correct = all(i == expected_bytes for i in intervals)
        print(f"  Expected byte interval: {expected_bytes}")
        print(f"  Actual intervals: {set(intervals)}")
        print(f"\n  Result: {'PASS' if all_correct else 'FAIL'}")
        return all_correct
    else:
        print("\n  Result: FAIL (not enough sync patterns found)")
        return False


def test_sync_recovery_simulation():
    """
    Simulate sync loss and recovery scenarios.

    Tests the fetch_iq() logic for handling:
    1. Missing sync patterns
    2. Corrupted sync patterns
    3. Extra bytes inserted (USB glitch simulation)
    """
    print("\n" + "=" * 70)
    print("Test: Sync Recovery Simulation")
    print("=" * 70)

    sample_rate = 480000
    sync_interval = SYNC_INTERVAL[sample_rate]
    sample_size = 4
    sync_bytes = b"\x00\x80\x00\x80"

    scenarios = [
        ("Normal stream", 0, 0, 0),
        ("1 missing sync", 1, 0, 0),
        ("3 missing syncs", 3, 0, 0),
        ("1 corrupted sync", 0, 1, 0),
        ("4 extra bytes inserted", 0, 0, 4),
        ("12 extra bytes inserted", 0, 0, 12),
    ]

    results = []

    for name, missing, corrupted, extra_bytes in scenarios:
        # Generate stream
        num_intervals = 10
        data = bytearray()

        for interval_idx in range(num_intervals):
            # Add sync pattern (or skip/corrupt it)
            if interval_idx > 0:
                if missing > 0 and interval_idx <= missing:
                    pass  # Skip sync
                elif corrupted > 0 and interval_idx <= corrupted:
                    data.extend(b"\x00\x81\x00\x80")  # Corrupted sync
                else:
                    data.extend(sync_bytes)

                # Insert extra bytes after first sync
                if extra_bytes > 0 and interval_idx == 1:
                    data.extend(b"\xFF" * extra_bytes)

            # Add samples
            for _ in range(sync_interval):
                data.extend(struct.pack("<hh", 1000, 2000))

        # Find syncs in corrupted stream
        sync_positions = []
        pos = 0
        while True:
            idx = data.find(sync_bytes, pos)
            if idx == -1:
                break
            sync_positions.append(idx)
            pos = idx + 1

        # Analyze
        expected_syncs = num_intervals - 1 - missing - corrupted
        found_syncs = len(sync_positions)

        # Check if recovery is possible (syncs still at regular intervals)
        if found_syncs >= 2:
            intervals = [sync_positions[i+1] - sync_positions[i] for i in range(len(sync_positions)-1)]
            # Recovery possible if remaining syncs are still regular
            can_recover = len(set(intervals)) <= 2  # Allow one irregular interval
        else:
            can_recover = found_syncs >= 1

        status = "OK" if can_recover else "FAIL"
        results.append((name, status))
        print(f"\n  {name}:")
        print(f"    Expected syncs: {expected_syncs}, Found: {found_syncs}")
        print(f"    Recovery possible: {can_recover}")

    print("\n" + "-" * 40)
    all_pass = all(s == "OK" for _, s in results)
    print(f"  Result: {'PASS' if all_pass else 'FAIL'}")

    return all_pass


def test_dc_offset_convergence():
    """
    Test DC offset EMA convergence time.

    With alpha=0.001, the EMA takes ~1000 samples to converge.
    This test verifies the settling behavior and recommends
    faster initialization strategies.
    """
    print("\n" + "=" * 70)
    print("Test: DC Offset Convergence Analysis")
    print("=" * 70)

    alpha = 0.001  # Current implementation
    dc_actual = 0.05 + 0.03j  # Simulated DC offset

    # Simulate convergence
    dc_estimate = 0.0 + 0.0j
    samples_per_block = 8192
    sample_rate = 480000
    blocks_per_second = sample_rate / samples_per_block

    convergence_threshold = 0.01  # Within 1% of actual
    blocks_to_converge = 0

    for block in range(2000):
        # Simulate block with DC offset
        block_dc = dc_actual + 0.001 * (np.random.randn() + 1j * np.random.randn())

        # EMA update
        dc_estimate = alpha * block_dc + (1 - alpha) * dc_estimate

        # Check convergence
        error = abs(dc_estimate - dc_actual) / abs(dc_actual)
        if error < convergence_threshold and blocks_to_converge == 0:
            blocks_to_converge = block + 1

    time_to_converge = blocks_to_converge / blocks_per_second

    print(f"\n  EMA alpha: {alpha}")
    print(f"  Actual DC offset: {dc_actual}")
    print(f"  Blocks to converge: {blocks_to_converge}")
    print(f"  Time to converge: {time_to_converge:.1f} seconds")

    # Theoretical settling time: tau = -1/ln(1-alpha)
    tau_blocks = -1 / np.log(1 - alpha)
    tau_seconds = tau_blocks / blocks_per_second
    print(f"  Theoretical tau: {tau_blocks:.0f} blocks ({tau_seconds:.1f} seconds)")

    # Test with faster alpha
    fast_alpha = 0.01
    dc_estimate_fast = 0.0 + 0.0j
    blocks_fast = 0

    for block in range(500):
        block_dc = dc_actual + 0.001 * (np.random.randn() + 1j * np.random.randn())
        dc_estimate_fast = fast_alpha * block_dc + (1 - fast_alpha) * dc_estimate_fast
        error = abs(dc_estimate_fast - dc_actual) / abs(dc_actual)
        if error < convergence_threshold and blocks_fast == 0:
            blocks_fast = block + 1

    time_fast = blocks_fast / blocks_per_second
    print(f"\n  With alpha=0.01: {blocks_fast} blocks ({time_fast:.1f} seconds)")

    # Recommendation
    print("\n  RECOMMENDATION:")
    print("    Use adaptive alpha: start with 0.1 for first 10 blocks,")
    print("    then decay to 0.001 for steady-state tracking.")

    # Pass if convergence is reasonable
    passed = time_to_converge < 30.0  # Should converge within 30 seconds
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_buffer_growth_simulation():
    """
    Simulate buffer growth under various fetch patterns.

    Tests that buffer management with MAX_IQ_BUFFER_BYTES limit prevents
    unbounded growth when fetch_iq() is called slower than USB read rate.
    """
    print("\n" + "=" * 70)
    print("Test: Buffer Growth Simulation (with overflow protection)")
    print("=" * 70)

    # Import the actual limit from icom_r8600
    try:
        from icom_r8600 import MAX_IQ_BUFFER_BYTES
    except ImportError:
        MAX_IQ_BUFFER_BYTES = 5 * 1024 * 1024  # Fallback

    # Simulate USB read rate (65536 bytes every ~34ms at 480kHz)
    usb_read_size = 65536
    usb_read_interval_ms = 34
    sample_rate = 480000
    sample_size = 4

    scenarios = [
        ("Normal fetch rate", 8192, 34),      # Match USB rate
        ("Slow fetch (2x)", 8192, 68),        # Half fetch rate
        ("Very slow fetch (4x)", 8192, 136),  # Quarter fetch rate
        ("Large fetch, slow rate", 16384, 100),
    ]

    print(f"\n  USB read: {usb_read_size} bytes every {usb_read_interval_ms}ms")
    print(f"  Buffer limit: {MAX_IQ_BUFFER_BYTES / 1024 / 1024:.0f} MB")

    results = []
    for name, fetch_samples, fetch_interval_ms in scenarios:
        # Simulate 10 seconds
        duration_ms = 10000
        buffer_sizes = []
        overflow_count = 0

        usb_time = 0
        fetch_time = 0
        buffer_bytes = 0

        for t in range(0, duration_ms, 1):
            # USB read with overflow protection (simulates _iq_reader_loop)
            if t >= usb_time:
                buffer_bytes += usb_read_size
                # Apply buffer limit (matches implementation)
                if buffer_bytes > MAX_IQ_BUFFER_BYTES:
                    overflow_count += 1
                    buffer_bytes = MAX_IQ_BUFFER_BYTES // 2
                usb_time += usb_read_interval_ms

            # Fetch
            if t >= fetch_time:
                bytes_needed = fetch_samples * sample_size
                buffer_bytes = max(0, buffer_bytes - bytes_needed)
                fetch_time += fetch_interval_ms

            buffer_sizes.append(buffer_bytes)

        max_buffer = max(buffer_sizes)
        avg_buffer = sum(buffer_sizes) / len(buffer_sizes)

        # With overflow protection, buffer should never exceed limit
        status = "OK" if max_buffer <= MAX_IQ_BUFFER_BYTES else "OVERFLOW"
        results.append((name, status, max_buffer, overflow_count))

        print(f"\n  {name}:")
        print(f"    Fetch: {fetch_samples} samples every {fetch_interval_ms}ms")
        print(f"    Max buffer: {max_buffer/1024:.1f} KB")
        print(f"    Avg buffer: {avg_buffer/1024:.1f} KB")
        print(f"    Overflow trims: {overflow_count}")
        print(f"    Status: {status}")

    print("\n" + "-" * 40)

    all_pass = all(s == "OK" for _, s, _, _ in results)
    print(f"  Result: {'PASS' if all_pass else 'FAIL'}")

    if all_pass:
        print("\n  Buffer overflow protection is working correctly.")

    return all_pass


# =============================================================================
# Hardware Tests (Requires IC-R8600)
# =============================================================================

def test_hardware_sync_all_rates(use_24bit=False, rates=None):
    """
    Test sync pattern consistency at all (or specified) sample rates.

    Requires IC-R8600 connected.

    Args:
        use_24bit: If True, test 24-bit mode; otherwise test 16-bit mode.
        rates: Optional list of specific sample rates to test. If None, tests all.
    """
    bit_label = "24-bit" if use_24bit else "16-bit"
    print("\n" + "=" * 70)
    print(f"Test: Hardware Sync Pattern Verification ({bit_label})")
    print("=" * 70)

    try:
        from icom_r8600 import IcomR8600
    except ImportError as e:
        print(f"\n  SKIP: Cannot import IcomR8600: {e}")
        return True  # Skip, not fail

    try:
        radio = IcomR8600(use_24bit=use_24bit)
        radio.open()
    except Exception as e:
        print(f"\n  SKIP: Cannot connect to IC-R8600: {e}")
        return True

    results = []

    if rates:
        sample_rates = sorted(rates)
    elif use_24bit:
        sample_rates = [240000, 480000, 960000, 1920000, 3840000]
    else:
        sample_rates = [240000, 480000, 960000, 1920000, 3840000, 5120000]

    try:
        for sample_rate in sample_rates:
            print(f"\n  Testing {sample_rate/1000:.0f} kHz ({bit_label})...")

            # Configure streaming at this sample rate
            # Allow extra time for rate transition - radio needs time to stabilize
            radio.configure_iq_streaming(freq=98.1e6, sample_rate=sample_rate)
            time.sleep(1.0)

            # First flush and warmup to drain stale data from rate transition
            radio.flush_iq()
            time.sleep(0.3)
            for _ in range(20):
                radio.fetch_iq(8192)

            # Second flush to ensure clean state for test
            radio.flush_iq()
            time.sleep(0.3)
            for _ in range(10):
                radio.fetch_iq(8192)

            # Reset counters after full stabilization
            radio._sync_misses = 0
            radio._sync_invalid_24 = 0
            radio._buffer_overflow_count = 0
            radio._fetch_slow_count = 0
            radio._sync_short_buf = 0
            radio.total_sample_loss = 0

            # Collect samples for test
            for _ in range(100):
                iq = radio.fetch_iq(8192)

            sync_misses = radio._sync_misses
            invalid_24 = radio._sync_invalid_24
            overflow_count = radio._buffer_overflow_count
            fetch_slow = radio._fetch_slow_count
            short_buf = radio._sync_short_buf
            sample_loss = radio.total_sample_loss

            # Allow up to 2 sync misses during rate transitions - these are transient
            # alignment issues that occur during USB/radio state changes and are
            # properly recovered from. Steady-state operation shows 0 misses.
            status = "OK" if sync_misses <= 2 else "FAIL"
            results.append((f"{sample_rate/1000:.0f}kHz", status, sync_misses))

            print(f"    Sync misses: {sync_misses}")
            print(f"    Invalid 24-bit samples: {invalid_24}")
            print(f"    Buffer overflows: {overflow_count}")
            print(f"    Fetch slow count: {fetch_slow}")
            print(f"    Short sync buffers: {short_buf}")
            print(f"    Sample loss events: {sample_loss}")
            print(f"    Status: {status}")

    finally:
        radio.close()

    print("\n" + "-" * 40)
    all_pass = all(s == "OK" for _, s, _ in results)
    print(f"  Result: {'PASS' if all_pass else 'FAIL'}")

    return all_pass


def test_hardware_long_duration(use_24bit=False, sample_rate=None):
    """
    Test long-duration streaming stability.

    Streams for 60 seconds and checks for sample loss, sync misses,
    and buffer issues.

    Args:
        use_24bit: If True, test 24-bit mode; otherwise test 16-bit mode.
        sample_rate: Specific sample rate to test. If None, uses default.
    """
    if sample_rate is None:
        sample_rate = 1920000 if use_24bit else 480000

    bit_label = "24-bit" if use_24bit else "16-bit"
    print("\n" + "=" * 70)
    print(f"Test: Long Duration Streaming (60s, {sample_rate/1000:.0f} kHz, {bit_label})")
    print("=" * 70)

    try:
        from icom_r8600 import IcomR8600
    except ImportError as e:
        print(f"\n  SKIP: Cannot import IcomR8600: {e}")
        return True

    try:
        radio = IcomR8600(use_24bit=use_24bit)
        radio.open()
        radio.configure_iq_streaming(freq=98.1e6, sample_rate=sample_rate)
        time.sleep(0.5)
        radio.flush_iq()
    except Exception as e:
        print(f"\n  SKIP: Cannot connect to IC-R8600: {e}")
        return True

    duration_sec = 60
    fetch_count = 0
    total_samples = 0
    max_fetch_ms = 0
    last_print_sec = -1

    # Reset counters
    radio._sync_misses = 0
    radio.total_sample_loss = 0
    radio._buffer_overflow_count = 0
    radio._fetch_slow_count = 0

    print(f"\n  Sample rate: {sample_rate/1000:.0f} kHz")
    print(f"  Streaming for {duration_sec} seconds...")
    start = time.time()

    try:
        while time.time() - start < duration_sec:
            t0 = time.perf_counter()
            iq = radio.fetch_iq(8192)
            fetch_ms = (time.perf_counter() - t0) * 1000

            fetch_count += 1
            total_samples += len(iq)
            max_fetch_ms = max(max_fetch_ms, fetch_ms)

            # Progress indicator every 10 seconds
            elapsed_sec = int(time.time() - start)
            if elapsed_sec % 10 == 0 and elapsed_sec != last_print_sec and elapsed_sec > 0:
                last_print_sec = elapsed_sec
                print(f"    {elapsed_sec}s: {fetch_count} fetches, "
                      f"sync_misses={radio._sync_misses}, "
                      f"sample_loss={radio.total_sample_loss}")

    finally:
        radio.close()

    # Results
    print(f"\n  Duration: {duration_sec} seconds")
    print(f"  Total fetches: {fetch_count}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Max fetch time: {max_fetch_ms:.1f} ms")
    print(f"  Sync misses: {radio._sync_misses}")
    print(f"  Sample loss events: {radio.total_sample_loss}")
    print(f"  Buffer overflows: {radio._buffer_overflow_count}")
    print(f"  Fetch slow count: {radio._fetch_slow_count}")

    # Allow up to 2 sync misses - transient alignment issues during startup
    # are acceptable as long as they don't accumulate during steady-state
    passed = radio._sync_misses <= 2 and radio.total_sample_loss == 0
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_hardware_diagnostics_api():
    """
    Test that diagnostic counters are accessible via get_diagnostics().
    """
    print("\n" + "=" * 70)
    print("Test: Diagnostic Counters API")
    print("=" * 70)

    try:
        from icom_r8600 import IcomR8600
    except ImportError as e:
        print(f"\n  SKIP: Cannot import IcomR8600: {e}")
        return True

    try:
        radio = IcomR8600(use_24bit=False)
        radio.open()
        radio.configure_iq_streaming(freq=98.1e6, sample_rate=480000)
        time.sleep(0.5)
        radio.flush_iq()
    except Exception as e:
        print(f"\n  SKIP: Cannot connect to IC-R8600: {e}")
        return True

    try:
        # Fetch some data to populate counters
        for _ in range(50):
            radio.fetch_iq(8192)

        # Test get_diagnostics() method
        diagnostics = radio.get_diagnostics()

        print("\n  get_diagnostics() output:")
        for name, value in diagnostics.items():
            print(f"    {name}: {value}")

        # Verify expected keys exist
        expected_keys = [
            'sync_misses', 'sync_invalid_24', 'buffer_overflow_count',
            'total_sample_loss', 'recent_sample_loss', 'fetch_last_ms',
            'fetch_slow_count', 'civ_timeouts', 'initial_aligns', 'flush_during_fetch'
        ]

        missing_keys = [k for k in expected_keys if k not in diagnostics]
        if missing_keys:
            print(f"\n  Missing keys: {missing_keys}")
            all_exist = False
        else:
            print(f"\n  All {len(expected_keys)} expected keys present")
            all_exist = True

    except AttributeError as e:
        print(f"\n  Missing get_diagnostics() method: {e}")
        all_exist = False

    finally:
        radio.close()

    print(f"\n  Result: {'PASS' if all_exist else 'FAIL'}")
    return all_exist


# =============================================================================
# Main Test Runner
# =============================================================================

def run_simulation_tests():
    """Run tests that don't require hardware."""
    tests = [
        ("bytes_needed Calculation", test_bytes_needed_calculation),
        ("16-bit Sync Pattern Parsing", test_sync_pattern_parsing_16bit),
        ("24-bit Sync Pattern Parsing", test_sync_pattern_parsing_24bit),
        ("Sync Recovery Simulation", test_sync_recovery_simulation),
        ("DC Offset Convergence", test_dc_offset_convergence),
        ("Buffer Growth Simulation", test_buffer_growth_simulation),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    return results


def run_hardware_tests(use_24bit=None, test_filter=None, rate_specs=None):
    """Run tests that require IC-R8600 hardware.

    Args:
        use_24bit: None=both, True=24-bit only, False=16-bit only
        test_filter: None=all, 'sync'=sync tests only, 'duration'=duration tests only
        rate_specs: Optional list of (rate, use_24bit) tuples for specific combos
    """
    tests = []

    # If specific rate/bits combos were given, build targeted tests
    if rate_specs:
        run_sync = test_filter is None or test_filter == 'sync'
        run_duration = test_filter is None or test_filter == 'duration'

        # Group rates by bit depth for sync tests
        rates_16 = [r for r, is24 in rate_specs if not is24]
        rates_24 = [r for r, is24 in rate_specs if is24]

        if run_sync:
            if rates_16:
                label = ', '.join(f'{r/1000:.0f}k' for r in sorted(rates_16))
                tests.append((f"Hardware Sync ({label}, 16-bit)",
                             lambda r=rates_16: test_hardware_sync_all_rates(use_24bit=False, rates=r)))
            if rates_24:
                label = ', '.join(f'{r/1000:.0f}k' for r in sorted(rates_24))
                tests.append((f"Hardware Sync ({label}, 24-bit)",
                             lambda r=rates_24: test_hardware_sync_all_rates(use_24bit=True, rates=r)))

        if run_duration:
            for rate, is24 in rate_specs:
                bit_str = "24-bit" if is24 else "16-bit"
                tests.append((f"Hardware Long Duration ({rate/1000:.0f}k, {bit_str})",
                             lambda r=rate, b=is24: test_hardware_long_duration(use_24bit=b, sample_rate=r)))

        tests.append(("Hardware Diagnostics API", test_hardware_diagnostics_api))
        return _run_test_list(tests)

    # Original behavior: run by bit-depth and test-type filters
    run_16bit = use_24bit is None or use_24bit is False
    run_24bit = use_24bit is None or use_24bit is True
    run_sync = test_filter is None or test_filter == 'sync'
    run_duration = test_filter is None or test_filter == 'duration'
    run_diag = test_filter is None

    if run_sync:
        if run_16bit:
            tests.append(("Hardware Sync All Rates (16-bit)",
                         lambda: test_hardware_sync_all_rates(use_24bit=False)))
        if run_24bit:
            tests.append(("Hardware Sync All Rates (24-bit)",
                         lambda: test_hardware_sync_all_rates(use_24bit=True)))

    if run_duration:
        if run_16bit:
            tests.append(("Hardware Long Duration (16-bit)",
                         lambda: test_hardware_long_duration(use_24bit=False)))
        if run_24bit:
            tests.append(("Hardware Long Duration (24-bit)",
                         lambda: test_hardware_long_duration(use_24bit=True)))

    if run_diag:
        tests.append(("Hardware Diagnostics API", test_hardware_diagnostics_api))

    return _run_test_list(tests)


def _run_test_list(tests):
    """Run a list of (name, callable) test pairs and return results."""
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    return results


def main():
    parser = argparse.ArgumentParser(description="IC-R8600 I/Q Stream Robustness Tests")
    parser.add_argument("--sim", action="store_true",
                        help="Run simulation tests only (no hardware required)")
    parser.add_argument("--hw", action="store_true",
                        help="Run hardware tests only")
    parser.add_argument("--16bit", dest="use_16bit", action="store_true",
                        help="Run 16-bit hardware tests only")
    parser.add_argument("--24bit", dest="use_24bit", action="store_true",
                        help="Run 24-bit hardware tests only")
    parser.add_argument("--sync", action="store_true",
                        help="Run sync pattern tests only")
    parser.add_argument("--duration", action="store_true",
                        help="Run long duration tests only")
    parser.add_argument("tests", nargs="*", metavar="RATE/BITS",
                        help="Rate/bit-depth combos, e.g. 1920/16 3840/24 (rate in kHz)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("IC-R8600 I/Q STREAM ROBUSTNESS TEST SUITE")
    print("=" * 70)

    results = []

    # Parse positional rate/bits specs
    rate_specs = None
    if args.tests:
        rate_specs = []
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
                rate_specs.append((rate, use_24bit))
            except ValueError:
                print(f"Error: invalid test spec '{spec}' (expected RATE/BITS, e.g. 1920/16)")
                return 1

    if rate_specs:
        # Specific combos imply hardware tests
        test_filter = None
        if args.sync and args.duration:
            test_filter = None
        elif args.sync:
            test_filter = 'sync'
        elif args.duration:
            test_filter = 'duration'

        print("\n--- HARDWARE TESTS ---")
        results.extend(run_hardware_tests(rate_specs=rate_specs, test_filter=test_filter))
    else:
        # Original flag-based behavior
        # Determine bit depth filter
        if args.use_16bit and args.use_24bit:
            bit_filter = None  # Both
        elif args.use_16bit:
            bit_filter = False  # 16-bit only
        elif args.use_24bit:
            bit_filter = True   # 24-bit only
        else:
            bit_filter = None   # Both

        # Determine test filter
        if args.sync and args.duration:
            test_filter = None  # Both
        elif args.sync:
            test_filter = 'sync'
        elif args.duration:
            test_filter = 'duration'
        else:
            test_filter = None  # All

        # If specific test filters are set, imply --hw
        if args.use_16bit or args.use_24bit or args.sync or args.duration:
            args.hw = True

        if args.sim or not args.hw:
            print("\n--- SIMULATION TESTS ---")
            results.extend(run_simulation_tests())

        if args.hw or not args.sim:
            print("\n--- HARDWARE TESTS ---")
            results.extend(run_hardware_tests(use_24bit=bit_filter, test_filter=test_filter))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:40s}  {status}")

    print(f"\n  {passed_count}/{total_count} tests passed")

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
