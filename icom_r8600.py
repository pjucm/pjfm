#!/usr/bin/env python3
"""
IC-R8600 I/Q Streaming Interface for pyfm

Provides a BB60D-compatible interface for the Icom IC-R8600's USB I/Q output.
Uses the protocol discovered via USB traffic capture analysis.

Requires: pip install pyusb
Linux: sudo access or udev rules for USB device
"""

import numpy as np
import os
import usb.core
import usb.util
import threading
import time

# Icom USB IDs
ICOM_VID = 0x0c26
R8600_PID_READY = 0x0023     # Device ready for I/Q streaming
R8600_PID_BOOTLOADER = 0x0022  # Device needs firmware upload

# Cypress FX2 CPUCS register address
CPUCS_ADDR = 0xE600

# USB Endpoints
EP_CMD_OUT = 0x02   # CI-V commands PC -> R8600
EP_IQ_IN = 0x86     # I/Q data R8600 -> PC
EP_RESP_IN = 0x88   # CI-V responses R8600 -> PC

# CI-V Protocol
CIV_PREAMBLE = 0xFE
CIV_TERMINATOR = 0xFD
CIV_OK = 0xFB
CIV_NG = 0xFA
R8600_ADDR = 0x96
PC_ADDR = 0xE0

# Sample rate codes for I/Q output (1A 13 01 01 BD SR)
# BD = bit depth (00=16-bit, 01=24-bit)
# SR = sample rate code
# Note: 5.12 MSPS only supports 16-bit; other rates support both
# Using 16-bit by default for stability; 24-bit available but needs more testing
SAMPLE_RATES = {
    5120000: (0x00, 0x01),  # 5.12 MSPS, 16-bit only (hardware limitation)
    3840000: (0x00, 0x02),  # 3.84 MSPS, 16-bit
    1920000: (0x00, 0x03),  # 1.92 MSPS, 16-bit
    960000:  (0x00, 0x04),  # 960 KSPS, 16-bit
    480000:  (0x00, 0x05),  # 480 KSPS, 16-bit
    240000:  (0x00, 0x06),  # 240 KSPS, 16-bit
}

# Alternative 24-bit sample rates (for future use when 24-bit is debugged)
SAMPLE_RATES_24BIT = {
    3840000: (0x01, 0x02),  # 3.84 MSPS, 24-bit
    1920000: (0x01, 0x03),  # 1.92 MSPS, 24-bit
    960000:  (0x01, 0x04),  # 960 KSPS, 24-bit
    480000:  (0x01, 0x05),  # 480 KSPS, 24-bit
    240000:  (0x01, 0x06),  # 240 KSPS, 24-bit
}

# Sync pattern interval (samples between sync markers) per Icom I/Q Reference Guide
# Used to verify alignment - if we don't see sync at expected interval, we're misaligned
SYNC_INTERVAL = {
    5120000: 10923,  # 16-bit only
    3840000: 8192,
    1920000: 4096,
    960000:  2048,
    480000:  1024,
    240000:  512,
}


def _build_civ_command(cmd_data):
    """Build a CI-V command packet with even-length padding."""
    packet = bytes([CIV_PREAMBLE, CIV_PREAMBLE, R8600_ADDR, PC_ADDR]) + bytes(cmd_data) + bytes([CIV_TERMINATOR])
    if len(packet) % 2 != 0:
        packet += bytes([0xFF])
    return packet


def _freq_to_bcd(freq_hz):
    """Convert frequency in Hz to BCD format (LSB first, 5 bytes)."""
    freq_str = f"{int(freq_hz):010d}"
    bcd = []
    for i in range(8, -2, -2):
        high = int(freq_str[i])
        low = int(freq_str[i + 1])
        bcd.append((high << 4) | low)
    return bytes(bcd)


def _load_hex_file(filename):
    """Parse Intel HEX file and return list of (addr, data) tuples."""
    segments = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith(':'):
                continue
            rec = bytes.fromhex(line[1:])
            length, addr_hi, addr_lo, rec_type = rec[0], rec[1], rec[2], rec[3]
            addr = (addr_hi << 8) | addr_lo
            data = rec[4:4+length]
            if rec_type == 0:
                segments.append((addr, data))
            elif rec_type == 1:
                break
    return segments


def _upload_firmware_stage(dev, segments):
    """Upload a firmware stage to Cypress FX2 device."""
    # Stop CPU
    dev.ctrl_transfer(0x40, 0xA0, CPUCS_ADDR, 0, bytes([0x01]), timeout=1000)

    # Upload segments
    for addr, data in segments:
        dev.ctrl_transfer(0x40, 0xA0, addr, 0, data, timeout=1000)

    # Start CPU
    dev.ctrl_transfer(0x40, 0xA0, CPUCS_ADDR, 0, bytes([0x00]), timeout=1000)


def _find_firmware_file():
    """Find the IC-R8600 firmware .spt file."""
    import os
    search_paths = [
        os.path.join(os.path.dirname(__file__), "IC-R8600_usb_iq.spt"),
        os.path.expanduser("~/dev/IC-R8600_usb_iq.spt"),
        os.path.expanduser("~/dev/r8600start/IC-R8600_usb_iq.spt"),
        os.path.expanduser("~/.local/share/pyfm/IC-R8600_usb_iq.spt"),
        "/usr/share/pyfm/IC-R8600_usb_iq.spt",
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None


def _extract_firmware_from_spt(spt_path):
    """Extract firmware stages from Cypress .spt script file."""
    import struct

    with open(spt_path, 'rb') as f:
        data = f.read()

    stages = []
    current_stage = []
    cpucs_state = None
    pos = 0

    while pos < len(data):
        if data[pos:pos+4] != b"CSPT":
            break

        length = struct.unpack("<L", data[pos+4:pos+8])[0]
        if length < 32 or pos + length > len(data):
            break

        # Parse chunk header
        fmt = "<LLLLBBHLLL"
        hdr_len = struct.calcsize(fmt)
        (_, _, _, _, request, _, addr, _, _, data_len) = struct.unpack(fmt, data[pos:pos+hdr_len])
        chunk_data = data[pos+hdr_len:pos+hdr_len+data_len]

        if request == 0xA0:  # Firmware upload request
            if addr == CPUCS_ADDR and len(chunk_data) == 1:
                # CPUCS write - controls CPU reset
                new_state = chunk_data[0] & 1
                if cpucs_state == 1 and new_state == 0:
                    # CPU started - end of stage
                    if current_stage:
                        stages.append(current_stage)
                        current_stage = []
                cpucs_state = new_state
            else:
                current_stage.append((addr, chunk_data))

        pos += length

    return stages


def _switch_to_iq_mode(bootloader_dev):
    """Upload firmware to switch IC-R8600 from bootloader (0x0022) to I/Q mode (0x0023)."""
    print("IC-R8600 detected in bootloader mode (PID 0x0022)")
    print("Uploading firmware to enable I/Q streaming...")

    # Find and extract firmware
    spt_path = _find_firmware_file()
    if not spt_path:
        raise RuntimeError(
            "IC-R8600 firmware file not found.\n"
            "Please copy IC-R8600_usb_iq.spt to ~/dev/ or ~/.local/share/pyfm/\n"
            "This file is from the Icom USB I/Q Package for HDSDR."
        )

    stages = _extract_firmware_from_spt(spt_path)
    if len(stages) < 2:
        raise RuntimeError(f"Invalid firmware file: expected 2 stages, got {len(stages)}")

    # Detach kernel driver
    try:
        if bootloader_dev.is_kernel_driver_active(0):
            bootloader_dev.detach_kernel_driver(0)
    except (usb.core.USBError, NotImplementedError):
        pass

    # Upload both firmware stages
    print(f"  Loading stage 1 ({sum(len(d) for _, d in stages[0])} bytes)...")
    _upload_firmware_stage(bootloader_dev, stages[0])
    time.sleep(0.5)

    # Re-find device (might have re-enumerated)
    dev = usb.core.find(idVendor=ICOM_VID)
    if dev is None:
        raise RuntimeError("Device lost after stage 1 firmware upload")

    if dev.idProduct == R8600_PID_READY:
        print("  Device switched to I/Q mode!")
        return dev

    # Need stage 2
    try:
        if dev.is_kernel_driver_active(0):
            dev.detach_kernel_driver(0)
    except (usb.core.USBError, NotImplementedError):
        pass

    print(f"  Loading stage 2 ({sum(len(d) for _, d in stages[1])} bytes)...")
    _upload_firmware_stage(dev, stages[1])

    # Wait for device to re-enumerate as PID 0x0023
    # This can take a few seconds as USB re-enumerates and udev applies rules
    print("  Waiting for device to initialize...")
    ready_dev = None
    for attempt in range(10):
        time.sleep(0.5)
        ready_dev = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_READY)
        if ready_dev is not None:
            break

    if ready_dev is None:
        raise RuntimeError("Firmware upload completed but device did not switch to I/Q mode")

    # Extra delay for device to fully initialize and udev rules to apply
    time.sleep(0.5)

    print("  Device switched to I/Q mode!")
    return ready_dev


class IcomR8600:
    """
    IC-R8600 I/Q streaming interface compatible with BB60D API.

    Provides the same methods as BB60D for use with pyfm's FMRadio class.
    """

    # Frequency limits - R8600 covers 10 kHz to 3 GHz
    MIN_FREQ = 10e3
    MAX_FREQ = 3000e6

    # FM broadcast band
    FM_MIN_FREQ = 88.0e6
    FM_MAX_FREQ = 108.0e6
    FM_STEP = 100e3

    DEFAULT_FREQ = 89.9e6

    def __init__(self, use_24bit=False):
        self.device = None
        self.frequency = self.DEFAULT_FREQ
        self.streaming_mode = None
        self.total_sample_loss = 0
        self.recent_sample_loss = 0

        # I/Q streaming state
        self.iq_sample_rate = 0
        self.iq_bandwidth = 0
        self._use_24bit = use_24bit
        self._bit_depth = 16
        self._bytes_per_sample = 4  # 16-bit: 2 bytes I + 2 bytes Q

        # I/Q gain - applied to normalized samples
        # For FM demodulation, this should be 1.0 since phase detection is
        # amplitude-independent. Higher gain only needed if downstream
        # processing requires it.
        self._iq_gain = 1.0

        # DC offset tracking - per Icom I/Q Reference Guide, there's a DC
        # component in the I/Q data that varies with sample rate. We use
        # an exponential moving average to track and remove it.
        self._dc_offset = 0.0 + 0.0j  # Complex DC offset estimate
        self._dc_alpha = 0.001  # EMA smoothing factor (lower = slower tracking)

        # I/Q data buffer and thread
        self._iq_buffer = []
        self._iq_byte_buf = bytearray()  # Persistent byte buffer for parsing
        self._iq_aligned = False  # Whether stream is aligned to sample boundaries
        self._samples_since_sync = 0  # Counter for sync interval verification
        self._sync_misses = 0  # Counter for missing sync patterns (should stay 0)
        self._sync_short_buf = 0  # Counter for sync checks with insufficient buffer
        self._flush_during_fetch = 0  # Counter for flush while fetch_iq active
        self._fetch_active = False  # Flag to track if fetch_iq is running
        self._initial_aligns = 0  # Counter for initial alignment attempts
        self._sync_invalid_24 = 0  # Counter for 24-bit samples rejected as invalid
        self._fetch_last_ms = 0.0  # Last fetch_iq duration in ms
        self._fetch_slow_count = 0  # Count of slow fetch_iq calls
        self._fetch_slow_threshold_ms = 50.0  # Slow fetch threshold in ms
        self._civ_timeouts = 0  # Count of CI-V response timeouts
        self._iq_lock = threading.Lock()
        self._iq_thread = None
        self._running = False

    def open(self):
        """Open connection to IC-R8600 I/Q USB port."""
        firmware_uploaded = False

        # First try to find device in ready state (PID 0x0023)
        self.device = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_READY)

        if self.device is None:
            # Check for bootloader mode (PID 0x0022) - needs firmware upload
            bootloader_dev = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_BOOTLOADER)
            if bootloader_dev is not None:
                self.device = _switch_to_iq_mode(bootloader_dev)
                firmware_uploaded = True
            else:
                raise RuntimeError(
                    f"IC-R8600 not found.\n"
                    "Make sure the radio is connected via the USB I/Q port (not the main USB port)."
                )

        # After firmware upload, the device handle may be stale - re-find it
        if firmware_uploaded:
            time.sleep(0.5)
            self.device = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_READY)
            if self.device is None:
                raise RuntimeError("Device lost after firmware upload")

        # Retry loop for USB initialization (device may need time to settle)
        last_error = None
        for attempt in range(3):
            try:
                # Detach kernel driver if needed
                try:
                    if self.device.is_kernel_driver_active(0):
                        self.device.detach_kernel_driver(0)
                except (usb.core.USBError, NotImplementedError):
                    pass

                # Set configuration
                self.device.set_configuration()

                # Claim interface 0 for communication
                usb.util.claim_interface(self.device, 0)

                # Success
                last_error = None
                break

            except usb.core.USBError as e:
                last_error = e
                if attempt < 2:
                    time.sleep(0.5)
                    # Re-find device in case it re-enumerated
                    self.device = usb.core.find(idVendor=ICOM_VID, idProduct=R8600_PID_READY)
                    if self.device is None:
                        raise RuntimeError("Device lost during initialization")

        if last_error is not None:
            raise RuntimeError(f"Failed to initialize USB device: {last_error}")

        # Get device info
        try:
            manufacturer = usb.util.get_string(self.device, self.device.iManufacturer)
            product = usb.util.get_string(self.device, self.device.iProduct)
            print(f"Connected to {manufacturer} {product}")
        except usb.core.USBError:
            print(f"Connected to IC-R8600 (VID={ICOM_VID:04x} PID={R8600_PID_READY:04x})")

        return self.device

    def close(self):
        """Close connection and disable I/Q mode."""
        self._running = False
        if self._iq_thread:
            self._iq_thread.join(timeout=1.0)
            self._iq_thread = None

        if self.device:
            try:
                # Disable I/Q output first, with delay for processing
                self._send_command(_build_civ_command([0x1A, 0x13, 0x01, 0x00]), timeout=1000)
                time.sleep(0.2)
                # Disable I/Q mode (turn off REMOTE LED)
                self._send_command(_build_civ_command([0x1A, 0x13, 0x00, 0x00]), timeout=1000)
                time.sleep(0.2)
            except (usb.core.USBError, RuntimeError):
                pass

            try:
                usb.util.release_interface(self.device, 0)
            except usb.core.USBError:
                pass

            try:
                usb.util.dispose_resources(self.device)
            except usb.core.USBError:
                pass
            self.device = None

    def _send_command(self, cmd_bytes, timeout=2000):
        """Send CI-V command and return response."""
        try:
            self.device.write(EP_CMD_OUT, cmd_bytes, timeout=timeout)
        except usb.core.USBError as e:
            raise RuntimeError(f"Failed to send command: {e}")

        try:
            response = self.device.read(EP_RESP_IN, 64, timeout=timeout)
            return bytes(response)
        except usb.core.USBTimeoutError:
            self._civ_timeouts += 1
            return None
        except usb.core.USBError as e:
            raise RuntimeError(f"Failed to read response: {e}")

    def configure_iq_streaming(self, freq=None, sample_rate=250000):
        """
        Configure IC-R8600 for I/Q streaming.

        Args:
            freq: Center frequency in Hz
            sample_rate: Desired sample rate (will use nearest available)
        """
        if freq is not None:
            self.frequency = freq

        self.frequency = max(self.MIN_FREQ, min(self.MAX_FREQ, self.frequency))

        # Find nearest available sample rate
        # For FM, we need at least 250 kHz, so 480 KSPS is minimum
        # Select rate table based on bit depth preference
        if self._use_24bit:
            rate_table = SAMPLE_RATES_24BIT
        else:
            rate_table = SAMPLE_RATES

        available_rates = sorted(rate_table.keys())
        chosen_rate = available_rates[0]
        for rate in available_rates:
            if rate >= sample_rate:
                chosen_rate = rate
                break
        else:
            chosen_rate = available_rates[-1]  # Use highest if requested is higher

        bit_depth, rate_code = rate_table[chosen_rate]
        self._bit_depth = 16 if bit_depth == 0x00 else 24
        self._bytes_per_sample = 4 if self._bit_depth == 16 else 6

        # First, ensure clean state by disabling any existing I/Q output
        # These may timeout if already disabled - that's OK
        try:
            self._send_command(_build_civ_command([0x1A, 0x13, 0x01, 0x00]), timeout=500)
        except RuntimeError:
            pass
        time.sleep(0.1)

        # Enable I/Q mode (REMOTE LED on)
        resp = self._send_command(_build_civ_command([0x1A, 0x13, 0x00, 0x01]))
        if resp and CIV_NG in resp:
            raise RuntimeError("Failed to enable I/Q mode")

        # Set frequency
        bcd_freq = _freq_to_bcd(self.frequency)
        resp = self._send_command(_build_civ_command([0x05] + list(bcd_freq)))
        if resp and CIV_NG in resp:
            raise RuntimeError("Failed to set frequency")

        # Set optimal RF settings for I/Q streaming (per HDSDR capture)
        # These settings revert when exiting I/Q mode (per Icom I/Q Reference Guide p.13)
        # ATT=0dB and RF Gain=255 for maximum sensitivity
        self._send_command(_build_civ_command([0x11, 0x00]))  # Attenuator OFF
        self._send_command(_build_civ_command([0x14, 0x02, 0x02, 0x55]))  # RF Gain 255 (BCD: 0255)

        # Query and display current RF settings
        rf_settings = self.query_rf_settings()
        if rf_settings:
            print(f"RF Settings: Gain={rf_settings.get('rf_gain', '?')}, "
                  f"Att={rf_settings.get('attenuator', '?')}dB, "
                  f"Preamp={'ON' if rf_settings.get('preamp') else 'OFF'}")

        # Enable HF BPF (for HF frequencies)
        if self.frequency < 30e6:
            self._send_command(_build_civ_command([0x1A, 0x13, 0x02, 0x01]))

        # Enable I/Q output with chosen sample rate
        resp = self._send_command(_build_civ_command([0x1A, 0x13, 0x01, 0x01, bit_depth, rate_code]))
        if resp and CIV_NG in resp:
            raise RuntimeError("Failed to enable I/Q output")

        self.iq_sample_rate = chosen_rate
        # Bandwidth is approximately 87% of sample rate for R8600
        self.iq_bandwidth = chosen_rate * 0.87
        self.streaming_mode = "iq"

        # Initialize buffers for new streaming session
        self._iq_byte_buf = bytearray()
        self._iq_aligned = False
        self._samples_since_sync = 0

        # Start I/Q reader thread
        self._running = True
        self._iq_buffer = []
        self._iq_thread = threading.Thread(target=self._iq_reader_loop, daemon=True)
        self._iq_thread.start()

        print(f"IQ Streaming: {self.iq_sample_rate/1e6:.3f} MS/s ({self._bit_depth}-bit), BW: {self.iq_bandwidth/1e3:.1f} kHz, Gain: {self._iq_gain:.0f}x")

    def _iq_reader_loop(self):
        """Background thread to continuously read I/Q data from USB."""
        # Set SCHED_FIFO for this I/Q reader thread
        try:
            param = os.sched_param(50)
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
        except (PermissionError, OSError):
            pass  # Silently fall back to normal scheduling

        while self._running:
            try:
                # Read large chunks for efficiency
                data = self.device.read(EP_IQ_IN, 65536, timeout=1000)
                if data:
                    with self._iq_lock:
                        self._iq_buffer.append(bytes(data))
            except usb.core.USBTimeoutError:
                continue
            except usb.core.USBError:
                if self._running:
                    self.recent_sample_loss += 1
                    self.total_sample_loss += 1
                break

    def fetch_iq(self, num_samples=8192):
        """
        Fetch IQ samples for software demodulation.

        Uses byte-level parsing with sync pattern detection to handle
        sync patterns that may straddle USB read boundaries.

        Args:
            num_samples: Number of IQ samples to fetch

        Returns:
            numpy array of complex64 IQ samples (always exactly num_samples, padded with zeros if needed)
        """
        if self.streaming_mode != "iq":
            raise RuntimeError("Device not in IQ streaming mode")

        start_time = time.perf_counter()
        self._fetch_active = True

        # Define sync pattern and sample parameters based on bit depth
        # Per Icom I/Q Reference Guide:
        # 16-bit sync: 0x8000, 0x8000 (little-endian: 00 80 00 80)
        # 24-bit sync: 0x8000, 0x8001, 0x8002 (little-endian: 00 80 01 80 02 80)
        if self._bit_depth == 16:
            sync_bytes = b"\x00\x80\x00\x80"
            sample_size = 4
        else:
            sync_bytes = b"\x00\x80\x01\x80\x02\x80"
            sample_size = 6

        bytes_needed = num_samples * sample_size + len(sync_bytes) * (num_samples // 1024 + 2)

        # Collect raw bytes from USB reader thread
        timeout = time.time() + 1.0
        while len(self._iq_byte_buf) < bytes_needed and time.time() < timeout:
            with self._iq_lock:
                while self._iq_buffer:
                    self._iq_byte_buf += self._iq_buffer.pop(0)
            if len(self._iq_byte_buf) < bytes_needed:
                time.sleep(0.001)

        # Align to sample boundary on first parse (critical for 24-bit)
        # 16-bit mode works without explicit alignment; 24-bit requires it
        # We verify alignment by checking that the next sync appears at expected interval
        if self._bit_depth == 24 and not self._iq_aligned:
            sync_interval = SYNC_INTERVAL.get(self.iq_sample_rate, 1024)
            expected_gap = sync_interval * sample_size  # bytes between sync patterns
            search_start = 0

            while time.time() < timeout:
                # Collect more data if needed
                with self._iq_lock:
                    while self._iq_buffer:
                        self._iq_byte_buf += self._iq_buffer.pop(0)

                # Find a sync pattern
                idx = self._iq_byte_buf.find(sync_bytes, search_start)
                if idx == -1:
                    # No sync found - keep tail bytes that might be partial sync
                    keep = max(0, len(sync_bytes) - 1)
                    if len(self._iq_byte_buf) > keep:
                        self._iq_byte_buf = self._iq_byte_buf[-keep:]
                    search_start = 0
                    time.sleep(0.001)
                    continue

                # Check if we have enough data to verify next sync
                next_sync_pos = idx + len(sync_bytes) + expected_gap
                if next_sync_pos + len(sync_bytes) > len(self._iq_byte_buf):
                    # Need more data to verify - wait and retry
                    time.sleep(0.001)
                    continue

                # Verify next sync is at expected position
                if self._iq_byte_buf[next_sync_pos:next_sync_pos + len(sync_bytes)] == sync_bytes:
                    # Verified! Position after first sync
                    self._iq_byte_buf = self._iq_byte_buf[idx + len(sync_bytes):]
                    self._iq_aligned = True
                    self._initial_aligns += 1
                    break
                else:
                    # False sync - skip past it and keep searching
                    search_start = idx + 1

            if not self._iq_aligned:
                # Still no sync after timeout - return zeros
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                self._fetch_last_ms = duration_ms
                if duration_ms > self._fetch_slow_threshold_ms:
                    self._fetch_slow_count += 1
                self._fetch_active = False
                return np.zeros(num_samples, dtype=np.complex64)

        # Parse samples from byte buffer using sync-based framing
        buf = self._iq_byte_buf
        idx = 0
        buf_len = len(buf)
        parsed_i = []
        parsed_q = []
        sync_len = len(sync_bytes)

        # Get expected sync interval for this sample rate
        sync_interval = SYNC_INTERVAL.get(self.iq_sample_rate, 1024)
        expected_gap = sync_interval * sample_size  # bytes between sync patterns

        while len(parsed_i) < num_samples and buf_len - idx >= sample_size:
            # Check for sync pattern at current position
            if buf[idx:idx + sync_len] == sync_bytes:
                if self._bit_depth == 24 and 0 < self._samples_since_sync < sync_interval:
                    # Early sync in 24-bit: verify next sync before accepting
                    next_sync_pos = idx + sync_len + expected_gap
                    if next_sync_pos + sync_len > buf_len:
                        # Need more data to verify; stop parsing for now
                        break
                    if buf[next_sync_pos:next_sync_pos + sync_len] != sync_bytes:
                        # Likely false sync; treat as data
                        pass
                    else:
                        idx += sync_len
                        self._samples_since_sync = 0
                        continue
                else:
                    idx += sync_len
                    self._samples_since_sync = 0
                    continue

            # Periodic alignment check: if we've parsed enough samples,
            # we should be near a sync. If not at expected position, search for it.
            if self._samples_since_sync > 0 and self._samples_since_sync >= sync_interval:
                # Track if we have insufficient buffer for reliable sync search
                if buf_len - idx < sync_len * 3:
                    self._sync_short_buf += 1
                # We should have seen a sync by now - search for it
                sync_pos = buf[idx:idx + sync_len * 3].find(sync_bytes)
                if sync_pos == -1:
                    # No sync where expected - we're misaligned, search further
                    self._sync_misses += 1
                    sync_pos = buf[idx:].find(sync_bytes)
                    if sync_pos != -1:
                        idx += sync_pos + sync_len
                        self._samples_since_sync = 0
                        continue
                    else:
                        # No sync in buffer - skip some bytes and try again
                        idx += sample_size
                        continue
                elif sync_pos == 0:
                    # Sync right here - good
                    idx += sync_len
                    self._samples_since_sync = 0
                    continue
                else:
                    # Sync nearby but not at position 0 - we drifted
                    self._sync_misses += 1
                    idx += sync_pos + sync_len
                    self._samples_since_sync = 0
                    continue

            if self._bit_depth == 16:
                i = int.from_bytes(buf[idx:idx + 2], "little", signed=True)
                q = int.from_bytes(buf[idx + 2:idx + 4], "little", signed=True)
                idx += 4
                # Track samples consumed, even if later discarded
                self._samples_since_sync += 1
                # Filter invalid samples (value -32768 only appears in sync)
                if i == -32768 or q == -32768:
                    continue
            else:
                i_raw = buf[idx:idx + 3]
                q_raw = buf[idx + 3:idx + 6]
                idx += 6
                # Track samples consumed, even if later discarded
                self._samples_since_sync += 1
                # 24-bit little-endian to int
                i = i_raw[0] | (i_raw[1] << 8) | (i_raw[2] << 16)
                q = q_raw[0] | (q_raw[1] << 8) | (q_raw[2] << 16)
                # Sign extend from 24-bit
                if i & 0x800000:
                    i -= 0x1000000
                if q & 0x800000:
                    q -= 0x1000000
                # Per Icom spec, valid range is -8387967 to +8387966
                # Values outside this are likely sync bytes misinterpreted
                if i < -8387967 or i > 8387966 or q < -8387967 or q > 8387966:
                    self._sync_invalid_24 += 1
                    continue
            parsed_i.append(i)
            parsed_q.append(q)

        # Remove parsed bytes from buffer
        if idx > 0:
            self._iq_byte_buf = self._iq_byte_buf[idx:]

        # Convert to complex samples
        if parsed_i:
            if self._bit_depth == 16:
                i_arr = np.asarray(parsed_i, dtype=np.int16).astype(np.float32)
                q_arr = np.asarray(parsed_q, dtype=np.int16).astype(np.float32)
                iq = (i_arr + 1j * q_arr) / 32768.0
            else:
                i_arr = np.asarray(parsed_i, dtype=np.int32).astype(np.float32)
                q_arr = np.asarray(parsed_q, dtype=np.int32).astype(np.float32)
                iq = (i_arr + 1j * q_arr) / 8388608.0

            # Remove DC offset (per Icom I/Q Reference Guide)
            block_dc = np.mean(iq)
            self._dc_offset = self._dc_alpha * block_dc + (1 - self._dc_alpha) * self._dc_offset
            iq = iq - self._dc_offset

            # Apply gain
            if self._iq_gain != 1.0:
                iq = iq * self._iq_gain

            iq = iq.astype(np.complex64)
        else:
            iq = np.zeros(0, dtype=np.complex64)

        # Track sample loss
        if len(iq) < num_samples:
            self.recent_sample_loss += 1
            self.total_sample_loss += 1
        else:
            self.recent_sample_loss = 0

        # Pad with zeros if we didn't get enough samples
        if len(iq) < num_samples:
            iq = np.concatenate([iq, np.zeros(num_samples - len(iq), dtype=np.complex64)])

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        self._fetch_last_ms = duration_ms
        if duration_ms > self._fetch_slow_threshold_ms:
            self._fetch_slow_count += 1
        self._fetch_active = False
        return iq[:num_samples]

    def flush_iq(self):
        """Flush stale IQ data from the buffer."""
        if self._fetch_active:
            self._flush_during_fetch += 1
        with self._iq_lock:
            self._iq_buffer.clear()
        self._iq_byte_buf = bytearray()
        self._iq_aligned = False  # Force re-alignment after flush
        self._samples_since_sync = 0
        # Reset DC offset estimate (DC varies with frequency per Icom docs)
        self._dc_offset = 0.0 + 0.0j
        self.total_sample_loss = 0
        self.recent_sample_loss = 0

    def set_frequency(self, freq):
        """Change the tuned frequency."""
        self.frequency = max(self.MIN_FREQ, min(self.MAX_FREQ, freq))

        if self.streaming_mode == "iq" and self.device:
            bcd_freq = _freq_to_bcd(self.frequency)
            resp = self._send_command(_build_civ_command([0x05] + list(bcd_freq)))
            if resp and CIV_NG in resp:
                raise RuntimeError("Failed to set frequency")

    def tune_up(self):
        """Tune up by FM step (100 kHz)."""
        new_freq = self.frequency + self.FM_STEP
        if new_freq > self.FM_MAX_FREQ:
            new_freq = self.FM_MIN_FREQ
        self.set_frequency(new_freq)
        # Reconfigure to apply new frequency
        if self.streaming_mode == "iq":
            self.flush_iq()

    def tune_down(self):
        """Tune down by FM step (100 kHz)."""
        new_freq = self.frequency - self.FM_STEP
        if new_freq < self.FM_MIN_FREQ:
            new_freq = self.FM_MAX_FREQ
        self.set_frequency(new_freq)
        if self.streaming_mode == "iq":
            self.flush_iq()

    @property
    def frequency_mhz(self):
        """Current frequency in MHz."""
        return self.frequency / 1e6

    @property
    def iq_gain(self):
        """Current I/Q software gain multiplier."""
        return self._iq_gain

    @iq_gain.setter
    def iq_gain(self, value):
        """Set I/Q software gain multiplier."""
        self._iq_gain = max(1.0, float(value))

    # ========== RF Gain Management ==========

    def query_rf_settings(self):
        """
        Query current RF settings from the radio.

        Returns:
            dict with rf_gain, attenuator, preamp, ip_plus settings
        """
        settings = {}

        # Query RF gain (14 02) - response includes 2-byte BCD value
        resp = self._send_command(_build_civ_command([0x14, 0x02]))
        if resp and len(resp) >= 8 and CIV_OK not in resp:
            # Response format: FE FE E0 96 14 02 XX XX FD
            # XX XX is BCD value 0000-0255
            try:
                idx = resp.index(0x14)
                if idx + 3 < len(resp):
                    high = resp[idx + 2]
                    low = resp[idx + 3]
                    # BCD decode: 0x02 0x55 = 255
                    settings['rf_gain'] = (high >> 4) * 1000 + (high & 0x0F) * 100 + (low >> 4) * 10 + (low & 0x0F)
            except (ValueError, IndexError):
                pass

        # Query attenuator (11) - response includes setting byte
        resp = self._send_command(_build_civ_command([0x11]))
        if resp and len(resp) >= 6:
            try:
                idx = resp.index(0x11)
                if idx + 1 < len(resp):
                    att_val = resp[idx + 1]
                    # 0x00=0dB, 0x10=10dB, 0x20=20dB, 0x30=30dB
                    settings['attenuator'] = (att_val >> 4) * 10
            except (ValueError, IndexError):
                pass

        # Query preamp (16 02)
        resp = self._send_command(_build_civ_command([0x16, 0x02]))
        if resp and len(resp) >= 7:
            try:
                idx = resp.index(0x16)
                if idx + 2 < len(resp) and resp[idx + 1] == 0x02:
                    settings['preamp'] = resp[idx + 2] == 0x01
            except (ValueError, IndexError):
                pass

        # Query IP+ (16 65)
        resp = self._send_command(_build_civ_command([0x16, 0x65]))
        if resp and len(resp) >= 7:
            try:
                idx = resp.index(0x16)
                if idx + 2 < len(resp) and resp[idx + 1] == 0x65:
                    settings['ip_plus'] = resp[idx + 2] == 0x01
            except (ValueError, IndexError):
                pass

        return settings

    def set_rf_gain(self, level):
        """
        Set RF gain level.

        Args:
            level: 0-255 (0=min, 255=max)

        Note: RF gain commands may be rejected if I/Q output is actively streaming.
        Set RF gain before enabling I/Q output, or temporarily disable I/Q output.
        """
        level = max(0, min(255, int(level)))
        # Convert to 4-digit BCD: 255 -> 0x02 0x55 (digits: 0,2,5,5)
        d0 = level // 1000         # thousands
        d1 = (level // 100) % 10   # hundreds
        d2 = (level // 10) % 10    # tens
        d3 = level % 10            # ones
        high = (d0 << 4) | d1      # 0x02 for 255
        low = (d2 << 4) | d3       # 0x55 for 255
        resp = self._send_command(_build_civ_command([0x14, 0x02, high, low]))
        if resp and CIV_NG in resp:
            raise RuntimeError(f"Failed to set RF gain to {level}")

    def set_attenuator(self, db):
        """
        Set attenuator level.

        Args:
            db: 0, 10, 20, or 30 dB
        """
        if db not in (0, 10, 20, 30):
            raise ValueError("Attenuator must be 0, 10, 20, or 30 dB")
        att_val = (db // 10) << 4  # 0->0x00, 10->0x10, 20->0x20, 30->0x30
        resp = self._send_command(_build_civ_command([0x11, att_val]))
        if resp and CIV_NG in resp:
            raise RuntimeError(f"Failed to set attenuator to {db} dB")

    def set_preamp(self, enabled):
        """
        Enable or disable preamplifier.

        Args:
            enabled: True to enable, False to disable
        """
        resp = self._send_command(_build_civ_command([0x16, 0x02, 0x01 if enabled else 0x00]))
        if resp and CIV_NG in resp:
            raise RuntimeError(f"Failed to set preamp to {enabled}")

    def set_ip_plus(self, enabled):
        """
        Enable or disable IP+ (Intercept Point Plus).

        Args:
            enabled: True to enable, False to disable
        """
        resp = self._send_command(_build_civ_command([0x16, 0x65, 0x01 if enabled else 0x00]))
        if resp and CIV_NG in resp:
            raise RuntimeError(f"Failed to set IP+ to {enabled}")

    def auto_gain(self):
        """
        Automatically adjust RF gain based on signal strength.
        For FM broadcast, aims for moderate gain to avoid clipping.
        """
        # Query S-meter (15 02) to check signal level
        resp = self._send_command(_build_civ_command([0x15, 0x02]))
        if resp:
            try:
                idx = resp.index(0x15)
                if idx + 2 < len(resp) and resp[idx + 1] == 0x02:
                    # S-meter value is 2-byte BCD, 0000-0255
                    high = resp[idx + 2]
                    low = resp[idx + 3] if idx + 3 < len(resp) else 0
                    s_meter = (high >> 4) * 1000 + (high & 0x0F) * 100 + (low >> 4) * 10 + (low & 0x0F)
                    print(f"S-Meter: {s_meter}")

                    # Adjust RF gain based on signal strength
                    # Strong signal (>150) -> reduce gain
                    # Weak signal (<50) -> increase gain
                    settings = self.query_rf_settings()
                    current_gain = settings.get('rf_gain', 128)

                    if s_meter > 200:
                        new_gain = max(0, current_gain - 50)
                    elif s_meter > 150:
                        new_gain = max(0, current_gain - 20)
                    elif s_meter < 30:
                        new_gain = min(255, current_gain + 20)
                    elif s_meter < 50:
                        new_gain = min(255, current_gain + 10)
                    else:
                        new_gain = current_gain

                    if new_gain != current_gain:
                        print(f"Adjusting RF gain: {current_gain} -> {new_gain}")
                        self.set_rf_gain(new_gain)
            except (ValueError, IndexError):
                pass


def get_api_version():
    """Return version string for compatibility with BB60D interface."""
    return "IC-R8600 USB I/Q 1.0"
