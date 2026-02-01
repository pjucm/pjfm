# pyfm

A real-time FM broadcast receiver with software-defined stereo demodulation and RDS decoding, supporting both SignalHound BB60D and Icom IC-R8600.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-proprietary-red.svg)

## Overview

pyfm is a command-line FM radio application that receives broadcast FM signals (88-108 MHz) and NOAA Weather Radio (162 MHz), performs all demodulation in software, and plays audio through the default Linux audio device. It features a rich terminal UI with real-time signal metrics, a 16-band spectrum analyzer, and full RDS (Radio Data System) decoding.

## Features

- **Dual hardware support**: SignalHound BB60D and Icom IC-R8600
- **Real-time FM stereo reception** with automatic mono/stereo switching
- **NOAA Weather Radio** (NBFM mode for NWS frequencies)
- **RDS decoding** with station identification, program type, radio text, and clock time
- **GPU acceleration** via PyTorch (ROCm/CUDA) for FM demod, FIR filters, and resampling
- **16-band audio spectrum analyzer** with peak hold
- **Signal quality metrics** including S-meter, SNR, and pilot detection
- **Frequency presets** (5 programmable, saved to config)
- **Tone controls** with bass and treble boost
- **Squelch** for muting weak signals
- **Responsive terminal UI** built with Rich

## Supported Hardware

### SignalHound BB60D
- USB 3.0 spectrum analyzer
- I/Q streaming at 312.5 kHz (40 MHz / 128 decimation)
- Requires SignalHound BB API and Python bindings

### Icom IC-R8600
- Communications receiver with USB I/Q output
- Sample rates: 240 kHz to 5.12 MHz
- Automatic firmware upload on connection
- Requires `IC-R8600_usb_iq.spt` from Icom USB I/Q Package for HDSDR
- CI-V protocol for frequency control

## Technical Architecture

### IQ Streaming and FM Demodulation

pyfm captures raw RF samples via I/Q streaming and performs FM demodulation entirely in software:

```
IQ Samples (250-480 kHz) -> Quadrature Discriminator -> Baseband (0-100 kHz)
```

**Quadrature Discriminator**: FM demodulation using the classic quadrature method:

```python
product = samples[n] * conj(samples[n-1])
baseband = angle(product) * (sample_rate / (2 * pi * deviation))
```

This extracts the instantaneous frequency deviation from the phase difference between consecutive samples, normalized by the FM deviation (75 kHz for broadcast FM, 5 kHz for NBFM).

### FM Stereo Decoding

Broadcast FM stereo uses a pilot-tone multiplexing system defined by the FCC:

| Frequency Range | Content |
|----------------|---------|
| 0-15 kHz | L+R (mono-compatible sum) |
| 19 kHz | Pilot tone |
| 23-53 kHz | L-R on 38 kHz DSB-SC carrier |
| 57 kHz | RDS subcarrier (optional) |

**Pilot Detection**: A Kaiser-windowed FIR bandpass filter (18.5-19.5 kHz) extracts the 19 kHz pilot tone. Pilot presence is detected by RMS level threshold.

**Stereo Matrix Decoding**: The 38 kHz carrier is regenerated using pilot-squaring, which exploits the trig identity cos(2x) = 2cos²(x) - 1:
```
L-R carrier = 2 * pilot² - 1  // pilot-squaring
L-R = BPF(baseband, 23-53 kHz) * carrier * 2
Left  = (L+R) + (L-R)
Right = (L+R) - (L-R)
```

**De-emphasis**: A 75 μs de-emphasis filter compensates for the pre-emphasis applied at the transmitter, rolling off high frequencies to reduce noise.

### Weather Radio (NBFM)

NOAA Weather Radio uses narrowband FM with 5 kHz deviation on seven frequencies:

| Channel | Frequency |
|---------|-----------|
| WX1 | 162.550 MHz |
| WX2 | 162.400 MHz |
| WX3 | 162.475 MHz |
| WX4 | 162.425 MHz |
| WX5 | 162.450 MHz |
| WX6 | 162.500 MHz |
| WX7 | 162.525 MHz |

### RDS (Radio Data System) Decoding

RDS transmits digital data at 1187.5 bps on a 57 kHz subcarrier (3× pilot frequency), using BPSK modulation with differential encoding.

#### Signal Processing Chain

1. **Bandpass Filtering**: 4th-order Butterworth filter extracts 57 kHz ± 2.4 kHz
2. **Coherent Demodulation**: Pilot tone is tripled using the identity:
   ```
   cos(3θ) = 4cos³(θ) - 3cos(θ)
   ```
   This derives a phase-locked 57 kHz carrier from the 19 kHz pilot.
3. **Differential Decoding**: RDS uses differential encoding where data is represented by phase *changes*. Decoded by multiplying with a one-symbol-delayed version.
4. **Symbol Timing Recovery**: Gardner Timing Error Detector (TED) with PI loop filter tracks symbol boundaries at 1187.5 Hz.

#### Block Synchronization and Error Correction

RDS data is organized into 26-bit blocks (16 data + 10 checkword), grouped into 4-block groups:

| Block | Offset Word | Content |
|-------|-------------|---------|
| A | 0x0FC | PI code (station ID) |
| B | 0x198 | Group type, PTY, flags |
| C/C' | 0x168/0x350 | Varies by group type |
| D | 0x1B4 | Varies by group type |

**CRC Polynomial**: x¹⁰ + x⁸ + x⁷ + x⁵ + x⁴ + x³ + 1 (0x1B9)

**Error Correction**: The (26,16) shortened cyclic code can correct burst errors up to 5 bits.

#### Decoded RDS Data

| Group Type | Content |
|------------|---------|
| 0A/0B | Program Service name (8 chars, station branding) |
| 2A/2B | RadioText (64 chars, now playing info) |
| 4A | Clock Time and Date (UTC with offset) |

**PI Code Decoding**: North American RBDS encodes station call letters in the PI code:
- 0x1000-0x54A7: K stations (KAAA-KZZZ)
- 0x54A8-0x994F: W stations (WAAA-WZZZ)

### GPU Acceleration

pyfm supports GPU-accelerated signal processing via PyTorch (ROCm for AMD, CUDA for NVIDIA):

- **FM Demodulation**: Arctangent and differentiation on GPU
- **FIR Filter Bank**: Parallel pilot BPF, L+R LPF, L-R BPF computation
- **Polyphase Resampler**: Efficient sample rate conversion with ~80 dB stopband

GPU processing is enabled by default when available (toggle with 'G' key).

### IC-R8600 I/Q Processing

The IC-R8600 USB I/Q interface requires special handling per the Icom I/Q Reference Guide:

- **Firmware Upload**: Automatic Cypress FX2 firmware upload on connection
- **Sync Pattern Filtering**: Periodic sync markers (0x8000, 0x8000 for 16-bit) are removed
- **DC Offset Removal**: EMA-tracked DC component subtracted from samples
- **Sample Rates**: 240 kHz to 5.12 MHz (480 kHz typical for FM)
- **Bit Depth**: 16-bit (default), 24-bit supported but needs testing

### Audio Processing

**Sample Rate Conversion**: Polyphase resampling from I/Q rate to 48 kHz audio output.

**Tone Controls**: Biquad shelf filters designed using the Audio EQ Cookbook:
- Bass boost: +3 dB low shelf at 250 Hz
- Treble boost: +3 dB high shelf at 3.5 kHz

**Stereo Blend**: Automatically reduces stereo separation on weak signals to minimize noise:

| SNR | Blend |
|-----|-------|
| < 15 dB | 100% mono |
| 15-30 dB | Gradual blend |
| > 30 dB | 100% stereo |

**Soft Limiting**: Prevents harsh clipping on over-modulated stations using tanh saturation:
```python
output = tanh(input * 1.5) / tanh(1.5)
```

### Signal Quality Metrics

**S-Meter**: Calibrated to VHF/UHF standard (S9 = -93 dBm, 6 dB/S-unit)

**SNR Estimation**: Pilot-referenced measurement comparing 19 kHz pilot power to noise floor at 205-225 kHz (beyond HD Radio sidebands, true noise floor). The pilot is broadcast at a fixed level (~9% of deviation), providing a consistent reference.

## Usage

```bash
# With BB60D (default)
./pyfm.py [frequency_mhz]

# With IC-R8600
./pyfm.py --icom [frequency_mhz]

# Show version info
./pyfm.py --version
```

### Controls

| Key | Function |
|-----|----------|
| ←/→ | Tune down/up (100 kHz FM, 25 kHz Weather) |
| ↑/↓ | Volume up/down |
| 1-5 | Recall preset (FM) or 1-7 for WX channels (Weather) |
| !@#$% | Set preset to current frequency (Shift+1-5, FM mode) |
| w | Toggle Weather radio mode (NBFM) |
| r | Toggle RDS decoding (FM mode) |
| b | Toggle bass boost |
| t | Toggle treble boost |
| a | Toggle spectrum analyzer |
| G | Toggle GPU acceleration (FM mode) |
| Q | Toggle squelch |
| q | Quit |

### Configuration

Settings are saved to `pyfm.cfg`:
- Last tuned frequency (restored on startup)
- Frequency presets
- Tone control settings

## Requirements

### Hardware (one of)
- SignalHound BB60D spectrum analyzer with BB API
- Icom IC-R8600 with USB I/Q firmware (`IC-R8600_usb_iq.spt`)

### Software
- Python 3.8+
- Linux with ALSA/PulseAudio

### Python Dependencies

```bash
pip install numpy scipy sounddevice rich

# For IC-R8600 support
pip install pyusb

# For GPU acceleration (optional)
pip install torch  # ROCm or CUDA version
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SignalHound BB60D  OR  Icom IC-R8600                 │
│                        IQ Streaming @ 250-480 kHz                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          │  DC Offset Removal │ (R8600 only)
                          └─────────┬─────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Quadrature Discriminator                          │
│                   baseband = angle(s[n] * conj(s[n-1]))                 │
│                        (GPU accelerated if available)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
      │   L+R LPF    │      │  Pilot BPF   │      │  RDS BPF     │
      │   0-15 kHz   │      │   19 kHz     │      │   57 kHz     │
      └──────────────┘      └──────────────┘      └──────────────┘
              │                     │                     │
              │                     ▼                     ▼
              │              ┌──────────────┐      ┌──────────────┐
              │              │    Pilot     │      │  Coherent    │
              │              │   Squaring   │      │   Demod      │
              │              │  (2p²-1)     │      └──────────────┘
              │              └──────────────┘             │
              │                     │                     ▼
              │                     ▼              ┌──────────────┐
              │              ┌──────────────┐      │  Symbol      │
              │              │  L-R Demod   │      │  Recovery    │
              │              │  @ 38 kHz    │      │  (Gardner)   │
              │              └──────────────┘      └──────────────┘
              │                     │                     │
              ▼                     ▼                     ▼
      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
      │   Stereo     │      │   Stereo     │      │  Block Sync  │
      │   Matrix     │◄─────┤   Blend      │      │  & CRC       │
      │   L = S+D    │      │  (SNR-based) │      └──────────────┘
      │   R = S-D    │      └──────────────┘             │
      └──────────────┘                                   ▼
              │                                   ┌──────────────┐
              ▼                                   │  RDS Data    │
      ┌──────────────┐                            │  PS, RT, CT  │
      │  Resample    │                            │  PI → Call   │
      │  → 48 kHz    │                            └──────────────┘
      │ (Polyphase)  │
      └──────────────┘
              │
              ▼
      ┌──────────────┐
      │ De-emphasis  │
      │ Tone Control │
      │ Soft Limiter │
      └──────────────┘
              │
              ▼
      ┌──────────────┐
      │    Audio     │
      │   Output     │
      │   (48 kHz)   │
      └──────────────┘
```

## License

Copyright (c) 2026 Phil Jensen. All rights reserved.

## Acknowledgments

- SignalHound for the BB60D API
- Icom for the IC-R8600 I/Q Reference Guide
- The Rich library for terminal UI
- IEC 62106 (RDS specification)
