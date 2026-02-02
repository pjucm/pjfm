pyfm is a software-defined FM radio receiver supporting the SignalHound BB60D and Icom IC-R8600.

It receives broadcast FM (88-108 MHz) and NOAA Weather Radio (162 MHz), performs all demodulation
in software, and plays audio through the default Linux audio device.

## Supported Hardware

- **SignalHound BB60D** - Spectrum analyzer with I/Q streaming at 312.5 kHz
- **Icom IC-R8600** - Communications receiver with USB I/Q output at 240 kHz - 5.12 MHz

## Running

```bash
# With BB60D (default)
./pyfm.py [frequency_mhz]

# With IC-R8600
./pyfm.py --icom [frequency_mhz]
```

## Controls

| Key | Function |
|-----|----------|
| ←/→ | Tune down/up (100 kHz FM, 25 kHz Weather) |
| ↑/↓ | Volume up/down |
| 1-5 | Recall preset (FM) or WX1-WX7 (Weather) |
| !@#$% | Set preset to current frequency (FM mode) |
| w | Toggle Weather radio mode (NBFM) |
| r | Toggle RDS decoding (FM mode) |
| b | Toggle bass boost (+3 dB) |
| t | Toggle treble boost (+3 dB) |
| a | Toggle spectrum analyzer |
| Q | Toggle squelch |
| q | Quit |

## Dependencies

```bash
pip install numpy scipy sounddevice rich pyusb  # pyusb for IC-R8600
```

## Architecture

### Core Components

- `pyfm.py` - Main CLI application with terminal UI (Rich), audio playback, spectrum analyzer
- `panadapter.py` - PyQt5 GUI with spectrum/waterfall display and FM demodulation
- `demodulator.py` - FM stereo decoder (FMStereoDecoder) and NBFM decoder (NBFMDecoder)
- `rds_decoder.py` - RDS/RBDS decoding (station ID, program type, radio text, clock)
- `bb60d.py` - SignalHound BB60D I/Q streaming interface
- `icom_r8600.py` - IC-R8600 USB I/Q interface with CI-V control

### pyfm.py Architecture

Main classes:
- **FMRadio** - Application controller, coordinates device, demodulators, audio, and UI
- **AudioPlayer** - Ring buffer audio output with adaptive PI rate control
- **SpectrumAnalyzer** - 16-band audio FFT with peak hold (ModPlug-style bars)

Key features:
- Adaptive rate control using PI controller to match audio card clock drift
- Real-time scheduling (SCHED_FIFO) when CAP_SYS_NICE is available
- Automatic RDS enable when pilot tone detected
- Persistent config (presets, tone settings, last frequency)

### IC-R8600 I/Q Interface (icom_r8600.py)

**Initialization**:
- Automatic Cypress FX2 firmware upload when device in bootloader mode (PID 0x0022)
- Firmware file: `IC-R8600_usb_iq.spt` (from Icom USB I/Q Package for HDSDR)
- Search paths: script directory, `~/dev/`, `~/.local/share/pyfm/`

**Sample Rates**: 240 kHz, 480 kHz, 960 kHz, 1.92 MHz, 3.84 MHz, 5.12 MHz
**Bit Depth**: 16-bit (default), 24-bit (available at all rates except 5.12 MHz)

**I/Q Stream Processing** (per Icom I/Q Reference Guide):
- Sync pattern filtering: 0x8000,0x8000 (16-bit) or 0x8000,0x8001,0x8002 (24-bit)
- DC offset removal via EMA tracking (alpha=0.001)
- Sync interval verification for alignment validation
- Background reader thread for continuous USB bulk transfers

**CI-V Protocol Commands**:
- 0x1A 0x13: I/Q mode enable/disable, sample rate selection
- 0x05: Frequency tuning (BCD format)
- 0x14 0x02: RF gain control (0-255)
- 0x11: Attenuator (0/10/20/30 dB)
- 0x16 0x02: Preamplifier on/off
- 0x16 0x65: IP+ (Intercept Point Plus) on/off

### Signal Processing

**FM Demodulation**: Quadrature discriminator extracts instantaneous frequency:
```python
baseband = angle(s[n] * conj(s[n-1])) * (sample_rate / (2π * deviation))
```

**Stereo Decoding** (FMStereoDecoder):
- 19 kHz pilot BPF with Kaiser window (201 taps)
- Pilot-squaring for 38 kHz carrier regeneration
- L+R/L-R bandpass filtering and matrix decoding
- SNR-based stereo blend (mono below 15 dB, full stereo above 30 dB)
- 75 µs de-emphasis filter

**Weather Radio** (NBFMDecoder): NBFM with 5 kHz deviation for NOAA NWS (162.400-162.550 MHz)

## Panadapter GUI (panadapter.py)

PyQt5-based spectrum analyzer with waterfall display for Weather Radio and FM broadcast.

### Running

```bash
# With BB60D (default)
python panadapter.py [--freq FREQ_MHZ]

# With IC-R8600
python panadapter.py --icom [--freq FREQ_MHZ]
python panadapter.py --icom --24bit                 # 24-bit I/Q
python panadapter.py --icom --sample-rate 960000    # Custom sample rate
```

### Architecture

**Main Classes**:
- **MainWindow** - Application window with mode switching (Weather/FM Broadcast)
- **SpectrumWidget** - Real-time FFT spectrum with click-to-tune
- **WaterfallWidget** - Scrolling waterfall display with viridis-style colormap
- **SMeterWidget** - S-meter display (S9 = -93 dBm, 6 dB/S-unit)
- **DataThread** - Background I/Q acquisition with rate-limited display updates (~30 fps)
- **AudioOutput** - Ring buffer audio output with volume control

**Demodulators**:
- **NBFMDemodulator** - NBFM for weather radio (5 kHz deviation, optional hum filter)
- **WBFMStereoDemodulator** - Wraps FMStereoDecoder with decimation for higher sample rates

### Features

- Dual mode: Weather Radio (NBFM) and FM Broadcast (WBFM stereo)
- Real-time spectrum with exponential averaging and peak hold
- Click-to-tune on spectrum or waterfall
- S-meter with dBm readout
- Stereo indicator and SNR display (FM mode)
- NOAA preset buttons (WX1-WX7)
- Squelch with visual indicator
- Persistent config (device, frequency, mode, spectrum span)

### Dependencies

```bash
pip install PyQt5 pyqtgraph numpy scipy sounddevice
pip install pyusb  # For IC-R8600
```
