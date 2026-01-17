# pyfm

pyfm is a small, command line project for the SignalHound BB60D.

It is intended to use the SignalHound Python API bindings to receive a broadcast 
FM stream (from 88-108Mhz) in North America, demodulate the FM stream, and play 
it back to the default Linux audio device.

You can see displayed the frequency being demodulated, and increase and decrease
the frequency during playback using left and right arrow keys, with the default 
inc/dec being 100KHz.
